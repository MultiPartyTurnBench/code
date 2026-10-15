"""WavLM-Large ANCHOR turn-taking model (``ZhuoyanTao/causal-wavlm-turn-taking``) (paper Table 3).

Port of TurnBench's ``baselines/wavlm_large_anchor/predict.py``: frozen WavLM-Large -> 4-layer Transformer
encoder -> 12-layer autoregressive decoder over a turn-taking vocabulary (ESPnet "Universa"), checkpoint
``universa_turn_taking_only_turn_a40``. Five classes per frame: C (continue), NA, I (interrupt), BC
(backchannel), T (turn end). EOT = P(NA) + P(T); INT = 1 - P(C) (TurnBench mapping), gated by
other-speaker activity at decode time. Run per channel.

Each output frame is an independent 4 s window of 16 kHz audio that *ends* at the frame time (left
zero-padded near the start); windows advance by 40 ms and the first ends at 0.24 s. Row k is stamped at
its true causal time 0.24 + 0.04 k s (TurnBench's own grid places it 200 ms earlier).

Environment: the ESPnet Universa branch (``ftshijt/espnet`` @ ``uni_versa.sh``) with the author's
modules overlaid, shadowing any installed ``espnet`` (see TurnBench's wavlm_large_anchor README and
``requirements/anchor.txt``). TF32 matmuls are enabled, as in TurnBench's runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np

HF_REPO = "ZhuoyanTao/causal-wavlm-turn-taking"
CKPT_DIR = "universa_turn_taking_only_turn_a40"
SR = 16000
FIRST_FRAME_S = 0.20
CONTEXT_S = 4.0
STRIDE_S = 0.04
BATCH_SIZE = 64
N_CLASSES = 5
TOKEN_TO_LABELINDEX = [0, 1, 2, 4, 3]   # vocabulary order -> [C, NA, I, BC, T]
C_IDX, NA_IDX, I_IDX, BC_IDX, T_IDX = 0, 1, 2, 3, 4


def frame_times(n: int) -> np.ndarray:
    """Causal commit time (s) of output rows ``0..n-1``: the end of each row's audio window."""
    k = np.arange(n, dtype=np.int64)
    frame_end = int(FIRST_FRAME_S * SR) + (k + 1) * int(STRIDE_S * SR)
    return frame_end / SR


class WavLMAnchor:
    sr = SR

    def __init__(self, device: str = "cuda", tf32: bool = True):
        import torch
        from espnet2.tasks.universa import UniversaTask
        from huggingface_hub import hf_hub_download

        self.torch = torch
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model_dir = Path(hf_hub_download(HF_REPO, f"{CKPT_DIR}/valid.loss.best.pth")).parent
        for f in ["config.yaml", "data/metric2id", "data/metric2type", "data/tokens.json"]:
            hf_hub_download(HF_REPO, f"{CKPT_DIR}/{f}")
        cwd = os.getcwd()
        os.chdir(model_dir)  # config.yaml references tokenizer files relative to the experiment dir
        try:
            model, _ = UniversaTask.build_model_from_file(str(model_dir / "config.yaml"),
                                                          str(model_dir / "valid.loss.best.pth"), device)
        finally:
            os.chdir(cwd)
        self.model = model.to(device).eval()
        self.universa = model.universa
        self.tt_meta_idx = self.universa.metric_tokenizer.vocab_indices["turn_taking@meta_label"]
        self.tt_val0_idx = self.universa.metric_tokenizer.vocab_indices["turn_taking@0"]
        self.device = device

    def _ar_decode_batch(self, audio_enc, audio_enc_lengths) -> np.ndarray:
        """Batched greedy decode until the turn-taking meta token; returns ``(B, 5)`` class probabilities."""
        torch = self.torch
        import torch.nn.functional as F
        uni = self.universa
        B = audio_enc.shape[0]
        device = audio_enc.device
        seq = torch.full((B, 1), uni.sos, dtype=torch.long, device=device)
        seq_l = torch.ones(B, dtype=torch.long, device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)
        all_probs = torch.full((B, N_CLASSES), 1.0 / N_CLASSES, dtype=torch.float32, device=device)

        for _ in range(50):
            dec_out, _ = uni.decoder(audio_enc, audio_enc_lengths, seq, seq_l)
            next_toks = dec_out[:, -1, :].argmax(dim=-1)
            hit = (~found) & (next_toks == self.tt_meta_idx)
            if hit.any():
                hit_idx = hit.nonzero(as_tuple=True)[0]
                seq_hit = torch.cat([
                    seq[hit_idx],
                    torch.full((hit_idx.shape[0], 1), self.tt_meta_idx, dtype=torch.long, device=device),
                ], dim=1)
                dec_out2, _ = uni.decoder(audio_enc[hit_idx], audio_enc_lengths[hit_idx], seq_hit,
                                          seq_l[hit_idx] + 1)
                val_logits = dec_out2[:, -1, self.tt_val0_idx:self.tt_val0_idx + N_CLASSES]
                all_probs[hit_idx] = F.softmax(val_logits.float(), dim=-1)
                found[hit_idx] = True
            if found.all():
                break
            active = ~found & (next_toks != uni.eos)
            if not active.any():
                break
            seq = torch.cat([seq, next_toks.unsqueeze(1)], dim=1)
            seq_l = seq_l + 1

        tp = all_probs.cpu().numpy()
        lp = np.zeros_like(tp)
        for ti, li in enumerate(TOKEN_TO_LABELINDEX):
            lp[:, li] = tp[:, ti]
        return lp

    def probs_for_signal(self, sig16k: np.ndarray) -> np.ndarray:
        """Sliding-window inference -> ``(T, 5)`` float32 probabilities ``[C, NA, I, BC, T]`` at 25 Hz."""
        torch = self.torch
        sig = np.asarray(sig16k, dtype=np.float32)
        context_samples = int(CONTEXT_S * SR)
        stride_samples = int(STRIDE_S * SR)
        windows: List[np.ndarray] = []
        frame_end = int(FIRST_FRAME_S * SR) + stride_samples
        while frame_end <= len(sig):
            chunk = sig[max(0, frame_end - context_samples):frame_end]
            if len(chunk) < context_samples:
                chunk = np.concatenate([np.zeros(context_samples - len(chunk), dtype=np.float32), chunk])
            windows.append(chunk)
            frame_end += stride_samples
        if not windows:
            return np.zeros((0, 5), dtype=np.float32)

        lengths = torch.full((BATCH_SIZE,), context_samples, dtype=torch.long, device=self.device)
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(windows), BATCH_SIZE):
                batch = windows[i:i + BATCH_SIZE]
                audio_batch = torch.tensor(np.stack(batch), dtype=torch.float32).to(self.device)
                feats, feats_len = self.model._extract_feats(audio_batch, lengths[:len(batch)])
                audio_enc, enc_len = self.universa.encode(feats, feats_len)
                all_probs.extend(self._ar_decode_batch(audio_enc, enc_len))
        return np.array(all_probs, dtype=np.float32)

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for s, x in ch.items():
            p = self.probs_for_signal(x)
            out[s] = {"t": frame_times(len(p)), "eot": p[:, NA_IDX] + p[:, T_IDX], "int": 1.0 - p[:, C_IDX]}
        return out
