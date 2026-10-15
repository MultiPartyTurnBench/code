"""ESPnet turn-taking model of "Talking Turns" (``espnet/Turn_taking_prediction_SWBD``) (paper Table 3).

Port of TurnBench's ``baselines/espnet_turntaking/predict.py`` (ESPnet commit 750e3749 + the
``espnet/whisper`` fork): Whisper-medium encoder over a 30 s growing/sliding window, one prediction every
40 ms (first at 0.2 s), five classes ``[C, NA, I, BC, T]``; TF32 matmuls enabled (TurnBench's ``TT_TF32=1``).

Two variants (both in the paper; the mix is the main-table row, the per-channel run is in the appendix):

* :class:`ESPnetTurnTakingPerChannel`: each isolated channel; EOT = P(T), INT = P(I).
* :class:`ESPnetTurnTakingMix`: the mono *mix* of all N channels; scores are attributed to speakers with
  TurnBench's energy rule generalised to N: ``a_k`` = speaker k's share of the frame energy, ``hold_k`` = mean ``a_k`` over
the previous 0.48 s, EOT_k = P(T) * hold_k, INT_k = P(I) * clip(a_k - hold_k, 0, 1) * (sum of the other
speakers' holds). INT uses the native Interruption class, gated by other-speaker activity at decode time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

HF_REPO = "espnet/Turn_taking_prediction_SWBD"
EXP_DIR = "exp/asr_train_asr_whisper_turn_taking_raw_en_word"


def _smooth_prev(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of the previous ``w`` values (excluding the current one); 0 for the first frame."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    return (c[i] - c[lo]) / np.maximum(1, i - lo)


class _ESPnetTurnTaking:
    sr = 16000
    START, HOP, WIN, BATCH, W = 3200, 640, 480000, 8, 12

    def __init__(self):
        import torch
        from espnet2.tasks.slu import SLUTask
        from huggingface_hub import snapshot_download
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        exp = Path(snapshot_download(HF_REPO)) / EXP_DIR
        self.m, _ = SLUTask.build_model_from_file(str(exp / "config.yaml"), str(exp / "valid.loss.ave.pth"), "cuda")
        self.m.eval()

    def probs(self, sig: np.ndarray) -> np.ndarray:
        """``(n, 5)`` class probabilities of a mono 16 kHz signal, one row per 40 ms hop."""
        torch, m = self.torch, self.m
        START, HOP, WIN, BATCH = self.START, self.HOP, self.WIN, self.BATCH
        sp = torch.from_numpy(np.ascontiguousarray(sig, np.float32)).cuda()
        n = (len(sig) - START) // HOP
        out = np.empty((max(n, 0), 5), np.float32)

        def heads(enc, olens):
            feats = m.transform_mean(m.act_fn(enc))
            last = torch.stack([feats[k, olens[k] - 1] for k in range(feats.shape[0])])
            return torch.softmax(m.transform_linear(last), dim=-1)

        with torch.no_grad():
            i = 0
            while i < n:  # growing window until it reaches 30 s
                end = (i + 1) * HOP + START
                if end - max(0, end - WIN) >= WIN:
                    break
                w = sp[max(0, end - WIN):end].unsqueeze(0)
                enc, ol = m.encode(w, torch.tensor([w.shape[1]], device="cuda"))
                out[i] = heads(enc, ol)[0].float().cpu().numpy()
                i += 1
            while i < n:  # full 30 s sliding windows, batched
                js = list(range(i, min(i + BATCH, n)))
                wins = [sp[(j + 1) * HOP + START - WIN:(j + 1) * HOP + START] for j in js]
                L = torch.full((len(js),), WIN, dtype=torch.long, device="cuda")
                enc, ol = m.encode(torch.stack(wins), L)
                out[js[0]:js[-1] + 1] = heads(enc, ol).float().cpu().numpy()
                i = js[-1] + 1
        return out

    def times(self, n: int) -> np.ndarray:
        return (self.START + (np.arange(n) + 1) * self.HOP) / self.sr

    def energy(self, sig: np.ndarray, n: int) -> np.ndarray:
        """Per-hop RMS of one channel on the prediction grid."""
        e = np.zeros(n, np.float32)
        for i in range(n):
            seg = sig[self.START + i * self.HOP:self.START + (i + 1) * self.HOP]
            if len(seg):
                e[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        return e



class ESPnetTurnTakingPerChannel(_ESPnetTurnTaking):
    """Each isolated channel: EOT = P(T), INT = P(I) (native Interruption class)."""

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for s, x in ch.items():
            p = self.probs(x)
            out[s] = {"t": self.times(len(p)), "eot": p[:, 4], "int": p[:, 2]}
        return out


class ESPnetTurnTakingMix(_ESPnetTurnTaking):
    """Mono mix of all channels with energy attribution."""

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        spk = list(ch)
        p = self.probs(np.sum([ch[s] for s in spk], axis=0).astype(np.float32))
        n = len(p)
        E = np.stack([self.energy(ch[s], n) for s in spk])
        tot = E.sum(0) + 1e-8
        A = E / tot * (tot > 0.25 * np.median(tot))
        H = np.stack([_smooth_prev(a, self.W) for a in A])
        t = self.times(n)
        out = {}
        for k, s in enumerate(spk):
            onset = np.clip(A[k] - H[k], 0, 1) * (H.sum(0) - H[k])
            out[s] = {"t": t, "eot": (p[:, 4] * H[k]).astype(np.float32),
                      "int": (p[:, 2] * onset).astype(np.float32)}
        return out
