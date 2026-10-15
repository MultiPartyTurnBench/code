"""VAP (Voice Activity Projection; Ekstedt & Skantze, 2022) adapted to N speakers (paper Table 3).

VAP is a dyadic stereo model that predicts, per 50 Hz frame, ``p_now`` (probability each of two parties
holds the floor in the next 0-0.4 s) and per-party voice activity. For N-party audio each speaker ``s``
is run as "s vs rest": channel 0 = s, channel 1 = the average of all other channels. Decoding follows
TurnBench's official VAP baseline:

* EOT[s] = rising edge of ``1 - p_now[s] > theta`` (s releases the floor);
* INT[s] = rising edge of ``p_now[s] > theta`` while the rest is voice-active (VAP's own VAD head,
  ``vad[rest] > 0.5``); 2 s refractory. VAP has no interruption output: INT is a derived score.

Long audio is processed with a causal 20 s context / 5 s step sliding window. The checkpoint is
TurnBench's fine-tuned ``oto.ckpt`` (``viks66/VAP_checkpoints``). The model code comes from
https://github.com/ErikEkstedt/VoiceActivityProjection (commit f39a78b); pass its checkout with
``vap_repo`` or ``$VAP_REPO``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

from .common import refractory, resample_16k, rising_edges

CKPT_REPO, CKPT_FILE, CKPT_REVISION = "viks66/VAP_checkpoints", "oto.ckpt", "b9aa0ba1718221153e04ef343c7f3b8cf84bfbc3"
THETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CONTEXT_S, STEP_S = 20, 5


def load_vap(device: str = "cuda", vap_repo: str = None, checkpoint: str = None):
    """Build ``VapGPT`` from the VAP repo and load the oto checkpoint. Returns ``(model, device)``."""
    import torch
    vap_repo = vap_repo or os.environ.get("VAP_REPO")
    if not vap_repo:
        raise ValueError("VAP code not found: pass vap_repo=... or set VAP_REPO to a VoiceActivityProjection checkout")
    if vap_repo not in sys.path:
        sys.path.insert(0, vap_repo)
    from vap.model import VapConfig, VapGPT
    # VAP's utils enable deterministic algorithms, which breaks CuBLAS matmuls; not needed for inference.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(CKPT_REPO, CKPT_FILE, revision=CKPT_REVISION)
    model = VapGPT(VapConfig())
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:  # Lightning checkpoint: strip the 'net.' prefix
        sd = {k.replace("net.", ""): v for k, v in sd["state_dict"].items() if "VAP.codebook" not in k}
    model.load_state_dict(sd, strict=False)
    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    return model.eval().to(dev), dev


def step_probs(model, wav, device, context_time: int = CONTEXT_S, step_time: int = STEP_S) -> dict:
    """Causal sliding-window VAP outputs over a long stereo waveform ``[1, 2, N]`` -> dict of ``[1, T, 2]``."""
    import torch
    sr, fhz = model.sample_rate, model.frame_hz
    n = wav.shape[-1]
    chunk_s = int((context_time + step_time) * sr)
    step_s = int(step_time * sr)
    step_f = int(step_time * fhz)
    if n <= chunk_s:
        with torch.no_grad():
            return model.probs(wav.to(device))
    folds = wav.unfold(-1, chunk_s, step_s).permute(2, 0, 1, 3)
    with torch.no_grad():
        out = model.probs(folds[0].to(device))
        for w in folds[1:]:
            o = model.probs(w.to(device))
            for k in ("vad", "p_now", "p_future"):
                out[k] = torch.cat([out[k], o[k][:, -step_f:]], dim=1)
        expected = round(n / sr * fhz)
        got = out["p_now"].shape[1]
        if expected > got:
            o = model.probs(wav[..., -chunk_s:].to(device))
            for k in ("vad", "p_now", "p_future"):
                out[k] = torch.cat([out[k], o[k][:, -(expected - got):]], dim=1)
    return out


class VAP:
    """VAP over the theta grid (``p_now`` computed once per speaker, thresholded per theta)."""

    thetas = THETAS

    def __init__(self, device: str = "cuda", vap_repo: str = None, checkpoint: str = None):
        self.model, self.dev = load_vap(device, vap_repo, checkpoint)

    def predict(self, audio: Dict[str, np.ndarray], sr: int, vad_thresh: float = 0.5
                ) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
        import torch
        speakers = list(audio)
        ch16 = {s: resample_16k(a, sr) for s, a in audio.items()}
        n = min(len(x) for x in ch16.values())
        ch16 = {s: x[:n] for s, x in ch16.items()}
        per_spk = {}
        for s in speakers:
            rest = np.zeros(n, dtype=np.float32)
            for o in speakers:
                if o != s:
                    rest += ch16[o]
            if len(speakers) > 1:
                rest /= (len(speakers) - 1)
            out = step_probs(self.model, torch.from_numpy(np.stack([ch16[s], rest])[None]), self.dev)
            p_now = out["p_now"][0].cpu().numpy()
            vad = out["vad"][0].cpu().numpy()
            per_spk[s] = (np.arange(p_now.shape[0]) / self.model.frame_hz, p_now, vad)
        result = {}
        for th in self.thetas:
            conv = {}
            for s, (times, p_now, vad) in per_spk.items():
                rest_voiced = vad[:, 1] > vad_thresh
                conv[s] = {"eot": refractory(rising_edges((1.0 - p_now[:, 0]) > th, times)),
                           "interruption": refractory(rising_edges((p_now[:, 0] > th) & rest_voiced, times))}
            result[float(th)] = conv
        return result
