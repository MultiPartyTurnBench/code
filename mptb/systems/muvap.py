"""MuVAP (Qi & Skantze, 2026), audio-only VAP release ``vap-speaker-mimi`` (paper Table 3).

Weights: ``Haotian-Qi/MuVAP`` (folder ``vap-speaker-mimi``: Mimi encoder, 25 Hz, speaker-based
projection codebook, trained on Fisher); model code: https://github.com/Haotian-Qi/MuVAP (commit 0fe3c66),
passed with ``muvap_repo`` or ``$MUVAP_REPO``. The full MuVAP model also needs face video, so the
audio-only release is used.

Decoding is VAP's (:mod:`mptb.systems.vap`) with one difference: MuVAP has no voice-activity head, so
"rest active" is WebRTC VAD (aggressiveness 2) OR-ed over the other speakers' channels.

* per speaker s: channel 0 = s, channel 1 = mean of the other speakers (16 kHz);
* 20 s context / 5 s step causal sliding window; ``p_now`` from ``ProjectionWindow.get_probs``;
* EOT[s] = rising edge of ``1 - p_now[s] > theta``; INT[s] = rising edge of ``p_now[s] > theta`` AND
  rest-active; 2 s refractory. INT is a derived score (floor-taking projection while others speak).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

from .common import SR16, refractory, resample_16k, rising_edges, webrtc_masks

HF_REPO, RELEASE = "Haotian-Qi/MuVAP", "vap-speaker-mimi"
THETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CTX_S, STEP_S = 20, 5


def rest_gate(masks: Dict[str, np.ndarray], s: str, n_frames: int, fhz: float) -> np.ndarray:
    """20 ms WebRTC masks -> per model frame "any other speaker active" (frame i spans [i/fhz, (i+1)/fhz))."""
    others = [m for o, m in masks.items() if o != s]
    L = max(len(m) for m in others)
    anyo = np.zeros(L, bool)
    for m in others:
        anyo[:len(m)] |= m
    per = int(round(0.02 ** -1 / fhz))
    g = np.zeros(n_frames, bool)
    for i in range(n_frames):
        g[i] = anyo[i * per:(i + 1) * per].any()
    return g


class MuVAP:
    """MuVAP over the theta grid."""

    thetas = THETAS

    def __init__(self, muvap_repo: str = None, weights_dir: str = None):
        import yaml
        muvap_repo = muvap_repo or os.environ.get("MUVAP_REPO")
        if not muvap_repo:
            raise ValueError("MuVAP code not found: pass muvap_repo=... or set MUVAP_REPO to a MuVAP checkout")
        sys.path.insert(0, muvap_repo)
        from models.release import load_weights, resolve
        from models.vap import build_vap
        from projection_window import ProjectionWindow
        if weights_dir is None:
            from huggingface_hub import snapshot_download
            weights_dir = os.path.join(snapshot_download(HF_REPO, allow_patterns=[f"{RELEASE}/*"]), RELEASE)
        w, c = resolve(weights_dir)
        cfg = yaml.safe_load(open(c))["vap"]
        self.model = load_weights(build_vap(cfg), w).eval().cuda()
        self.pw = ProjectionWindow(**cfg["projection_window"])
        self.fhz = float(cfg["projection_window"]["frame_hz"])

    def p_now(self, wav) -> np.ndarray:
        """Sliding-window ``p_now`` over a ``[2, N]`` 16 kHz tensor (same fold/stitch logic as VAP)."""
        import torch
        n = wav.shape[-1]
        chunk, step = (CTX_S + STEP_S) * SR16, STEP_S * SR16
        step_f = int(STEP_S * self.fhz)

        def run(x):
            with torch.no_grad():
                return self.pw.get_probs(self.model(x[None].cuda()))["p_now"][0].float().cpu()

        if n <= chunk:
            return run(wav).numpy()
        out = run(wav[:, :chunk])
        start = step
        while start + chunk <= n:
            out = torch.cat([out, run(wav[:, start:start + chunk])[-step_f:]])
            start += step
        expected = round(n / SR16 * self.fhz)
        if expected > out.shape[0]:
            out = torch.cat([out, run(wav[:, -chunk:])[-(expected - out.shape[0]):]])
        return out.numpy()

    def predict(self, audio: Dict[str, np.ndarray], sr: int) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
        import torch
        spk = list(audio)
        ch = {s: resample_16k(a, sr) for s, a in audio.items()}
        n = min(len(x) for x in ch.values())
        ch = {s: x[:n] for s, x in ch.items()}
        masks = webrtc_masks(ch, SR16, aggressiveness=2)
        res = {float(t): {} for t in self.thetas}
        for s in spk:
            rest = np.mean([ch[o] for o in spk if o != s], axis=0).astype(np.float32)
            ps = self.p_now(torch.from_numpy(np.stack([ch[s], rest])))[:, 0]
            gate = rest_gate(masks, s, len(ps), self.fhz)
            times = np.arange(len(ps)) / self.fhz
            for th in self.thetas:
                res[float(th)][s] = {"eot": refractory(rising_edges((1.0 - ps) > th, times)),
                                     "interruption": refractory(rising_edges((ps > th) & gate, times))}
        return res
