"""Mimi Endpointer (``viks66/mimi-endpointer``, checkpoint ``oto_d1f.pt``) (paper Table 3).

A two-stream LSTM over Mimi features from TurnBench. Run as "s vs rest" (stream 1 = s, stream 2 = the
mean of the other speakers); class 4 is P(user). EOT = 1 - P(user), INT = P(user) (TurnBench mapping;
INT is a derived floor-holding score, gated by other-speaker activity at decode time). Frames are
25 Hz and commit at their end. The model code is TurnBench's ``baselines/mimi_endpointer/model.py``
(https://github.com/SesameAILabs/turnbench, commit 38a6f87); pass the checkout with ``turnbench_repo``
or ``$TURNBENCH_REPO``.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np

from .common import rest_mix

HF_REPO, CKPT = "viks66/mimi-endpointer", "oto_d1f.pt"
FRAME_HZ = 25.0


def frame_end_times(n: int, hz: float) -> np.ndarray:
    """Commit time of frame i = the end of the audio it has seen, ``(i + 1) / hz`` (TurnBench convention)."""
    return (np.arange(n) + 1) / hz


class MimiEndpointer:
    sr = 24000

    def __init__(self, turnbench_repo: str = None):
        import torch
        repo = turnbench_repo or os.environ.get("TURNBENCH_REPO")
        if not repo:
            raise ValueError("TurnBench code not found: pass turnbench_repo=... or set TURNBENCH_REPO")
        sys.path.insert(0, os.path.join(repo, "baselines", "mimi_endpointer"))
        from huggingface_hub import hf_hub_download
        from model import AUDIO_DEFAULTS, AudioFeatureExtractor, load_model
        self.torch = torch
        self.model = load_model(hf_hub_download(HF_REPO, CKPT), device="cuda")
        self.ext = AudioFeatureExtractor(**AUDIO_DEFAULTS, device="cuda")

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        torch = self.torch
        out = {}
        with torch.no_grad():
            feats = {s: self.ext(x) for s, x in ch.items()}
            for s in ch:
                fr = self.ext(rest_mix(ch, s))
                T = min(feats[s].shape[-1], fr.shape[-1])
                x = torch.stack([feats[s][:, :, :T], fr[:, :, :T]], dim=1)
                p = torch.softmax(self.model.infer(x)[0], -1).float().cpu().numpy()
                pu = p[:, 4]
                out[s] = {"t": frame_end_times(T, FRAME_HZ), "eot": 1.0 - pu, "int": pu}
        return out
