"""Shared building blocks of the offline baselines: causal commit rule, audio helpers, activity gates.

Every system turns per-speaker scores (or voiced masks) into *committed* decision times with the
benchmark's commit rule: one event per rising edge of the decision condition, followed by a 2 s
refractory period (paper Section 4, "Prediction and Event Matching"). Times are rounded to 1 ms.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

REFRACTORY_S = 2.0
GRID_DT = 0.02          # 20 ms voice-activity grid (WebRTC / energy / Silero masks)
SR16 = 16000


def refractory(times: Sequence[float], refractory_s: float = REFRACTORY_S) -> List[float]:
    """Keep a commit only if at least ``refractory_s`` seconds passed since the last kept one."""
    out: List[float] = []
    last = -1e9
    for t in sorted(times):
        if t - last >= refractory_s:
            out.append(round(float(t), 3))
            last = t
    return out


def rising_edges(cond: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Times at which a boolean condition switches from False to True (the first frame counts if True)."""
    c = np.asarray(cond, bool)
    prev = np.concatenate([[False], c[:-1]])
    return np.asarray(times)[c & ~prev]


def resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample a mono float signal to 16 kHz with torchaudio (identity if already 16 kHz)."""
    if sr == SR16:
        return audio.astype(np.float32)
    return resample(audio, sr, SR16)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float signal with ``torchaudio.functional.resample``."""
    import torch
    import torchaudio.functional as AF
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    return AF.resample(x, sr, target_sr).numpy()


def equal_length_channels(audio: Dict[str, np.ndarray], sr: int, target_sr: int = None
                          ) -> Dict[str, np.ndarray]:
    """Resample every channel to ``target_sr`` (None = keep ``sr``) and trim all to the shortest one."""
    if target_sr and sr != target_sr:
        audio = {s: resample(a, sr, target_sr) for s, a in audio.items()}
    n = min(len(a) for a in audio.values())
    return {s: np.ascontiguousarray(a[:n], dtype=np.float32) for s, a in audio.items()}


def webrtc_masks(audio: Dict[str, np.ndarray], sr: int, aggressiveness: int = 2) -> Dict[str, np.ndarray]:
    """Per-channel WebRTC VAD decisions on 20 ms frames of 16 kHz int16 PCM (one bool per frame)."""
    import webrtcvad
    vad = webrtcvad.Vad(int(aggressiveness))
    masks = {}
    for s, a in audio.items():
        x = resample_16k(a, sr)
        pcm = np.clip(x * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        fbytes = int(SR16 * GRID_DT) * 2
        flags = [vad.is_speech(pcm[off:off + fbytes], SR16) for off in range(0, len(pcm) - fbytes + 1, fbytes)]
        masks[s] = np.array(flags, dtype=bool)
    return masks


def rest_mix(audio: Dict[str, np.ndarray], s: str) -> np.ndarray:
    """Second channel of the "s vs rest" reduction: the mean of all other speakers' channels."""
    others = [audio[o] for o in audio if o != s]
    return np.mean(others, axis=0).astype(np.float32) if others else np.zeros_like(audio[s])


def rest_active(masks: Dict[str, np.ndarray], s: str, t: np.ndarray, dt: float) -> np.ndarray:
    """For frames committing at times ``t`` (frame span ``(t - dt, t]``), whether any *other* speaker is
    voiced (20 ms WebRTC masks) anywhere in the span. Used to gate INT: an interruption needs another
    speaker holding the floor."""
    others = [m for o, m in masks.items() if o != s]
    if not others:
        return np.zeros(len(t), bool)
    L = max(len(m) for m in others)
    anyo = np.zeros(L + 1, bool)
    for m in others:
        anyo[:len(m)] |= m
    c = np.concatenate([[0], np.cumsum(anyo)])
    lo = np.clip(np.floor((t - dt) / GRID_DT + 1e-9).astype(int), 0, L)
    hi = np.clip(np.ceil(t / GRID_DT - 1e-9).astype(int), 0, L)
    hi = np.maximum(hi, lo + 1)
    hi = np.minimum(hi, L)
    return (c[hi] - c[lo]) > 0
