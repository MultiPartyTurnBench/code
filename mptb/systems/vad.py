"""Acoustic VAD baselines: RMS energy, WebRTC VAD and Silero VAD (paper Table 3, "Acoustic VAD").

Each backend produces a per-channel voiced mask on a 20 ms grid; :func:`masks_to_predictions` turns
the masks into commits:

* EOT[s] = a speech-to-silence offset on s that stays silent for ``eot_hang_s`` (committed after the
  hangover, causally);
* INT[s] = a speech onset on s while any other channel is voiced (committed after ``int_confirm_s``).

The operating point ``theta`` is backend specific (:data:`THETAS`): the energy threshold in dBFS, the
WebRTC aggressiveness (0-3) or the Silero speech-probability threshold.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .common import GRID_DT, SR16, refractory, resample_16k, webrtc_masks

THETAS: Dict[str, tuple] = {
    "energy": (-50.0, -45.0, -40.0, -35.0, -30.0),
    "webrtc": (0, 1, 2, 3),
    "silero": (0.3, 0.4, 0.5, 0.6, 0.7),
}
EOT_HANG_S = 0.3
INT_CONFIRM_S = 0.1


def masks_to_predictions(masks: Dict[str, np.ndarray], frame_dt: float = GRID_DT, *,
                         eot_hang_s: float = EOT_HANG_S, int_confirm_s: float = INT_CONFIRM_S
                         ) -> Dict[str, Dict[str, List[float]]]:
    """Per-speaker voiced masks (same grid) -> ``{speaker: {"eot": [...], "interruption": [...]}}``."""
    speakers = list(masks)
    nmin = min((len(masks[s]) for s in speakers), default=0)
    masks = {s: masks[s][:nmin] for s in speakers}
    hang = max(1, int(round(eot_hang_s / frame_dt)))
    conf = max(1, int(round(int_confirm_s / frame_dt)))
    times = np.arange(nmin) * frame_dt
    other_active = {}
    for s in speakers:
        acc = np.zeros(nmin, dtype=bool)
        for o in speakers:
            if o != s:
                acc |= masks[o]
        other_active[s] = acc

    preds: Dict[str, Dict[str, List[float]]] = {}
    for s in speakers:
        m = masks[s]
        eot_raw: List[float] = []
        int_raw: List[float] = []
        for i in range(1, nmin):
            if m[i - 1] and not m[i]:
                j = min(nmin, i + hang)
                if not m[i:j].any():
                    eot_raw.append(times[min(nmin - 1, i + hang - 1)])
            if (not m[i - 1]) and m[i] and other_active[s][i]:
                int_raw.append(times[min(nmin - 1, i + conf)])
        preds[s] = {"eot": refractory(eot_raw), "interruption": refractory(int_raw)}
    return preds


def rms_db(audio: np.ndarray, hop: int) -> np.ndarray:
    """Frame-wise RMS level in dBFS over non-overlapping frames of ``hop`` samples."""
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def energy_masks(audio: Dict[str, np.ndarray], sr: int, vad_db: float) -> Dict[str, np.ndarray]:
    """Voiced = 20 ms frame RMS above ``vad_db`` dBFS."""
    hop = max(1, int(sr * GRID_DT))
    return {s: (rms_db(a, hop) > vad_db) for s, a in audio.items()}


def _intervals_to_grid(intervals, n_frames: int) -> np.ndarray:
    """Rasterise speech ``(start_s, end_s)`` intervals onto the 20 ms grid."""
    m = np.zeros(n_frames, dtype=bool)
    for a, b in intervals:
        i0 = max(0, int(a / GRID_DT))
        i1 = min(n_frames, int(np.ceil(b / GRID_DT)))
        if i1 > i0:
            m[i0:i1] = True
    return m


def silero_masks(audio: Dict[str, np.ndarray], sr: int, threshold: float, model=None) -> Dict[str, np.ndarray]:
    """Silero VAD speech timestamps (16 kHz, processed in 120 s chunks) rasterised onto the 20 ms grid."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    m = model or load_silero_vad()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        m = m.to(dev)
    except Exception:
        dev = "cpu"
    masks = {}
    chunk = 120 * SR16
    for s, a in audio.items():
        x = resample_16k(a, sr)
        n_frames = int(len(x) / (SR16 * GRID_DT))
        intervals = []
        for off in range(0, len(x), chunk):
            seg = x[off:off + chunk]
            ts = get_speech_timestamps(torch.from_numpy(seg).to(dev), m, sampling_rate=SR16,
                                       threshold=float(threshold))
            intervals += [((off + d["start"]) / SR16, (off + d["end"]) / SR16) for d in ts]
            if dev == "cuda":
                torch.cuda.empty_cache()
        masks[s] = _intervals_to_grid(intervals, n_frames)
    return masks


class VADBaseline:
    """One acoustic VAD backend evaluated over its threshold grid."""

    def __init__(self, backend: str):
        if backend not in THETAS:
            raise ValueError(f"unknown VAD backend {backend!r}")
        self.backend = backend
        self.thetas = THETAS[backend]
        self._silero = None
        if backend == "silero":
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad()

    def masks(self, audio: Dict[str, np.ndarray], sr: int, theta) -> Dict[str, np.ndarray]:
        if self.backend == "energy":
            return energy_masks(audio, sr, vad_db=float(theta))
        if self.backend == "webrtc":
            return webrtc_masks(audio, sr, aggressiveness=int(theta))
        return silero_masks(audio, sr, threshold=float(theta), model=self._silero)

    def predict(self, audio: Dict[str, np.ndarray], sr: int) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
        """``{theta: {speaker: {"eot", "interruption"}}}`` for every theta of the backend's grid."""
        return {float(th): masks_to_predictions(self.masks(audio, sr, th)) for th in self.thetas}
