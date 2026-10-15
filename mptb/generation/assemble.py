"""Channel assembly (paper Section 3.1 steps 4-5).

Renders one mono channel per speaker: each placed clip from Stage 2 is synthesised (Stage 3)
and written onto its speaker's silent track at its absolute start time; clips truncated by
an interruption are hard-cut to their placed duration with a 30 ms linear fade-out; each
channel is then peak-normalised to -1 dBFS independently. The resulting float32 buffers are
the clean channels (written as 16-bit FLAC) and the input to the Stage-4 noise mixer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .stage2_orchestrate import Clip


def _peak_normalize(x: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Scale ``x`` so its peak is ``target_dbfs`` (unchanged if silent)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-9:
        return x
    target = 10 ** (target_dbfs / 20.0)
    return x * (target / peak)


def _fade_out(x: np.ndarray, sr: int, ms: float = 30.0) -> np.ndarray:
    """Linear fade to zero over the last ``ms`` milliseconds."""
    n = min(len(x), int(sr * ms / 1000.0))
    if n > 0:
        x = x.copy()
        x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return x


def assemble_channels(clips: List[Clip], voice_map: Dict[str, str], tts,
                      speakers: List[str], duration_s: float,
                      sample_rate: int = 24000) -> Dict[str, np.ndarray]:
    """Render ``{speaker: float32 channel}`` of ``round(duration_s * sample_rate)`` samples.

    ``tts`` must expose ``synth(text, voice_id)`` returning an object with ``.audio``.
    Clips are placed additively (Stage 2 guarantees no same-speaker overlap); truncated
    clips are cut to their placed duration with a fade-out.
    """
    n = int(round(duration_s * sample_rate)) + sample_rate  # +1s safety pad
    buffers = {s: np.zeros(n, dtype=np.float32) for s in speakers}
    for c in clips:
        ta = tts.synth(c.text, voice_map[c.speaker])
        audio = ta.audio.astype(np.float32)
        if c.truncated:
            target = max(1, int(round(c.dur * sample_rate)))
            if len(audio) > target:
                audio = _fade_out(audio[:target], sample_rate)
        start = int(round(c.start_s * sample_rate))
        end = start + len(audio)
        if end > len(buffers[c.speaker]):
            end = len(buffers[c.speaker])
            audio = audio[: end - start]
        buffers[c.speaker][start:end] += audio
    # trim all to a common length and normalize per channel
    out_len = int(round(duration_s * sample_rate))
    for s in speakers:
        buffers[s] = _peak_normalize(buffers[s][:out_len])
    return buffers


def write_channels(buffers: Dict[str, np.ndarray], out_dir: str, conversation_id: str,
                   sample_rate: int = 24000) -> Dict[str, str]:
    """Write ``<out_dir>/<id>_<speaker>.flac`` (16-bit PCM) per channel; return {speaker: file name}."""
    import soundfile as sf
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    for s, buf in buffers.items():
        fname = f"{conversation_id}_{s}.flac"
        sf.write(str(Path(out_dir) / fname), buf, sample_rate, subtype="PCM_16")
        files[s] = fname
    return files
