"""Stage 4 -- acoustic scene rendering (paper Section 3.1 step 4, Table "Stage-4 noise model").

Given clean per-speaker channels and the conversation's noise profile
``{"ambient", "snr_db", "discrete_events", "background_speech"}`` this module overlays ONE
shared noise realization on every channel (one room picked up by N microphones):

* a procedural, voice-free ambient bed (filtered pink noise, optional mains hum, slow
  amplitude drift) at the profile SNR, floored at 20 dB;
* one MUSAN ``noise/sound-bible`` clip (<= 4 s, 12 dB below speech) per discrete event label,
  at a random time;
* optionally, 2-4 faded 1.5-3 s snippets of one MUSAN ``speech`` recording (26 dB below
  speech) as distant third-party speech.

SNRs are measured against the active-speech RMS pooled over the clean channels. All
randomness flows through one ``random.Random(seed)``; the per-conversation seeds of the
released corpus are in ``metadata/noise_seeds.json``. MUSAN (OpenSLR 17) is resampled from
16 kHz to the target rate by linear interpolation. :func:`mix_noise_profile` returns new
buffers plus the ``noise`` block recorded in the render annotation; :func:`peak_limit` is
applied before writing.
"""
from __future__ import annotations

import glob
import hashlib
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MUSAN_CATEGORIES = ("noise", "speech")
_EPS = 1e-9

# Distant third-party speech level (dB below active speech).
BG_SPEECH_SNR_DB = 26.0
# Discrete event level (dB below active speech).
DISCRETE_EVENT_SNR_DB = 12.0
# The ambient bed is never mixed louder than this SNR (dB).
AMBIENT_SNR_FLOOR_DB = 20.0

# Ambient scene -> (lowpass cutoff in Hz, mains hum on/off).
AMBIENT_PROFILES = {
    "office_hum": (450, True), "office_hvac": (320, True), "quiet_office": (260, True),
    "quiet_apartment": (300, False), "dorm_room": (900, False), "cafe": (1600, False),
    "studio_room_tone": (200, False), "school_hallway": (1300, False),
    "living_room_tv_distant": (2200, False), "outdoor_park": (1100, False),
}
_DEFAULT_AMBIENT = (600, False)


def noise_seed(conversation_id: str, seed: int) -> int:
    """Default Stage-4 seed for a new conversation, from its id and spec seed (process-independent)."""
    h = int.from_bytes(hashlib.blake2b(conversation_id.encode(), digest_size=8).digest(), "big")
    return (int(seed) * 1_000_003 + h) & 0x7FFFFFFF


def synth_ambient(name: str, n: int, sr: int, rng: random.Random) -> np.ndarray:
    """Procedurally synthesize a steady, VOICE-FREE ambient bed (unit peak).

    Filtered pink noise shaped by a per-scene lowpass (+ faint 100/200 Hz mains hum
    for machine-room scenes), with a very slow (<0.2 Hz) amplitude drift for
    naturalness. No syllabic modulation, so it can never be mistaken for speech."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    cutoff, hum = AMBIENT_PROFILES.get(name, _DEFAULT_AMBIENT)
    # white -> spectrum -> pink (1/sqrt f) * gentle 2nd-order lowpass -> time
    npr = np.random.default_rng(rng.randint(0, 2**31 - 1))
    X = np.fft.rfft(npr.standard_normal(n).astype(np.float32))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    shape = (1.0 / np.sqrt(freqs)) / (1.0 + (freqs / cutoff) ** 2)
    x = np.fft.irfft(X * shape, n).astype(np.float32)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak
    if hum:
        t = np.arange(n, dtype=np.float32) / sr
        x = x + 0.12 * np.sin(2 * np.pi * 100.0 * t) + 0.06 * np.sin(2 * np.pi * 200.0 * t)
    # slow amplitude drift (natural, non-syllabic)
    t = np.arange(n, dtype=np.float32) / sr
    x = x * (1.0 + 0.12 * np.sin(2 * np.pi * 0.07 * t + npr.uniform(0, 6.28)))
    peak = float(np.max(np.abs(x))) or 1.0
    return (x / peak).astype(np.float32)


# ---------------------------------------------------------------------------
# MUSAN indexing + IO
# ---------------------------------------------------------------------------
def index_musan(musan_root: str) -> Dict[str, List[str]]:
    """Return ``{"noise": [wav paths], "speech": [wav paths]}`` (sorted) under a MUSAN root."""
    idx: Dict[str, List[str]] = {}
    for cat in MUSAN_CATEGORIES:
        idx[cat] = sorted(glob.glob(os.path.join(musan_root, cat, "**", "*.wav"),
                                    recursive=True))
    if not any(idx.values()):
        raise FileNotFoundError(f"no MUSAN wavs under {musan_root} (expected noise/ and speech/)")
    return idx


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resampling."""
    if src_sr == dst_sr or x.size == 0:
        return x
    n_dst = int(round(len(x) * dst_sr / src_sr))
    if n_dst <= 1:
        return np.zeros(max(n_dst, 0), dtype=np.float32)
    src_t = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    dst_t = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def load_clip(path: str, target_sr: int) -> np.ndarray:
    """Load a wav as mono float32 at target_sr (linear resample; downmix stereo)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return _resample_linear(np.asarray(x, dtype=np.float32), sr, target_sr)


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    """Root-mean-square of ``x`` (0 for empty input)."""
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _active_rms(x: np.ndarray, floor_db: float = -50.0) -> float:
    """RMS over the audible part only. For sparse/transient clips (a bell padded with
    silence) whole-clip RMS is tiny, so matching it to a target over-amplifies the
    transient; measuring the active part keeps the audible level controlled."""
    floor = 10 ** (floor_db / 20.0)
    a = x[np.abs(x) > floor]
    return rms(a) if a.size else rms(x)


def active_speech_rms(buffers: Dict[str, np.ndarray], thresh_db: float = -50.0) -> float:
    """RMS over samples above a small absolute floor, pooled across channels —
    an estimate of speech level that ignores the long silences on each channel."""
    floor = 10 ** (thresh_db / 20.0)
    parts = [b[np.abs(b) > floor] for b in buffers.values()]
    parts = [p for p in parts if p.size]
    if not parts:
        return 0.0
    return rms(np.concatenate(parts))


def scale_to_snr(noise: np.ndarray, signal_rms: float, snr_db: float) -> np.ndarray:
    """Scale `noise` so signal_rms / rms(scaled_noise) == 10**(snr_db/20)."""
    nrms = rms(noise)
    if nrms < _EPS or signal_rms < _EPS:
        return noise * 0.0
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    return noise * (target_noise_rms / nrms)


def peak_limit(buffers: Dict[str, np.ndarray], target_dbfs: float = -1.0
               ) -> Dict[str, np.ndarray]:
    """Scale ALL channels by ONE common gain so the loudest sample across channels
    hits target_dbfs, only if it currently exceeds it. A shared gain preserves both
    inter-channel level relationships and the shared bed. Applied after
    mix_noise_profile and before writing PCM (the additive overlay can exceed 1.0)."""
    peak = max((float(np.max(np.abs(b))) if b.size else 0.0) for b in buffers.values())
    target = 10 ** (target_dbfs / 20.0)
    if peak <= target or peak < _EPS:
        return buffers
    g = target / peak
    return {s: (b * g).astype(np.float32) for s, b in buffers.items()}


# ---------------------------------------------------------------------------
# Component overlays (operate on a length-n bed, added to channels by caller)
# ---------------------------------------------------------------------------
def build_ambient_bed(ambient_name: str, n: int, sr: int,
                      signal_rms: float, snr_db: float, rng: random.Random,
                      manifest: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Procedural, voice-free ambient bed at the given SNR (see synth_ambient)."""
    bed = synth_ambient(ambient_name, n, sr, rng)
    bed = scale_to_snr(bed, signal_rms, snr_db)
    if manifest is not None:
        manifest["ambient"] = ambient_name
        manifest["ambient_source"] = "procedural"
        manifest["ambient_snr_db"] = round(snr_db, 1)
    return bed


def build_discrete_events(musan: Dict[str, List[str]], n: int, sr: int,
                          signal_rms: float, event_labels: List[str],
                          rng: random.Random, per_event_snr_db: float = DISCRETE_EVENT_SNR_DB,
                          manifest: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """One MUSAN point-source clip per event label, placed at a random time.
    The clip pool is MUSAN ``noise/sound-bible`` (falling back to ``free-sound``)."""
    bed = np.zeros(n, dtype=np.float32)
    # Prefer sound-bible (curated short SFX: bells, knocks, chimes — matches the
    # discrete-event labels); fall back to free-sound only if sound-bible is absent.
    pool = ([p for p in musan["noise"] if "sound-bible" in p]
            or [p for p in musan["noise"] if "free-sound" in p] or musan["noise"])
    placed = []
    for label in event_labels:
        if not pool:
            break
        path = rng.choice(pool)
        clip = load_clip(path, sr)
        if clip.size == 0:
            continue
        # cap very long clips to <=4 s so a "discrete event" stays discrete
        max_len = min(len(clip), int(4.0 * sr))
        clip = clip[:max_len].copy()
        ar = _active_rms(clip)  # scale by audible level, not silence-padded whole-clip RMS
        if ar > _EPS:
            clip = clip * (signal_rms / (10 ** (per_event_snr_db / 20.0)) / ar)
        if n - len(clip) <= 0:
            start = 0
            clip = clip[:n]
        else:
            start = rng.randint(0, n - len(clip))
        bed[start:start + len(clip)] += clip
        placed.append({"label": label, "clip": os.path.basename(path),
                       "start_s": round(start / sr, 3)})
    if manifest is not None:
        manifest["discrete_events"] = placed
    return bed


def build_background_speech(musan: Dict[str, List[str]], n: int, sr: int,
                            signal_rms: float, rng: random.Random,
                            snr_db: float = BG_SPEECH_SNR_DB,
                            manifest: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Distant third-party speech (one MUSAN speech recording) as a false-positive stress:
    2-4 short faded snippets at random times. The SNR is measured against the snippets'
    own active energy, not diluted by the silence between them."""
    if not musan["speech"] or n <= 0:
        return np.zeros(n, dtype=np.float32)
    path = rng.choice(musan["speech"])
    clip = load_clip(path, sr)
    if clip.size == 0:
        return np.zeros(n, dtype=np.float32)
    bed = np.zeros(n, dtype=np.float32)
    n_snips = rng.randint(2, 4)
    fade = max(1, int(0.15 * sr))
    placed = []
    for _ in range(n_snips):
        seg_len = min(n, int(rng.uniform(1.5, 3.0) * sr))
        if len(clip) > seg_len:
            src = rng.randint(0, len(clip) - seg_len)
            seg = clip[src:src + seg_len].copy()
        else:
            seg = clip[:seg_len].copy()
        if len(seg) >= 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        start = 0 if n - len(seg) <= 0 else rng.randint(0, n - len(seg))
        bed[start:start + len(seg)] += seg[: n - start]
        placed.append(round(start / sr, 1))
    ar = _active_rms(bed)
    if ar > _EPS:
        target_rms = signal_rms / (10 ** (snr_db / 20.0))
        bed *= target_rms / ar
    if manifest is not None:
        manifest["background_speech_clip"] = os.path.basename(path)
        manifest["background_speech_snr_db"] = round(snr_db, 1)
        manifest["background_speech_starts_s"] = placed
    return bed


# ---------------------------------------------------------------------------
# Scene mixer
# ---------------------------------------------------------------------------
def mix_noise_profile(buffers: Dict[str, np.ndarray], noise_profile: Dict[str, Any],
                      musan: Dict[str, List[str]], sr: int, rng: random.Random,
                      ambient_snr_floor: float = AMBIENT_SNR_FLOOR_DB
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Overlay one shared noise realization of ``noise_profile`` on every clean channel.

    Returns ``(noisy_buffers, manifest)`` where ``manifest`` is the ``noise`` block of the
    render annotation (signal RMS, ambient scene/SNR, placed event clips, background-speech
    clip and snippet starts). Input buffers are not mutated; the result is not yet
    peak-limited (see :func:`peak_limit`). RNG draw order: ambient bed, discrete events,
    background speech.
    """
    speakers = list(buffers)
    n = len(next(iter(buffers.values()))) if speakers else 0
    signal_rms = active_speech_rms(buffers)
    ambient_name = str(noise_profile.get("ambient", "quiet_office"))
    ambient_snr = max(float(noise_profile.get("snr_db", 25.0)), ambient_snr_floor)
    manifest: Dict[str, Any] = {"shared": True, "signal_rms": round(signal_rms, 6)}

    bed = build_ambient_bed(ambient_name, n, sr, signal_rms, ambient_snr, rng, manifest)
    if noise_profile.get("discrete_events"):
        bed = bed + build_discrete_events(musan, n, sr, signal_rms,
                                          list(noise_profile["discrete_events"]), rng,
                                          manifest=manifest)
    if noise_profile.get("background_speech"):
        bed = bed + build_background_speech(musan, n, sr, signal_rms, rng, manifest=manifest)

    out = {s: (buffers[s] + bed).astype(np.float32) for s in speakers}
    return out, manifest


def add_noise(buffers: Dict[str, np.ndarray], noise_profile: Dict[str, Any],
              musan: Dict[str, List[str]], sr: int, seed: int
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Full Stage 4 for one conversation: :func:`mix_noise_profile` with ``random.Random(seed)``
    followed by :func:`peak_limit`. Returns ``(noisy_buffers, manifest)``."""
    noisy, manifest = mix_noise_profile(buffers, noise_profile, musan, sr, random.Random(seed))
    return peak_limit(noisy), manifest
