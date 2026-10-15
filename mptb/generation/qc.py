"""Acoustic quality control of rendered conversations (``metadata/qc_report.json``).

For each conversation the clean per-speaker channels are checked against the placement plan
(``render``) and the gold event log with a 10 ms energy VAD (-40 dBFS on the peak-normalised
channel):

* Q1 duration        -- channel length matches the gold duration (within 1.6 s);
* Q2 speech ratio    -- each channel is neither near-silent (< 3 %) nor saturated (> 99 %);
* Q3 clip coverage   -- at least 50 % of the frames inside a speaker's placed clips are voiced;
* Q4 leakage         -- at most 30 % of a speaker's voiced frames fall outside its clips;
* Q5 overlap realism -- at >= 60 % of interruptions, failed claims and backchannels both the
  mover and the current holder are voiced (>= 30 % of frames) in the overlap window.

QC runs on the clean channels (the noise bed would trip the leakage check).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

FRAME_MS = 10.0
VAD_DB = -40.0          # dBFS threshold on peak-normalized audio
DUR_TOL_S = 0.6
SPEECH_RATIO_MIN = 0.03
SPEECH_RATIO_MAX = 0.99
# Turns with natural inter-sentence pauses voice only ~0.5 of their placed span.
CLIP_COVERAGE_MIN = 0.50
LEAKAGE_MAX = 0.30
OVERLAP_MIN_FRAC = 0.30


def _vad_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per-10 ms-frame voiced mask (RMS above VAD_DB)."""
    hop = max(1, int(sr * FRAME_MS / 1000.0))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=bool)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    return db > VAD_DB


def _mask_ratio_in(mask: np.ndarray, sr: int, lo: float, hi: float) -> Tuple[float, int]:
    """Fraction of frames in [lo,hi] that are voiced, and #frames in range."""
    i0 = max(0, int(lo * 1000.0 / FRAME_MS))
    i1 = min(len(mask), int(np.ceil(hi * 1000.0 / FRAME_MS)))
    if i1 <= i0:
        return 0.0, 0
    seg = mask[i0:i1]
    return float(seg.mean()), int(i1 - i0)


def qc_conversation(gold: Dict[str, Any], render: Dict[str, Any], audio_dir: str) -> Dict[str, Any]:
    """QC one conversation; channels are read from ``<audio_dir>/<gold.audio.files[speaker]>``."""
    import soundfile as sf
    speakers = gold["speakers"]
    sr = render.get("sample_rate", gold.get("metadata", {}).get("sample_rate", 24000))
    files = gold["audio"]["files"]
    audio: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for s in speakers:
        a, file_sr = sf.read(str(Path(audio_dir) / files[s]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        audio[s] = a
        masks[s] = _vad_mask(a, file_sr)
        sr = file_sr

    checks: List[Dict[str, Any]] = []
    fails: List[str] = []

    # Q1 duration
    max_len = max(len(a) for a in audio.values()) / sr
    dur_ok = abs(max_len - gold["duration_s"]) <= DUR_TOL_S + 1.0
    if not dur_ok:
        fails.append(f"Q1 duration {max_len:.1f}s vs gold {gold['duration_s']:.1f}s")

    # clips grouped per speaker
    clips_by_spk: Dict[str, List[Tuple[float, float]]] = {s: [] for s in speakers}
    for c in render["clips"]:
        clips_by_spk[c["speaker"]].append((c["start_s"], c["end_s"]))

    # Q2 speech ratio, Q3 coverage, Q4 leakage
    for s in speakers:
        m = masks[s]
        ratio = float(m.mean()) if len(m) else 0.0
        if not (SPEECH_RATIO_MIN <= ratio <= SPEECH_RATIO_MAX):
            fails.append(f"Q2 {s} speech_ratio={ratio:.3f} out of range")
        # coverage within clips
        cov_voiced = cov_total = 0
        clip_frames = np.zeros(len(m), dtype=bool)
        for (a0, b0) in clips_by_spk[s]:
            r, nf = _mask_ratio_in(m, sr, a0, b0)
            cov_voiced += r * nf
            cov_total += nf
            i0 = max(0, int(a0 * 1000 / FRAME_MS)); i1 = min(len(m), int(np.ceil(b0 * 1000 / FRAME_MS)))
            clip_frames[i0:i1] = True
        coverage = cov_voiced / cov_total if cov_total else 1.0
        if coverage < CLIP_COVERAGE_MIN:
            fails.append(f"Q3 {s} clip_coverage={coverage:.3f} < {CLIP_COVERAGE_MIN}")
        # leakage: voiced frames outside any clip / total voiced
        voiced_total = int(m.sum())
        voiced_out = int((m & ~clip_frames).sum())
        leakage = voiced_out / voiced_total if voiced_total else 0.0
        if leakage > LEAKAGE_MAX:
            fails.append(f"Q4 {s} leakage={leakage:.3f} > {LEAKAGE_MAX}")
        checks.append({"speaker": s, "speech_ratio": round(ratio, 3),
                       "clip_coverage": round(coverage, 3), "leakage": round(leakage, 3)})

    # Q5 overlap realism at interruptions + vocalizations
    overlap_events = 0
    overlap_ok = 0
    for ev in gold["events"]:
        holder = ev.get("current_holder")
        mover = ev.get("speaker")
        if holder is None or mover == holder or mover not in masks or holder not in masks:
            continue
        if ev["label"] == "CLAIM_FLOOR" and ev.get("resolved_as") == "INTERRUPTION":
            lo, hi = ev["onset"], ev.get("floor_transfer_time", ev["onset"] + 0.5)
        elif ev["label"] in ("VOCALIZE_NO_CLAIM",) or (
                ev["label"] == "CLAIM_FLOOR" and ev.get("resolved_as") == "NON_FLOOR_TAKING_INTERRUPTION"):
            lo, hi = ev["onset"], ev.get("offset", ev["onset"] + 0.3)
        else:
            continue
        overlap_events += 1
        rm, _ = _mask_ratio_in(masks[mover], sr, lo, hi)
        rh, _ = _mask_ratio_in(masks[holder], sr, max(0.0, lo - 0.1), hi)
        if rm >= OVERLAP_MIN_FRAC and rh >= OVERLAP_MIN_FRAC:
            overlap_ok += 1
    overlap_rate = overlap_ok / overlap_events if overlap_events else None
    if overlap_rate is not None and overlap_rate < 0.6:
        fails.append(f"Q5 overlap_realism={overlap_rate:.2f} ({overlap_ok}/{overlap_events})")

    return {
        "conversation_id": gold["conversation_id"],
        "num_speakers": gold["num_speakers"],
        "duration_measured_s": round(max_len, 2),
        "duration_gold_s": gold["duration_s"],
        "per_speaker": checks,
        "overlap_events": overlap_events,
        "overlap_ok": overlap_ok,
        "overlap_rate": None if overlap_rate is None else round(overlap_rate, 3),
        "fails": fails,
        "passed": len(fails) == 0,
    }


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-conversation QC results into the ``qc_report.json`` structure."""
    results = list(results)
    n_pass = sum(1 for r in results if r["passed"])
    return {"n_conversations": len(results), "n_passed": n_pass,
            "n_failed": len(results) - n_pass, "results": results}
