"""Build the audio streamed to the agents: one *pack* per scenario (Appendix H.3, Table "protocol constants").

All conversation audio is the released *noisy* evaluation audio (24 kHz); the agent hears the mono mix of
all participants and joins as an additional party. No event is ever signalled symbolically.

START packs (``true_eot`` / ``hold_pause``)
    ``stream`` = the mix from ``context_start_s`` (20 s before the target turn onset) up to the event,
    followed by

    * ``true_eot``:   :data:`~mptb.fd.scenarios.OPEN_FLOOR_S` of silence (the floor is open);
    * ``hold_pause``: the real audio through the same speaker's resume plus :data:`HOLD_TAIL_S`.

STOP packs (``true_int`` / ``backchannel``)
    ``stream`` = :data:`STOP_CONTEXT_S` of real conversation, then one of ten trivial spoken questions
    (Kokoro TTS, in a voice the conversation does not use) that makes the agent talk for a few seconds.
    ``inject`` = the real gold event cut from the target speaker's own channel; the runner plays it
    :data:`~mptb.fd.protocols.INJECT_DELAY_S` after the agent starts speaking.

Output layout: ``<out>/<scenario_id>/{stream.flac, inject.flac, pack.json}`` and ``<out>/packs.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data import Dataset

SR = 24000
AUDIO_CONDITION = "noisy"
STOP_CONTEXT_S = 12.0      # real conversation heard before the agent is addressed
PRE_PROMPT_GAP_S = 0.6
POST_PROMPT_TAIL_S = 0.4
HOLD_TAIL_S = 3.0          # audio kept after a mid-turn pause resumes
STREAM_PEAK_DBFS = -1.0
PROMPT_PEAK_DBFS = -3.0
INJECT_PEAK_DBFS = -3.0

# Trivial prompts: each yields a 4-8 s spoken answer and none requires reasoning, so the response
# content cannot confound the stop decision. Scenario i (0-based position in the scenario list)
# gets PROMPTS[i % 10].
PROMPTS = [
    "Could you briefly describe what happens when water boils?",
    "Can you give me a short description of what a rainbow is?",
    "Please briefly explain what clouds are made of.",
    "Could you tell me in a few sentences what a library is for?",
    "Can you briefly describe what happens during sunrise?",
    "Please give me a short explanation of why leaves change colour.",
    "Could you briefly say what a bicycle is used for?",
    "Can you describe in a couple of sentences what snow is?",
    "Please briefly explain what a map shows.",
    "Could you give a short description of what a garden is?",
]


def pick_prompt_voice(voice_ids: Sequence[str], gold: dict, idx: int) -> str:
    """A corpus voice id not used in this conversation (round-robin by scenario index)."""
    used = set((gold.get("metadata", {}).get("voices") or {}).values())
    free = [v for v in sorted(voice_ids) if v not in used]
    if not free:
        free = sorted(voice_ids)
    return free[idx % len(free)]


def peak_norm(x: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale ``x`` so its absolute peak is ``dbfs`` (silence is returned unchanged)."""
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p < 1e-9:
        return x
    return (x * (10 ** (dbfs / 20.0) / p)).astype(np.float32)


def mix(chans: Dict[str, np.ndarray], t0: float, t1: float) -> np.ndarray:
    """Sum of all channels over ``[t0, t1)`` seconds (zero-padded outside the recording)."""
    a, b = int(round(t0 * SR)), int(round(t1 * SR))
    out = np.zeros(max(0, b - a), dtype=np.float32)
    for x in chans.values():
        lo, hi = max(a, 0), min(b, len(x))
        if hi > lo:
            out[lo - a: hi - a] += x[lo:hi]
    return out


def cut(chan: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Samples of one channel over ``[t0, t1)`` seconds, clipped to the recording."""
    a, b = int(round(t0 * SR)), int(round(t1 * SR))
    a, b = max(0, a), min(len(chan), b)
    return chan[a:b].astype(np.float32) if b > a else np.zeros(0, dtype=np.float32)


def silence(sec: float) -> np.ndarray:
    """``sec`` seconds of digital silence."""
    return np.zeros(max(0, int(round(sec * SR))), dtype=np.float32)


def build_start_pack(sc: Dict[str, Any], chans: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Stream (before peak normalisation) and metadata of a START scenario."""
    ctx0 = float(sc["context_start_s"])
    ev = float(sc["event_time"])
    head = mix(chans, ctx0, ev)
    if sc["event_type"] == "true_eot":
        stream = np.concatenate([head, silence(float(sc["open_floor_s"]))])
        neg_span = None
    else:
        resume = float(sc["resume_time"])
        stream = np.concatenate([head, mix(chans, ev, resume + HOLD_TAIL_S)])
        neg_span = [0.0, resume - ev]   # relative to the event
    return stream, {"event_offset_s": round(ev - ctx0, 4),
                    "negative_span_rel": neg_span,
                    "stream_duration_s": round(len(stream) / SR, 4)}


def build_stop_pack(sc: Dict[str, Any], chans: Dict[str, np.ndarray], tts, voice: str, idx: int
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Stream, inject clip (both before final normalisation) and metadata of a STOP scenario."""
    ev = float(sc["event_time"])
    ctx1 = max(0.0, ev - 0.05)
    ctx0 = max(0.0, ctx1 - STOP_CONTEXT_S)
    context = mix(chans, ctx0, ctx1)

    prompt_text = PROMPTS[idx % len(PROMPTS)]
    pa = tts.synth(prompt_text, voice)
    if getattr(pa, "sample_rate", SR) != SR:
        raise ValueError(f"TTS sample rate {pa.sample_rate} != pack sample rate {SR}")
    pw = peak_norm(np.asarray(pa.audio, dtype=np.float32), PROMPT_PEAK_DBFS)

    stream = np.concatenate([context, silence(PRE_PROMPT_GAP_S), pw, silence(POST_PROMPT_TAIL_S)])
    clip = sc["injected_clip"]
    inject = cut(chans[sc["target_speaker"]], float(clip["start_s"]), float(clip["end_s"]))
    return stream, inject, {
        "prompt_text": prompt_text,
        "prompt_voice": voice,
        "prompt_onset_s": round((len(context) + len(silence(PRE_PROMPT_GAP_S))) / SR, 4),
        "stream_duration_s": round(len(stream) / SR, 4),
        "inject_duration_s": round(len(inject) / SR, 4),
        "inject_source": {"conversation_id": sc["conversation_id"], "speaker": sc["target_speaker"],
                          "clip_id": clip["id"], "start_s": clip["start_s"], "end_s": clip["end_s"]},
    }


def build_pack_audio(sc: Dict[str, Any], idx: int, ds: Dataset, tts=None
                     ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Final (peak-normalised) ``stream``, ``inject`` (None for START) and ``pack.json`` content.

    ``idx`` is the scenario's 0-based position in the full scenario list; it selects the STOP prompt
    and prompt voice. ``tts`` (a :class:`mptb.generation.tts.KokoroTTS`) is required for STOP scenarios.
    """
    gold = ds.gold(sc["conversation_id"])
    chans, sr = ds.audio(sc["conversation_id"], AUDIO_CONDITION)
    if sr != SR:
        raise RuntimeError(f"{sc['conversation_id']}: sample rate {sr} != {SR}")
    if sc["protocol"] == "START":
        stream, meta = build_start_pack(sc, chans)
        inject = None
    else:
        voice = pick_prompt_voice(list(tts.binding), gold, idx)
        stream, inject, meta = build_stop_pack(sc, chans, tts, voice, idx)
        inject = peak_norm(inject, INJECT_PEAK_DBFS)
    stream = peak_norm(stream, STREAM_PEAK_DBFS)
    pack = {"scenario_id": sc["scenario_id"], "protocol": sc["protocol"], "event_type": sc["event_type"],
            "audio_condition": AUDIO_CONDITION, "sample_rate": SR, "stream": "stream.flac",
            "inject": "inject.flac" if inject is not None else None, **meta}
    return stream, inject, pack


def write_packs(scenarios: List[Dict[str, Any]], ds: Dataset, out: Path,
                only: Optional[Sequence[str]] = None, device: str = "cuda") -> List[Dict[str, Any]]:
    """Write every pack (or only the scenario ids in ``only``) under ``out``; returns the pack list."""
    import soundfile as sf

    todo = [(i, sc) for i, sc in enumerate(scenarios) if not only or sc["scenario_id"] in set(only)]
    tts = None
    if any(sc["protocol"] == "STOP" for _, sc in todo):
        from ..generation.tts import KokoroTTS
        tts = KokoroTTS(device=device)
    packs = []
    for n, (i, sc) in enumerate(todo, 1):
        stream, inject, pack = build_pack_audio(sc, i, ds, tts)
        d = out / sc["scenario_id"]
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / "stream.flac"), stream, SR, subtype="PCM_16")
        if inject is not None:
            sf.write(str(d / "inject.flac"), inject, SR, subtype="PCM_16")
        (d / "pack.json").write_text(json.dumps(pack, indent=1))
        packs.append(pack)
        print(f"  [{n}/{len(todo)}] {sc['scenario_id']} {sc['event_type']:<11} "
              f"stream={pack['stream_duration_s']:.1f}s")
    (out / "packs.json").write_text(json.dumps(packs, indent=1))
    return packs


def load_packs(audio_root: Path) -> Dict[str, Dict[str, Any]]:
    """``{scenario_id: pack}`` from ``<audio_root>/packs.json`` (or the per-scenario ``pack.json`` files)."""
    pj = audio_root / "packs.json"
    if pj.exists():
        return {p["scenario_id"]: p for p in json.loads(pj.read_text())}
    packs = {}
    for d in sorted(audio_root.iterdir()):
        if (d / "pack.json").exists():
            p = json.loads((d / "pack.json").read_text())
            packs[p["scenario_id"]] = p
    return packs
