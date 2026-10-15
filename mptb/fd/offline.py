"""The agents as offline detectors on the same 120 events (Appendix G, "Offline detector pass"; Table "Agents as offline detectors").

Each agent is scored the way MPTB scores a turn-taking detector, so its offline scores can be set
against its behaviour as a participant:

* EOT track (``true_eot`` positives, ``hold_pause`` negatives): the agent hears the *isolated channel*
  of the speaker whose turn ends, with :data:`EOT_PROMPT`.
* INT track (``true_int`` positives, ``backchannel`` negatives): the agent hears the *agent-free mix*
  of all speakers, with :data:`INT_PROMPT`.

The agent's response onsets are read out as commit times and scored with the MPTB window: a positive
is detected if a commit lands in ``[t - 0.25, t + 3.0]``; a negative is correct if no commit lands in
``[span_start - 0.25, span_end]``. ``E_eot`` / ``I_int`` are balanced accuracies of the two tracks.
Where the provider exposes server VAD (``speech_started`` / ``speech_stopped``), ``vad_*`` scores are
computed from those events as a cross-check (not used in the paper).

Raw detector sessions are cached one JSON line per scenario in ``<raw>/<model>.jsonl``; re-running with
the same cache only rescores.
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..data import Dataset
from ..eval.scorer import TAU_MAX, TAU_PRE
from .adapters import get_adapter
from .adapters.base import SessionConfig
from .protocols import Streamer

SR = 24000
TAIL_S = 4.0                 # audio after the event (true_eot: silence; INT: real audio)
HOLD_TAIL_S = 3.0            # audio kept after a hold resumes
INT_PEAK_DBFS = -3.0
DRAIN_S = 1.5

EOT_PROMPT = (
    "You are listening to one person speaking. Say a short acknowledgement the "
    "moment you believe they have finished what they are saying and are handing "
    "over. Stay completely silent while they are still speaking."
)
INT_PROMPT = (
    "You are listening to a conversation. Say a short acknowledgement the moment "
    "you believe someone has cut in to take the floor from the person who was "
    "speaking. Stay completely silent for brief listening noises such as "
    "'mm-hm' or 'yeah' that do not take the floor."
)


def cut_padded(x: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Samples over ``[t0, t1)`` seconds, zero-padded past the end of the recording."""
    a, b = max(0, int(round(t0 * SR))), max(0, int(round(t1 * SR)))
    seg = x[a:b]
    if len(seg) < b - a:
        seg = np.pad(seg, (0, b - a - len(seg)))
    return seg


def negative_span_rel(sc: Dict[str, Any]) -> Optional[List[float]]:
    """Negative span relative to ``context_start_s`` for ``hold_pause`` / ``backchannel``, else None."""
    if sc["event_type"] not in ("hold_pause", "backchannel"):
        return None
    span = sc.get("negative_span")
    if not span:
        return None
    t0 = float(sc["context_start_s"])
    return [float(sc["event_time"]) - t0, float(span[1]) - t0]


def build_stimulus(sc: Dict[str, Any], ds: Dataset):
    """``(audio, event_time_rel, negative_span_rel)`` of the detector stimulus for one scenario."""
    chans, sr = ds.audio(sc["conversation_id"], "noisy")
    if sr != SR:
        raise RuntimeError(f"unexpected sample rate {sr}")
    t0 = float(sc["context_start_s"])
    ev = float(sc["event_time"])
    if sc["protocol"] == "START":
        chan = chans[sc["target_speaker"]]
        if sc["event_type"] == "true_eot":
            audio = np.concatenate([cut_padded(chan, t0, ev), np.zeros(int(TAIL_S * SR), dtype=np.float32)])
            return audio, ev - t0, None
        end = float(sc["negative_span"][1]) + HOLD_TAIL_S
        return cut_padded(chan, t0, end), ev - t0, negative_span_rel(sc)

    end = ev + TAIL_S
    mixed = None
    for s in ds.gold(sc["conversation_id"])["speakers"]:
        seg = cut_padded(chans[s], t0, end)
        mixed = seg if mixed is None else mixed + seg
    peak = float(np.max(np.abs(mixed))) if mixed is not None else 0.0
    if peak > 0:
        mixed = mixed * (10 ** (INT_PEAK_DBFS / 20.0) / peak)
    return mixed.astype(np.float32), ev - t0, negative_span_rel(sc)


async def detect_once(cls, prompt: str, audio: np.ndarray, sc_id: str,
                      kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """One detector session: response onsets (``commits``) and provider VAD start/stop times."""
    adapter = cls(**kwargs)
    cfg = SessionConfig(system_prompt=prompt, sample_rate_in=SR, protocol="DETECT", scenario_id=sc_id, run=0)
    try:
        await adapter.start_session(cfg)
        adapter.clock.start()
        st = Streamer(adapter, SR)
        await st.send(audio)
        await st.drain(DRAIN_S)
    finally:
        await adapter.close_session()
    tr = adapter.trace()
    return {"commits": [a for a, _ in adapter.get_model_speech_intervals()],
            "vad_stops": [e["detail"].get("audio_end_ms", 0) / 1000.0 for e in tr["provider_events"]
                          if e["type"].endswith("speech_stopped") and e["detail"].get("audio_end_ms") is not None],
            "vad_starts": [e["detail"].get("audio_start_ms", 0) / 1000.0 for e in tr["provider_events"]
                           if e["type"].endswith("speech_started")
                           and e["detail"].get("audio_start_ms") is not None]}


def hit(commits: Sequence[float], ev: float, neg: Optional[Sequence[float]]) -> bool:
    """MPTB per-event outcome: positive detected in the window / negative span left clean."""
    if neg is None:
        return any(ev - TAU_PRE <= c <= ev + TAU_MAX for c in commits)
    lo, hi = neg[0] - TAU_PRE, neg[1]
    return not any(lo <= c <= hi for c in commits)


def load_raw(path: Path) -> Dict[str, Dict[str, Any]]:
    """``{scenario_id: raw detector record}`` from a raw JSONL cache (later lines win)."""
    done: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        for line in path.open():
            d = json.loads(line)
            done[d["scenario_id"]] = d
    return done


async def collect(model: str, scenarios: List[Dict[str, Any]], ds: Dataset, raw_path: Path,
                  timeout: float = 900.0, adapter_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Run every detector session missing from ``raw_path`` and append it; returns all raw records."""
    cls = get_adapter(model)
    done = load_raw(raw_path)
    print(f"\n=== {model}: {len(scenarios) - len(done)} detector sessions to run")
    t0 = time.time()
    with raw_path.open("a") as fh:
        for i, sc in enumerate(scenarios, 1):
            sid = sc["scenario_id"]
            if sid in done:
                continue
            audio, ev, neg = build_stimulus(sc, ds)
            prompt = EOT_PROMPT if sc["protocol"] == "START" else INT_PROMPT
            base = {"scenario_id": sid, "event_type": sc["event_type"], "protocol": sc["protocol"]}
            try:
                r = await asyncio.wait_for(detect_once(cls, prompt, audio, sid, adapter_kwargs or {}), timeout=timeout)
                rec = {**base, "ok": True, "event_time_rel": round(ev, 4), "negative_span_rel": neg, **r}
            except Exception as e:
                rec = {**base, "ok": False, "error": repr(e)[:300], "traceback": traceback.format_exc()[-800:],
                       "event_time_rel": round(ev, 4), "negative_span_rel": neg,
                       "commits": [], "vad_stops": [], "vad_starts": []}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[sid] = rec
            if i % 10 == 0 or i == len(scenarios):
                print(f"  [{i}/{len(scenarios)}] {(time.time() - t0) / max(i, 1):.1f}s/session", flush=True)
    return done


def _summarise(t: Dict[str, int]) -> Dict[str, Any]:
    pos, negc = t["tp"] + t["fn"], t["fp"] + t["tn"]
    return {"recall": t["tp"] / pos if pos else None, "fpr": t["fp"] / negc if negc else None,
            "n_positives": pos, "n_negative_spans": negc, **t}


def _balanced(s: Dict[str, Any]) -> Optional[float]:
    if s["recall"] is None or s["fpr"] is None:
        return None
    return 0.5 * (s["recall"] + (1 - s["fpr"]))


def score_detector(scenarios: List[Dict[str, Any]], done: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-agent offline scores from raw detector records.

    Returns ``{E_eot, I_int, eot, int, vad_available, vad_E_eot, vad_I_int, vad_eot, vad_int,
    n_sessions, per_event}``; ``per_event[scenario_id]`` is the detector's correctness on that event
    (used by the event-level contingency). Negative spans are recomputed from the scenario.
    """
    per_event: Dict[str, bool] = {}
    tracks = {k: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for k in ("eot", "int")}
    vad_tracks = {k: {"tp": 0, "fn": 0, "fp": 0, "tn": 0} for k in ("eot", "int")}
    n_vad = 0
    for sc in scenarios:
        d = done.get(sc["scenario_id"])
        if not d or not d.get("ok"):
            continue
        track = "eot" if sc["protocol"] == "START" else "int"
        ev, neg = d["event_time_rel"], negative_span_rel(sc)
        ok = hit(d["commits"], ev, neg)
        per_event[sc["scenario_id"]] = ok
        tracks[track][("tp" if ok else "fn") if neg is None else ("tn" if ok else "fp")] += 1
        vsig = d["vad_stops"] if track == "eot" else d["vad_starts"]
        if vsig:
            n_vad += 1
            vok = hit(vsig, ev, neg)
            vad_tracks[track][("tp" if vok else "fn") if neg is None else ("tn" if vok else "fp")] += 1

    eot, intr = _summarise(tracks["eot"]), _summarise(tracks["int"])
    vadeot, vadint = _summarise(vad_tracks["eot"]), _summarise(vad_tracks["int"])
    return {
        "E_eot": _balanced(eot), "I_int": _balanced(intr),
        "eot": eot, "int": intr,
        "vad_available": n_vad > 0,
        "vad_E_eot": _balanced(vadeot) if n_vad else None,
        "vad_I_int": _balanced(vadint) if n_vad else None,
        "vad_eot": vadeot if n_vad else None,
        "vad_int": vadint if n_vad else None,
        "n_sessions": len([1 for s in scenarios if done.get(s["scenario_id"], {}).get("ok")]),
        "per_event": per_event,
    }
