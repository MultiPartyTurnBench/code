"""The START and STOP probes and their trial scoring (Section 4 "Full-Duplex Behavioral Diagnostics", Appendix H.3).

START: "should I start?"
    The agent hears real conversation ending at a floor event. ``true_eot``: the speaker yielded, so a
    correct agent starts speaking inside the TurnBench window ``[t - 0.25, t + 3.0]`` s (correct start,
    S). ``hold_pause``: the speaker only paused and resumes, so any agent speech onset inside the MPTB
    negative span ``[t - 0.25, resume]`` is a false start (FS).

STOP: "should I stop?"
    A trivial spoken question gets the agent talking; :data:`INJECT_DELAY_S` after its speech actually
    begins, a real gold event is played. ``true_int``: a genuine interruption, so a correct agent's
    ongoing utterance ends within ``[0, 1.5]`` s of the injection (yield, Y). ``backchannel``: stopping
    in that window is a false yield (FY). A STOP trial is valid only if the agent is speaking when the
    event is injected.

Only speech onset and cessation, measured from the virtual playout, are scored; response content
never is.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters.base import FDAdapter, SessionConfig

FRAME_S = 0.02

# Identical for every scenario and agent; it never mentions turn-taking, interruptions or measurement.
SYSTEM_PROMPT = (
    "You are taking part in a spoken conversation with other people. "
    "Participate naturally. Respond when another participant clearly gives you the "
    "floor. Do not interrupt speakers who are still holding the floor."
)

START_WINDOW = (-0.25, 3.0)   # TurnBench / MPTB matching window (tau_pre, tau_max)
STOP_WINDOW = (0.0, 1.5)      # a stop inside this window after injection counts as yielding

WAIT_FOR_SPEECH_S = 14.0      # how long the agent may take to start answering the question
INJECT_DELAY_S = 2.0          # the event lands this long after the agent starts speaking
POST_INJECT_S = 8.0           # observation window after the event
POST_STREAM_DRAIN_S = 1.5     # keep draining briefly so late audio is not missed
RESUME_WINDOW_S = 2.0         # "resumed": a new utterance starts within this time of the stop


@dataclass
class TrialResult:
    """One trial record (written as one JSONL line by the runner)."""
    scenario_id: str
    model: str
    run: int
    protocol: str
    event_type: str
    ok: bool = True
    invalid_reason: Optional[str] = None
    correct: Optional[bool] = None
    latency_ms: Optional[float] = None
    model_speech_start: List[float] = field(default_factory=list)
    model_speech_stop: List[float] = field(default_factory=list)
    provider_events: List[Dict[str, Any]] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)


def load_pack_audio(path) -> Tuple[np.ndarray, int]:
    """Read a pack FLAC as mono float32 and its sample rate."""
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr


class Streamer:
    """Feeds audio to an adapter in 20 ms frames and keeps its receive loop serviced.

    Live (``realtime``) adapters are paced at wall-clock rate; local adapters run in stream time.
    """

    def __init__(self, adapter: FDAdapter, sr: int):
        self.a = adapter
        self.sr = sr
        self.frame = max(1, int(round(FRAME_S * sr)))

    async def send(self, audio: np.ndarray) -> None:
        """Stream ``audio``."""
        for i in range(0, len(audio), self.frame):
            await self._one(audio[i:i + self.frame])

    async def send_silence(self, seconds: float) -> None:
        """Stream ``seconds`` of silence."""
        n = int(round(seconds / FRAME_S))
        z = np.zeros(self.frame, dtype=np.float32)
        for _ in range(n):
            await self._one(z)

    async def send_until(self, seconds: float, predicate: Callable[[], bool]) -> bool:
        """Stream up to ``seconds`` of silence, stopping early (returning True) once ``predicate()`` holds."""
        n = int(round(seconds / FRAME_S))
        z = np.zeros(self.frame, dtype=np.float32)
        for _ in range(n):
            await self._one(z)
            if predicate():
                return True
        return False

    async def _one(self, chunk: np.ndarray) -> None:
        t = self.a.clock.now()
        await self.a.send_audio(chunk, t)
        await self.a.receive_audio()
        if self.a.realtime:
            drift = (len(self.a.input_log) * FRAME_S) - self.a.clock.now()
            if drift > 0:
                await asyncio.sleep(min(drift, FRAME_S))

    async def drain(self, seconds: float) -> None:
        """Keep receiving for ``seconds`` after the stream ends (stream-time adapters: one service call)."""
        if not self.a.realtime:
            await self.a.receive_audio()
            self.a.tick()
            return
        deadline = self.a.clock.now() + seconds
        while self.a.clock.now() < deadline:
            await self.a.receive_audio()
            await asyncio.sleep(0.005)
        self.a.tick()


async def run_start_trial(adapter: FDAdapter, scenario: Dict[str, Any], pack: Dict[str, Any],
                          pack_dir: Path, run: int) -> TrialResult:
    """Stream a START pack and score the agent's speech onsets."""
    stream, sr = load_pack_audio(pack_dir / pack["stream"])
    cfg = SessionConfig(system_prompt=SYSTEM_PROMPT, sample_rate_in=sr,
                        protocol="START", scenario_id=scenario["scenario_id"], run=run)
    res = TrialResult(scenario["scenario_id"], adapter.name, run, "START", scenario["event_type"])
    try:
        await adapter.start_session(cfg)
        adapter.clock.start()
        st = Streamer(adapter, sr)
        await st.send(stream)
        await st.drain(POST_STREAM_DRAIN_S)
    finally:
        await adapter.close_session()

    intervals = adapter.get_model_speech_intervals()
    res.model_speech_start = [round(a, 4) for a, _ in intervals]
    res.model_speech_stop = [round(b, 4) for _, b in intervals]
    res.provider_events = adapter.trace()["provider_events"]
    return score_start(res, scenario["event_type"], float(pack["event_offset_s"]),
                       [a for a, _ in intervals], pack.get("negative_span_rel"))


def score_start(res: TrialResult, event_type: str, ev: float, onsets: Sequence[float],
                negative_span_rel: Optional[Sequence[float]]) -> TrialResult:
    """Fill ``correct`` / ``latency_ms`` / ``detail`` of a START trial from the agent's speech onsets.

    ``ev`` is the event time in stream seconds. For ``hold_pause`` the strict MPTB rule (a false start
    lies inside ``[ev - 0.25, ev + span_end]``) is primary; ``false_start_ext_window`` records the
    full-window variant for reference.
    """
    lo, hi = START_WINDOW
    pre = [o for o in onsets if o < ev + lo]
    if event_type == "true_eot":
        inwin = [o for o in onsets if ev + lo <= o <= ev + hi]
        res.correct = bool(inwin)
        if inwin:
            res.latency_ms = round((inwin[0] - ev) * 1000.0, 1)
        res.detail = {"delta_start_s": (round(inwin[0] - ev, 4) if inwin else None),
                      "window": [ev + lo, ev + hi],
                      "spurious_pre_event_starts": len(pre)}
    else:
        span = negative_span_rel or [0.0, 0.0]
        a, b = ev + lo, ev + float(span[1])
        fired = [o for o in onsets if a <= o <= b]
        res.correct = not fired
        if fired:
            res.latency_ms = round((fired[0] - ev) * 1000.0, 1)
        b_ext = ev + hi
        fired_ext = [o for o in onsets if a <= o <= b_ext]
        res.detail = {"false_start": bool(fired), "negative_span_abs": [a, b],
                      "negative_span_len_s": round(b - a, 4),
                      "false_start_ext_window": bool(fired_ext),
                      "ext_window_abs": [a, b_ext],
                      "spurious_pre_event_starts": len(pre)}
    res.detail["event_offset_s"] = ev
    return res


async def run_stop_trial(adapter: FDAdapter, scenario: Dict[str, Any], pack: Dict[str, Any],
                         pack_dir: Path, run: int) -> TrialResult:
    """Stream a STOP pack, inject the event once the agent is speaking, and score its cessation."""
    stream, sr = load_pack_audio(pack_dir / pack["stream"])
    inject, sr2 = load_pack_audio(pack_dir / pack["inject"])
    if sr2 != sr:
        raise RuntimeError("stream/inject sample-rate mismatch")
    cfg = SessionConfig(system_prompt=SYSTEM_PROMPT, sample_rate_in=sr,
                        protocol="STOP", scenario_id=scenario["scenario_id"], run=run)
    res = TrialResult(scenario["scenario_id"], adapter.name, run, "STOP", scenario["event_type"])

    t_inject = None
    speaking_at_inject = False
    try:
        await adapter.start_session(cfg)
        adapter.clock.start()
        st = Streamer(adapter, sr)
        await st.send(stream)
        spoke = await st.send_until(WAIT_FOR_SPEECH_S, lambda: adapter.playout.speaking)
        if not spoke:
            res.ok = False
            res.invalid_reason = "agent_never_spoke"
        else:
            await st.send_silence(INJECT_DELAY_S)
            speaking_at_inject = adapter.playout.speaking
            t_inject = adapter.clock.now()
            adapter.log_event("experiment.inject", {
                "t": round(t_inject, 4),
                "event_audio_s": round(len(inject) / sr, 4),
                "agent_speaking": speaking_at_inject})
            await st.send(inject)
            await st.send_silence(POST_INJECT_S)
            await st.drain(POST_STREAM_DRAIN_S)
    finally:
        await adapter.close_session()

    intervals = adapter.get_model_speech_intervals()
    res.model_speech_start = [round(a, 4) for a, _ in intervals]
    res.model_speech_stop = [round(b, 4) for _, b in intervals]
    res.provider_events = adapter.trace()["provider_events"]
    return score_stop(res, scenario["event_type"], intervals, t_inject, speaking_at_inject)


def score_stop(res: TrialResult, event_type: str, intervals: Sequence[Sequence[float]],
               t_inject: Optional[float], speaking_at_inject: bool) -> TrialResult:
    """Validity check and scoring of a STOP trial from the agent's speech intervals.

    The scored utterance is the one in progress at ``t_inject``; ``latency_ms`` is its end relative to
    the injection. ``provider_cancel_events`` counts logged events whose type contains "cancel" or
    "interrupt" (used for the barge-in flag of Table 19).
    """
    if not res.ok:
        return res
    if t_inject is None:
        res.ok = False
        res.invalid_reason = "no_injection"
        return res
    if not speaking_at_inject:
        res.ok = False
        res.invalid_reason = "agent_not_speaking_at_injection"
        return res

    cur = [(a, b) for a, b in intervals if a <= t_inject <= b + 1e-6]
    if not cur:
        cur = [(a, b) for a, b in intervals if a <= t_inject]
        cur = cur[-1:] if cur else []
    if not cur:
        res.ok = False
        res.invalid_reason = "no_utterance_at_injection"
        return res
    a0, b0 = cur[-1]
    delta = b0 - t_inject
    lo, hi = STOP_WINDOW
    stopped = lo <= delta <= hi
    resumed = any(a > b0 + 1e-6 and a <= b0 + RESUME_WINDOW_S for a, _ in intervals)
    cancels = [e for e in res.provider_events if "cancel" in e["type"] or "interrupt" in e["type"]]

    res.correct = bool(stopped) if event_type == "true_int" else not stopped
    res.latency_ms = round(delta * 1000.0, 1)
    res.detail = {
        "t_inject_s": round(t_inject, 4),
        "utterance": [round(a0, 4), round(b0, 4)],
        "delta_stop_s": round(delta, 4),
        "stopped_in_window": bool(stopped),
        "resumed_within_2s": bool(resumed),
        "provider_cancel_events": len(cancels),
        "window": list(STOP_WINDOW),
    }
    return res


async def run_trial(adapter: FDAdapter, scenario: Dict[str, Any], pack: Dict[str, Any],
                    pack_dir: Path, run: int) -> TrialResult:
    """Run one START or STOP trial of ``scenario`` on a fresh ``adapter``."""
    if scenario["protocol"] == "START":
        return await run_start_trial(adapter, scenario, pack, pack_dir, run)
    return await run_stop_trial(adapter, scenario, pack, pack_dir, run)
