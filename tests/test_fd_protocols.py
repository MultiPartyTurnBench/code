"""Dry-run tests of the START / STOP probes with a scripted stream-time agent (no model, no network)."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from mptb.fd.adapters.base import FDAdapter, VirtualPlayout  # noqa: E402
from mptb.fd.adapters.local_base import StreamClock  # noqa: E402
from mptb.fd.protocols import TrialResult, run_trial, score_start, score_stop  # noqa: E402

SR = 24000


class ScriptedAgent(FDAdapter):
    """Speaks from ``speak_at`` for ``utter_s`` seconds; optionally stops ``yield_delay`` s after it hears
    loud input while speaking. Driven in stream time like the local adapters."""

    name = "scripted"
    realtime = False

    def __init__(self, speak_at=None, utter_s=6.0, yield_delay=None, **kw):
        super().__init__(**kw)
        self.clock = StreamClock()
        self.speak_at, self.utter_s, self.yield_delay = speak_at, utter_s, yield_delay
        self.stop_at = None

    async def start_session(self, config):
        self.sr_in = config.sample_rate_in
        self.clock.start()

    async def send_audio(self, pcm, timestamp):
        self.input_log.append((timestamp, len(pcm)))
        now, dur = self.clock.now(), len(pcm) / self.sr_in
        if self.speak_at is not None and self.speak_at <= now < self.speak_at + self.utter_s:
            if self.yield_delay is not None and self.stop_at is None and now > self.speak_at + 0.1 \
                    and float(np.abs(pcm).max()) > 0.05:
                self.stop_at = now + self.yield_delay
            if self.stop_at is None or now < self.stop_at:
                self.on_output_audio(len(pcm), self.sr_in)
        self.clock.advance(dur)

    async def receive_audio(self):
        self.tick()

    async def close_session(self):
        pass


def _write(path, x):
    sf.write(str(path), x.astype(np.float32), SR, subtype="PCM_16")


def _start_pack(tmp_path, event_type, neg=None):
    rng = np.random.default_rng(0)
    _write(tmp_path / "stream.flac", np.concatenate([0.3 * rng.standard_normal(3 * SR), np.zeros(4 * SR)]))
    return {"stream": "stream.flac", "event_offset_s": 3.0, "negative_span_rel": neg}


def _stop_pack(tmp_path):
    rng = np.random.default_rng(1)
    _write(tmp_path / "stream.flac", 0.3 * rng.standard_normal(2 * SR))
    _write(tmp_path / "inject.flac", 0.3 * rng.standard_normal(SR // 2))
    return {"stream": "stream.flac", "inject": "inject.flac"}


def _run(agent, event_type, protocol, pack, tmp_path):
    sc = {"scenario_id": "s1", "event_type": event_type, "protocol": protocol}
    return asyncio.run(run_trial(agent, sc, pack, tmp_path, 0))


def test_start_true_eot_correct_start(tmp_path):
    res = _run(ScriptedAgent(speak_at=4.0), "true_eot", "START", _start_pack(tmp_path, "true_eot"), tmp_path)
    assert res.ok and res.correct
    assert res.latency_ms == pytest.approx(1000.0, abs=1)


def test_start_true_eot_too_late(tmp_path):
    res = _run(ScriptedAgent(speak_at=6.5), "true_eot", "START", _start_pack(tmp_path, "true_eot"), tmp_path)
    assert res.ok and res.correct is False and res.latency_ms is None


def test_start_hold_pause_false_start(tmp_path):
    pack = _start_pack(tmp_path, "hold_pause", neg=[0.0, 1.0])
    res = _run(ScriptedAgent(speak_at=3.5), "hold_pause", "START", pack, tmp_path)
    assert res.correct is False and res.detail["false_start"]
    res = _run(ScriptedAgent(speak_at=4.5), "hold_pause", "START", pack, tmp_path)
    assert res.correct is True and res.detail["false_start_ext_window"]


def test_stop_yield_and_false_yield(tmp_path):
    pack = _stop_pack(tmp_path)
    res = _run(ScriptedAgent(speak_at=2.2, yield_delay=0.3), "true_int", "STOP", pack, tmp_path)
    assert res.ok and res.correct and res.detail["stopped_in_window"]
    assert 0 <= res.latency_ms <= 1500
    res = _run(ScriptedAgent(speak_at=2.2, yield_delay=0.3), "backchannel", "STOP", pack, tmp_path)
    assert res.ok and res.correct is False


def test_stop_keeps_talking(tmp_path):
    res = _run(ScriptedAgent(speak_at=2.2, utter_s=10.0), "true_int", "STOP", _stop_pack(tmp_path), tmp_path)
    assert res.ok and res.correct is False and res.latency_ms > 1500


def test_stop_invalid_when_agent_never_speaks(tmp_path):
    res = _run(ScriptedAgent(), "true_int", "STOP", _stop_pack(tmp_path), tmp_path)
    assert not res.ok and res.invalid_reason == "agent_never_spoke"


def test_stop_invalid_when_silent_at_injection(tmp_path):
    res = _run(ScriptedAgent(speak_at=2.2, utter_s=1.0), "true_int", "STOP", _stop_pack(tmp_path), tmp_path)
    assert not res.ok and res.invalid_reason == "agent_not_speaking_at_injection"


def test_virtual_playout_merges_jitter_and_cuts():
    p = VirtualPlayout()
    p.add(0.0, 1.0)
    p.add(1.2, 1.0)          # 0.2 s gap < MIN_GAP_S: same utterance
    p.settle(2.6)            # 0.4 s after the buffer drained: utterance closed
    assert p.intervals == [[0.0, 2.2]] and not p.speaking
    p.add(3.0, 5.0)
    assert p.cut(4.0) == 4.0
    assert p.finalize(10.0) == [[0.0, 2.2], [3.0, 4.0]]


def test_score_functions_directly():
    r = score_start(TrialResult("s", "m", 0, "START", "true_eot"), "true_eot", 10.0, [9.0, 9.8, 12.0], None)
    assert r.correct and r.latency_ms == -200.0 and r.detail["spurious_pre_event_starts"] == 1
    events = [{"type": "provider.cancelled"}, {"type": "transcript"}]
    r = score_stop(TrialResult("s", "m", 0, "STOP", "true_int", provider_events=events), "true_int",
                   [[1.0, 5.4], [6.0, 7.0]], 5.0, True)
    assert r.correct and r.detail["resumed_within_2s"] and r.detail["provider_cancel_events"] == 1


def test_runner_writes_jsonl_and_resumes(tmp_path, monkeypatch):
    from mptb.fd import runner

    (tmp_path / "s1").mkdir()
    pack = dict(_start_pack(tmp_path / "s1", "true_eot"), scenario_id="s1", audio_condition="noisy")
    sc = {"scenario_id": "s1", "event_type": "true_eot", "protocol": "START", "N": 2,
          "conversation_type": "casual", "conversation_id": "dev_00001", "target_speaker": "speaker_1",
          "source_event_id": "e1", "event_time": 10.0, "gold_agent_behavior": "start_responding",
          "polarity": "positive"}

    class Agent(ScriptedAgent):
        def __init__(self, **kw):
            super().__init__(speak_at=4.0, **kw)

    monkeypatch.setattr(runner, "get_adapter", lambda name: Agent)
    out = tmp_path / "runs"
    out.mkdir()
    args = ("scripted", {"s1": sc}, {"s1": pack}, tmp_path, out, ["s1"])
    path = asyncio.run(runner.run_model(*args, runs=2))
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(x)["correct"] for x in lines)
    asyncio.run(runner.run_model(*args, runs=2))
    assert len(path.read_text().splitlines()) == 2
