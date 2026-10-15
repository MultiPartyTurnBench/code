"""Offline tests of the provider adapters with a fake WebSocket (no network, no API keys used)."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import types

import numpy as np
import pytest

from mptb.fd.adapters import ADAPTERS, available, get_adapter
from mptb.fd.adapters.base import SessionConfig, api_key, float_to_pcm16, pcm16_to_float, resample


class FakeWS:
    """Minimal async WebSocket: replays queued server messages, records sent ones."""

    def __init__(self, messages=()):
        self.inbox = list(messages)
        self.sent = []

    async def recv(self):
        if self.inbox:
            return self.inbox.pop(0)
        await asyncio.sleep(0.05)
        raise RuntimeError("unreachable")

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def close(self):
        pass


def _fake_websockets(monkeypatch, ws):
    mod = types.ModuleType("websockets")

    async def connect(url, max_size=None, open_timeout=None, ssl=None, ping_interval=None, additional_headers=None):
        ws.url, ws.headers = url, additional_headers
        return ws

    mod.connect = connect
    monkeypatch.setitem(sys.modules, "websockets", mod)


def _pcm_b64(x):
    return base64.b64encode(float_to_pcm16(x)).decode()


CFG = SessionConfig(system_prompt="hello", sample_rate_in=24000, scenario_id="s1", run=0)


def test_registry_lists_the_five_paper_agents():
    assert available() == sorted(["gpt-live-1", "gemini-3.8-live", "step-audio-3-realtime",
                                  "minicpm-o-4.5", "realtime-venus"])
    for name in ("gpt-live-1", "gemini-3.8-live", "step-audio-3-realtime"):
        assert get_adapter(name).name == name
    with pytest.raises(KeyError):
        get_adapter("nope")
    assert set(ADAPTERS) == set(available())


def test_api_key_from_environment(monkeypatch):
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        api_key("STEPFUN_API_KEY")
    monkeypatch.setenv("STEPFUN_API_KEY", "k")
    assert api_key("STEPFUN_API_KEY") == "k"


def test_pcm_and_resample_roundtrip():
    x = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    assert np.allclose(pcm16_to_float(float_to_pcm16(x)), x, atol=1e-4)
    assert len(resample(x, 24000, 16000)) == 320


def test_gemini_session_audio_and_barge_in(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    ws = FakeWS([json.dumps({"setupComplete": {}})])
    _fake_websockets(monkeypatch, ws)
    a = get_adapter("gemini-3.8-live")()

    async def go():
        await a.start_session(CFG)
        a.clock.start()
        await a.send_audio(np.zeros(480, dtype=np.float32), 0.0)
        ws.inbox += [json.dumps({"serverContent": {"modelTurn": {"parts": [{"inlineData": {
                         "mimeType": "audio/pcm;rate=24000", "data": _pcm_b64(0.1 * np.ones(2400))}}]}}}),
                     json.dumps({"serverContent": {"interrupted": True}})]
        await a.receive_audio()
        await a.close_session()

    asyncio.run(go())
    setup = ws.sent[0]["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"
    assert setup["realtimeInputConfig"] == {"automaticActivityDetection": {}}
    assert ws.sent[1]["realtimeInput"]["audio"]["mimeType"] == "audio/pcm;rate=16000"
    assert len(a.chunks) == 1 and a.chunks[0].duration_s == pytest.approx(0.1)
    assert [e.type for e in a.events][-1] == "provider.interrupted"
    assert len(a.get_model_speech_intervals()) == 1


def test_step_audio_session_update_and_cancel(monkeypatch):
    monkeypatch.setenv("STEPFUN_API_KEY", "test")
    ws = FakeWS([json.dumps({"type": "session.created"})])
    _fake_websockets(monkeypatch, ws)
    a = get_adapter("step-audio-3-realtime")()

    async def go():
        await a.start_session(CFG)
        a.clock.start()
        ws.inbox += [json.dumps({"type": "response.audio.delta", "delta": _pcm_b64(0.1 * np.ones(4800)),
                                 "response_id": "r1"}),
                     json.dumps({"type": "input_audio_buffer.speech_started", "audio_start_ms": 100}),
                     json.dumps({"type": "response.done", "response": {"status": "cancelled"}})]
        await a.receive_audio()

    asyncio.run(go())
    assert ws.url.endswith("model=stepaudio-3-realtime-preview")
    assert ws.headers == {"Authorization": "Bearer test"}
    sess = ws.sent[0]["session"]
    assert sess["turn_detection"] == {"type": "server_vad", "silence_duration_ms": 500,
                                      "create_response": True, "interrupt_response": True}
    types_ = [e.type for e in a.events]
    assert "input_audio_buffer.speech_started" in types_ and types_[-1] == "provider.cancelled"


def test_gpt_live_energy_gate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    ws = FakeWS([json.dumps({"type": "session.started", "session": {"id": "x"}})])
    _fake_websockets(monkeypatch, ws)
    a = get_adapter("gpt-live-1")()
    speech = np.concatenate([np.zeros(4800), 0.05 * np.ones(2400)]).astype(np.float32)

    async def go():
        await a.start_session(CFG)
        a.clock.start()
        ws.inbox += [json.dumps({"type": "session.output_audio.delta", "delta": _pcm_b64(np.zeros(4800))}),
                     json.dumps({"type": "session.output_audio.delta", "delta": _pcm_b64(speech)})]
        await a.receive_audio()

    asyncio.run(go())
    assert ws.sent[0]["type"] == "session.start" and ws.sent[0]["session"]["audio"]["output"]["voice"] == "marin"
    assert len(a.chunks) == 1 and a.chunks[0].n_samples == 2400
    assert not a.supports_cancel_events
