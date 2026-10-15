"""Adapter for OpenAI's GPT-Live full-duplex voice model (``gpt-live-1``; Table 4, Appendix F).

GPT-Live does not use the Realtime API protocol; it has its own session surface:

  endpoint       ``wss://api.openai.com/v1/live/sessions``
  client events  ``session.start``, ``session.input_audio.append``
  server events  ``session.started``, ``session.output_audio.delta``, ``session.*_transcript.delta``,
                 ``session.usage.updated``, ``session.closed``

Two properties matter for scoring:

1. **The output stream is continuous.** GPT-Live streams 24 kHz PCM for the whole session and sends
   digital silence when the agent is not talking, so output audio is energy-gated (frames with RMS
   above :data:`SPEECH_RMS`, about -60 dBFS; measured idle RMS is exactly 0, speech 2e-2..1.2e-1) and
   only audible samples are handed to the virtual playout.
2. **There are no barge-in / cancel events.** GPT-Live arbitrates the floor internally, so
   ``supports_cancel_events`` is False and its barge-in rate is a structural zero.

Voice: *marin*. Credentials: ``OPENAI_API_KEY``. Transcripts are logged for the audit trail only.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Dict, Optional

import numpy as np

from .base import FDAdapter, SessionConfig, api_key, float_to_pcm16, pcm16_to_float, resample, ws_connect

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
SPEECH_RMS = 1e-3
GATE_FRAME_S = 0.02


class GptLive1Adapter(FDAdapter):
    """GPT-Live-1 over the OpenAI Live session WebSocket."""

    name = "gpt-live-1"
    model_id = "gpt-live-1"
    native_sample_rate = 24000
    out_sr = 24000
    voice = "marin"
    supports_cancel_events = False
    realtime = True

    def __init__(self, model_id: Optional[str] = None, voice: Optional[str] = None, **kw: Any):
        super().__init__(**kw)
        if model_id:
            self.model_id = model_id
        if voice:
            self.voice = voice
        self.ws = None
        self._audible_s = 0.0
        self._silent_s = 0.0

    async def start_session(self, config: SessionConfig) -> None:
        """Connect, send ``session.start`` and wait for ``session.started``."""
        self.sr_in = int(config.sample_rate_in)
        self.ws = await ws_connect(LIVE_URL, {"Authorization": f"Bearer {api_key('OPENAI_API_KEY')}"})
        await self.ws.send(json.dumps({
            "type": "session.start",
            "event_id": f"{config.scenario_id or 'fdb'}-{config.run}",
            "session": {
                "model": self.model_id,
                "instructions": config.system_prompt,
                "audio": {"output": {"voice": self.voice},
                          "format": {"type": "audio/pcm", "rate": self.native_sample_rate}},
            },
        }))
        raw = await asyncio.wait_for(self.ws.recv(), 45)
        d = json.loads(raw)
        if d.get("type") != "session.started":
            raise RuntimeError(f"gpt-live session.start failed: {str(raw)[:300]}")
        self.log_event("session.started", {"session_id": (d.get("session") or {}).get("id")})

    async def send_audio(self, pcm: np.ndarray, timestamp: float) -> None:
        """Send one frame as ``session.input_audio.append`` (24 kHz PCM16)."""
        if self.ws is None:
            return
        x = resample(pcm, self.sr_in, self.native_sample_rate)
        b = base64.b64encode(float_to_pcm16(x)).decode()
        await self.ws.send(json.dumps({"type": "session.input_audio.append", "audio": b}))
        self.input_log.append((timestamp, len(x)))

    async def receive_audio(self) -> None:
        """Drain everything currently pending without blocking the stream loop."""
        if self.ws is None:
            return
        while True:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), 0.001)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                break
            except Exception as e:
                self.log_event("ws.error", {"err": repr(e)})
                self.ws = None
                break
            self._handle(msg)
        self.tick()

    def audible_samples(self, x: np.ndarray) -> int:
        """Number of samples of ``x`` in 20 ms frames whose RMS exceeds :data:`SPEECH_RMS`."""
        n = max(1, int(GATE_FRAME_S * self.out_sr))
        usable = (len(x) // n) * n
        if usable == 0:
            return int(np.sqrt(float((x ** 2).mean())) > SPEECH_RMS) * len(x)
        f = x[:usable].reshape(-1, n)
        active = int((np.sqrt((f ** 2).mean(axis=1)) > SPEECH_RMS).sum())
        return active * n

    def _handle(self, msg) -> None:
        try:
            d = json.loads(msg if isinstance(msg, str) else msg.decode("utf8", "replace"))
        except Exception:
            return
        t = d.get("type", "")

        if t == "session.output_audio.delta":
            data = d.get("delta") or d.get("audio") or ""
            if not data:
                return
            x = pcm16_to_float(base64.b64decode(data))
            n = self.audible_samples(x)
            if n:
                self._audible_s += n / float(self.out_sr)
                self.on_output_audio(n, self.out_sr)
            else:
                self._silent_s += len(x) / float(self.out_sr)
            return

        if t == "session.closed":
            self.log_event("session.closed", {"reason": d.get("reason")})
            self.ws = None
            return

        if t == "error":
            self.log_event("error", {"detail": str(d.get("error"))[:300]})
            return

        if "transcript" in t:
            self.log_event("transcript", {"type": t, "text": str(d.get("delta", ""))[:200]})
            return

        if t == "session.usage.updated":
            return

        self.log_event(t, {})

    def trace(self) -> Dict[str, Any]:
        """Base trace plus the output energy-gate statistics."""
        tr = super().trace()
        tr["gate"] = {"audible_s": round(self._audible_s, 3),
                      "gated_silence_s": round(self._silent_s, 3),
                      "speech_rms_threshold": SPEECH_RMS}
        return tr

    async def close_session(self) -> None:
        """Close the WebSocket."""
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
