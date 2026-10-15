"""Step-Audio3-Realtime adapter (``step-audio-3-realtime``; Table 4, Appendix F).

StepFun's ``stepaudio-3-realtime-preview`` speaks the OpenAI Realtime WebSocket protocol (beta event
shape). Provider turn detection is left on as a normally deployed agent would use it: ``server_vad``
with 500 ms silence, ``create_response`` and ``interrupt_response`` enabled.

Logged provider signals:

* ``input_audio_buffer.speech_started`` / ``.speech_stopped`` (provider VAD), ``response.created`` / ``.done``.
* ``response.cancelled``, ``output_audio_buffer.cleared``, ``response.output_audio.cleared`` and
  ``response.done`` with status ``cancelled``: barge-in; cut the virtual playout and count as provider
  cancel events.
* ``*transcript*``: audit trail only, never a control signal or scored.

Credentials: ``STEPFUN_API_KEY``.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Dict, Optional

import numpy as np

from .base import FDAdapter, SessionConfig, api_key, float_to_pcm16, pcm16_to_float, resample, ws_connect

AUDIO_DELTA_TYPES = {"response.audio.delta", "response.output_audio.delta"}
CANCEL_TYPES = {"response.cancelled", "output_audio_buffer.cleared", "response.output_audio.cleared"}


class StepAudio3RealtimeAdapter(FDAdapter):
    """Step-Audio3-Realtime over StepFun's OpenAI-compatible realtime WebSocket."""

    name = "step-audio-3-realtime"
    model_id = "stepaudio-3-realtime-preview"
    native_sample_rate = 24000
    out_sr = 24000
    supports_cancel_events = True
    turn_detection: Dict[str, Any] = {"type": "server_vad", "silence_duration_ms": 500}

    def __init__(self, model_id: Optional[str] = None, voice: Optional[str] = None, **kw: Any):
        super().__init__(**kw)
        if model_id:
            self.model_id = model_id
        self.voice = voice
        self.ws = None

    def ws_url(self) -> str:
        """Realtime endpoint URL for :attr:`model_id`."""
        return f"wss://api.stepfun.ai/v1/realtime?model={self.model_id}"

    async def start_session(self, config: SessionConfig) -> None:
        """Connect, wait for ``session.created`` and send ``session.update``."""
        self.sr_in = int(config.sample_rate_in)
        self.ws = await ws_connect(self.ws_url(), {"Authorization": f"Bearer {api_key('STEPFUN_API_KEY')}"})
        raw = await asyncio.wait_for(self.ws.recv(), 45)
        self.log_event("session.created", {"resp": str(raw)[:200]})

        td = dict(self.turn_detection)
        td.setdefault("create_response", True)
        td.setdefault("interrupt_response", True)
        session: Dict[str, Any] = {
            "modalities": ["audio", "text"],
            "instructions": config.system_prompt,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": td,
        }
        if self.voice:
            session["voice"] = self.voice
        await self.ws.send(json.dumps({"type": "session.update", "session": session}))
        self.log_event("session.update.sent", {"turn_detection": td})

    async def send_audio(self, pcm: np.ndarray, timestamp: float) -> None:
        """Send one frame as ``input_audio_buffer.append`` (24 kHz PCM16)."""
        if self.ws is None:
            return
        x = resample(pcm, self.sr_in, self.native_sample_rate)
        b = base64.b64encode(float_to_pcm16(x)).decode()
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b}))
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

    def _handle(self, msg) -> None:
        try:
            d = json.loads(msg if isinstance(msg, str) else msg.decode("utf8", "replace"))
        except Exception:
            return
        t = d.get("type", "")

        if t in AUDIO_DELTA_TYPES:
            data = d.get("delta") or d.get("audio") or ""
            if data:
                n = len(pcm16_to_float(base64.b64decode(data)))
                self.on_output_audio(n, self.out_sr, response_id=d.get("response_id"))
            return

        if t in CANCEL_TYPES:
            self.on_cancel("provider.cancelled", {"event": t})
            return

        if t == "response.done":
            status = ((d.get("response") or {}).get("status")) or ""
            if status == "cancelled":
                self.on_cancel("provider.cancelled", {"event": t, "status": status})
            else:
                self.log_event(t, {"status": status})
            return

        if t in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
                 "response.created", "response.audio.done", "response.output_audio.done",
                 "input_audio_buffer.committed"):
            self.log_event(t, {k: d[k] for k in ("audio_start_ms", "audio_end_ms", "item_id", "response_id")
                               if k in d})
            return

        if t == "error":
            self.log_event("error", {"detail": str(d.get("error"))[:300]})
            return

        if "transcript" in t:
            self.log_event("transcript", {"type": t, "text": str(d.get("transcript", ""))[:200]})

    async def close_session(self) -> None:
        """Close the WebSocket."""
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
