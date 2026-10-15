"""Gemini Live adapter (``gemini-3.8-live``; Table 4, Appendix F "Agent configuration").

Raw BidiGenerateContent WebSocket (not the SDK), so every send/receive is stamped directly on the
experiment clock. The provider's automatic activity detection is left on (we measure the deployed
floor-control behaviour) and the voice is *Puck*.

Logged provider signals:

* ``serverContent.interrupted``: the server aborted its own speech because it detected the user
  taking the floor (barge-in). It cuts the virtual playout and counts as a provider cancel event.
* ``generationComplete`` / ``turnComplete``.
* ``outputTranscription``: audit trail only, never a control signal or scored.

Credentials: ``GEMINI_API_KEY``.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Dict, Optional

import numpy as np

from .base import FDAdapter, SessionConfig, api_key, float_to_pcm16, pcm16_to_float, resample

HOST = "generativelanguage.googleapis.com"
PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
IN_SR = 16000     # Live API input rate
OUT_SR = 24000    # Live API output rate


class GeminiLiveAdapter(FDAdapter):
    """Gemini Live over the BidiGenerateContent WebSocket."""

    name = "gemini-3.8-live"
    native_sample_rate = IN_SR
    supports_cancel_events = True
    model_id = "models/gemini-3.8-live"

    def __init__(self, model_id: Optional[str] = None, voice: str = "Puck", **kw: Any):
        super().__init__(**kw)
        if model_id:
            self.model_id = model_id
        self.voice = voice
        self.ws = None
        self._resid = 0

    async def start_session(self, config: SessionConfig) -> None:
        """Connect and send the setup message (audio output, system instruction, provider VAD on)."""
        import websockets
        self.sr_in = int(config.sample_rate_in)
        url = f"wss://{HOST}{PATH}?key={api_key('GEMINI_API_KEY')}"
        self.ws = await websockets.connect(url, max_size=None, open_timeout=45, ping_interval=20)
        setup: Dict[str, Any] = {"setup": {
            "model": self.model_id,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}},
            },
            "systemInstruction": {"parts": [{"text": config.system_prompt}]},
            "realtimeInputConfig": {"automaticActivityDetection": {}},
            "outputAudioTranscription": {},
        }}
        await self.ws.send(json.dumps(setup))
        raw = await asyncio.wait_for(self.ws.recv(), 45)
        self.log_event("session.setup", {"resp": _short(raw)})

    async def send_audio(self, pcm: np.ndarray, timestamp: float) -> None:
        """Resample to 16 kHz and send as a ``realtimeInput`` audio message."""
        if self.ws is None:
            return
        x = resample(pcm, self.sr_in, IN_SR)
        b = base64.b64encode(float_to_pcm16(x)).decode()
        await self.ws.send(json.dumps({"realtimeInput": {"audio": {
            "data": b, "mimeType": f"audio/pcm;rate={IN_SR}"}}}))
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
        sc = d.get("serverContent") or {}

        if sc.get("interrupted"):
            self.on_cancel("provider.interrupted", {"src": "serverContent.interrupted"})

        mt = sc.get("modelTurn") or {}
        for part in mt.get("parts", []) or []:
            inline = part.get("inlineData") or {}
            data = inline.get("data")
            if not data:
                continue
            mime = inline.get("mimeType", "") or ""
            sr = OUT_SR
            if "rate=" in mime:
                try:
                    sr = int(mime.split("rate=")[1].split(";")[0])
                except ValueError:
                    pass
            n = len(pcm16_to_float(base64.b64decode(data)))
            self.on_output_audio(n, sr, response_id=f"r{self._resid}")

        if sc.get("generationComplete"):
            self._resid += 1
            self.log_event("generationComplete", {})
        if sc.get("turnComplete"):
            self.log_event("turnComplete", {})
        if sc.get("outputTranscription"):
            self.log_event("outputTranscription",
                           {"text": (sc["outputTranscription"] or {}).get("text", "")[:200]})
        if d.get("goAway"):
            self.log_event("goAway", {"detail": _short(msg)})
        if d.get("usageMetadata"):
            self.log_event("usage", {})

    async def close_session(self) -> None:
        """Close the WebSocket."""
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


def _short(x, n: int = 240) -> str:
    s = x if isinstance(x, str) else x.decode("utf8", "replace")
    return s[:n]
