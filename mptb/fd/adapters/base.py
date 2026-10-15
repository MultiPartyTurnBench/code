"""Common full-duplex adapter interface and the listener-side speech measurement (Appendix F, "Agent configuration").

Every adapter implements four coroutines::

    start_session(config)      open a session with the session instruction
    send_audio(pcm, t)         push one 20 ms frame of float32 mono audio
    receive_audio()            drain pending provider messages (non-blocking)
    close_session()

and records, on one experiment clock, every input frame, every output audio chunk and every provider
event (including cancel/interrupt events where the provider exposes them).

When is the agent speaking?
    Providers return output audio faster than real time, so the arrival of the last chunk is not when
    the agent stops being audible. :class:`VirtualPlayout` models the agent's loudspeaker: chunks are
    queued and played at wall-clock rate; an utterance starts when audio arrives on an empty buffer and
    ends when the buffer drains (gaps shorter than :attr:`VirtualPlayout.MIN_GAP_S` are jitter) or when
    the provider cancels the response. Speech intervals are scored only from this playout, never from
    provider control events or transcripts.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def api_key(env_var: str) -> str:
    """API key from the environment variable ``env_var`` (raises if unset)."""
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"set the {env_var} environment variable")
    return key


class ExperimentClock:
    """Monotonic wall clock; t = 0 is the instant the scenario audio starts streaming."""

    def __init__(self) -> None:
        self._t0: Optional[float] = None

    def start(self) -> float:
        """Set t = 0 to now."""
        self._t0 = time.monotonic()
        return self._t0

    @property
    def started(self) -> bool:
        """True once :meth:`start` was called."""
        return self._t0 is not None

    def now(self) -> float:
        """Seconds since :meth:`start` (0 before it)."""
        if self._t0 is None:
            return 0.0
        return time.monotonic() - self._t0

    def stamp(self) -> float:
        """Alias of :meth:`now` used for event timestamps."""
        return self.now()


@dataclass
class AudioChunk:
    """One received output-audio chunk."""
    arrival_s: float
    duration_s: float
    n_samples: int
    sample_rate: int
    response_id: Optional[str] = None


@dataclass
class ProviderEvent:
    """One logged provider (or experiment) event."""
    t: float
    type: str
    detail: Dict[str, Any] = field(default_factory=dict)


class VirtualPlayout:
    """Simulated loudspeaker of the agent, from which audible speech intervals are derived."""

    # A buffer underrun shorter than this is jitter inside one utterance, not a stop.
    MIN_GAP_S = 0.35

    def __init__(self) -> None:
        self.intervals: List[List[float]] = []
        self._play_until: float = -1.0
        self._open: bool = False
        self._open_start: float = 0.0
        self._pending_stop: Optional[float] = None

    def add(self, t_arrival: float, duration_s: float) -> None:
        """Queue ``duration_s`` of audio that arrived at ``t_arrival``."""
        self._resolve_pending(t_arrival)
        if not self._open:
            self._open = True
            self._open_start = t_arrival
            self._play_until = t_arrival + duration_s
        else:
            self._play_until = max(self._play_until, t_arrival) + duration_s
        self._pending_stop = None

    def _resolve_pending(self, now: float) -> None:
        if self._open and self._play_until >= 0 and now - self._play_until >= self.MIN_GAP_S:
            self.intervals.append([self._open_start, self._play_until])
            self._open = False
            self._play_until = -1.0

    def settle(self, now: float) -> None:
        """Close the open utterance if the buffer has been empty for at least ``MIN_GAP_S``."""
        self._resolve_pending(now)

    def cut(self, t: float) -> Optional[float]:
        """Provider cancel / barge-in at ``t``: the client silences the speaker; returns the stop time."""
        if self._open:
            stop = min(self._play_until, t) if self._play_until >= 0 else t
            stop = max(stop, self._open_start)
            self.intervals.append([self._open_start, stop])
            self._open = False
            self._play_until = -1.0
            return stop
        return None

    def finalize(self, now: float) -> List[List[float]]:
        """Close any open utterance at the end of its queued audio and return all intervals."""
        if self._open:
            stop = self._play_until if self._play_until >= 0 else now
            self.intervals.append([self._open_start, min(stop, max(now, self._open_start))])
            self._open = False
        return self.intervals

    @property
    def speaking(self) -> bool:
        """True while an utterance is open."""
        return self._open


@dataclass
class SessionConfig:
    """Everything an adapter needs that is not scenario audio (never the event type or label)."""
    system_prompt: str
    sample_rate_in: int = 24000
    voice: Optional[str] = None
    protocol: str = "START"          # START | STOP | DETECT
    scenario_id: str = ""
    run: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class FDAdapter(ABC):
    """Base class of all agent adapters."""

    name: str = "base"
    native_sample_rate: int = 24000
    supports_cancel_events: bool = False
    # True: live provider session paced in wall-clock time. False: local model driven in stream time.
    realtime: bool = True

    def __init__(self, **kwargs: Any) -> None:
        self.clock = ExperimentClock()
        self.playout = VirtualPlayout()
        self.events: List[ProviderEvent] = []
        self.chunks: List[AudioChunk] = []
        self.input_log: List[Tuple[float, int]] = []   # (send time, n samples)
        self.sr_in = 24000
        self.kwargs = kwargs

    @abstractmethod
    async def start_session(self, config: SessionConfig) -> None:
        """Open the provider session / prepare the local model."""

    @abstractmethod
    async def send_audio(self, pcm: np.ndarray, timestamp: float) -> None:
        """Send one frame of float32 mono audio at ``config.sample_rate_in``, stamped on the experiment clock."""

    @abstractmethod
    async def receive_audio(self) -> None:
        """Drain pending provider messages and record chunks and events (non-blocking)."""

    @abstractmethod
    async def close_session(self) -> None:
        """Close the session (idempotent)."""

    def log_event(self, etype: str, detail: Optional[Dict[str, Any]] = None) -> None:
        """Record a provider/experiment event at the current clock time."""
        self.events.append(ProviderEvent(self.clock.stamp(), etype, detail or {}))

    def on_output_audio(self, n_samples: int, sample_rate: int, response_id: Optional[str] = None) -> None:
        """Record an audible output chunk and queue it in the virtual playout."""
        t = self.clock.stamp()
        dur = n_samples / float(sample_rate)
        self.chunks.append(AudioChunk(t, dur, n_samples, sample_rate, response_id))
        self.playout.add(t, dur)

    def on_cancel(self, reason: str = "provider_cancel", detail: Optional[Dict[str, Any]] = None) -> None:
        """Record a provider cancel/interrupt event and cut the playout."""
        t = self.clock.stamp()
        stop = self.playout.cut(t)
        d = dict(detail or {})
        d["playout_stop_s"] = stop
        self.events.append(ProviderEvent(t, reason, d))

    def tick(self) -> None:
        """Advance the playout model (called from the streaming loop)."""
        self.playout.settle(self.clock.stamp())

    def get_model_speech_intervals(self) -> List[List[float]]:
        """Audible ``[start, stop]`` speech intervals of the agent on the experiment clock."""
        return [list(x) for x in self.playout.finalize(self.clock.stamp())]

    def trace(self) -> Dict[str, Any]:
        """Session record: input frames, output chunks, provider events and speech intervals."""
        return {
            "adapter": self.name,
            "input_frames": len(self.input_log),
            "input_first_s": self.input_log[0][0] if self.input_log else None,
            "input_last_s": self.input_log[-1][0] if self.input_log else None,
            "output_chunks": [
                {"t": round(c.arrival_s, 4), "dur": round(c.duration_s, 4),
                 "sr": c.sample_rate, "resp": c.response_id} for c in self.chunks],
            "provider_events": [
                {"t": round(e.t, 4), "type": e.type, "detail": e.detail} for e in self.events],
            "speech_intervals": [[round(a, 4), round(b, 4)] for a, b in self.get_model_speech_intervals()],
        }


def float_to_pcm16(x: np.ndarray) -> bytes:
    """float32 [-1, 1] -> little-endian int16 PCM bytes."""
    return np.clip(np.asarray(x, dtype=np.float32) * 32768.0, -32768, 32767).astype("<i2").tobytes()


def pcm16_to_float(b: bytes) -> np.ndarray:
    """Little-endian int16 PCM bytes -> float32."""
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampling (all rate changes here are simple 16k/24k ratios of speech)."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    n_out = int(round(len(x) * sr_out / float(sr_in)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    t_in = np.arange(len(x), dtype=np.float64)
    t_out = np.linspace(0, len(x) - 1, n_out, dtype=np.float64)
    return np.interp(t_out, t_in, np.asarray(x, dtype=np.float64)).astype(np.float32)



async def ws_connect(url: str, headers: Dict[str, str]):
    """Open a WebSocket with HTTP/1.1 ALPN and auth headers (works across ``websockets`` versions)."""
    import inspect
    import ssl

    import websockets
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.set_alpn_protocols(["http/1.1"])
    kw: Dict[str, Any] = {"max_size": None, "open_timeout": 45, "ssl": ssl_ctx, "ping_interval": 20}
    sig = inspect.signature(websockets.connect)
    kw["additional_headers" if "additional_headers" in sig.parameters else "extra_headers"] = headers
    return await websockets.connect(url, **kw)
