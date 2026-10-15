"""Shared machinery for locally hosted, causal full-duplex models (MiniCPM-o-4.5, Realtime-Venus; Appendix F).

Local models are driven in *stream time*: input is resampled to 16 kHz and fed in fixed 1.0 s causal
chunks; the experiment clock (:class:`StreamClock`) advances with the audio fed, not with wall-clock
time, so the model never sees audio ahead of the current stream position and latencies are in stream
seconds. Per-chunk wall-clock compute time is logged as ``compute`` events.

Each chunk returns ``is_listen`` (the model's own listen/speak state) and generated speech. Output of a
speaking chunk is queued in the same :class:`~mptb.fd.adapters.base.VirtualPlayout` used for the API
agents; a speak -> listen transition while audio is playing is recorded as ``model_yield`` and cuts
the playout.
"""
from __future__ import annotations

import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .base import FDAdapter, SessionConfig, resample


class StreamClock:
    """:class:`~mptb.fd.adapters.base.ExperimentClock`-compatible clock advanced explicitly by stream time."""

    def __init__(self) -> None:
        self._started = False
        self._t = 0.0

    def start(self) -> float:
        """Reset to t = 0."""
        self._started = True
        self._t = 0.0
        return self._t

    @property
    def started(self) -> bool:
        """True once started."""
        return self._started

    def now(self) -> float:
        """Current stream time in seconds."""
        return float(self._t if self._started else 0.0)

    def stamp(self) -> float:
        """Alias of :meth:`now` used for event timestamps."""
        return self.now()

    def advance(self, seconds: float) -> float:
        """Move the clock forward by ``seconds`` of fed audio."""
        if not self._started:
            self.start()
        self._t += max(0.0, float(seconds))
        return self._t


@dataclass
class LocalChunkResult:
    """Output of one model step: the raw result dict, generated audio (or None) and its sample rate."""
    raw: Dict[str, Any]
    output_audio: Optional[np.ndarray]
    output_sample_rate: int


_MODEL_CACHE: Dict[str, Any] = {}


def cached_model(key: str, builder: Callable[[], Any]) -> Any:
    """Build the (multi-GB) model once per process; every later session reuses the weights."""
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = builder()
    return _MODEL_CACHE[key]


def resolve_checkpoint(spec: str, revision: Optional[str] = None, subfolder: Optional[str] = None) -> Path:
    """Local checkpoint directory for ``spec``.

    ``spec`` is either an existing directory (used as is) or a Hugging Face repo id, which is
    downloaded at ``revision`` with :func:`huggingface_hub.snapshot_download`; ``subfolder`` selects a
    model inside a multi-model repository.
    """
    if os.path.isdir(spec):
        return Path(spec)
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(repo_id=spec, revision=revision,
                                   allow_patterns=[f"{subfolder}/*"] if subfolder else None))
    return local / subfolder if subfolder else local


def read_audio_16k(path: Path) -> np.ndarray:
    """Read an audio file as 16 kHz mono float32."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return resample(np.asarray(wav, dtype=np.float32), int(sr), 16000)


def load_duplex_model(cache_key: str, model_dir: Path, ref_audio_path: Path, system_prompt: str):
    """Load a MiniCPM-o-style checkpoint (bf16, SDPA, CUDA) and start a fresh duplex session.

    The weights are cached per process; ``as_duplex`` + ``prepare`` rebuild the session state, so no
    conversational context leaks between trials.
    """
    import torch
    from transformers import AutoModel

    model = cached_model(cache_key, lambda: AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().cuda())
    duplex = model.as_duplex(generate_audio=True)
    ref_audio = read_audio_16k(ref_audio_path) if ref_audio_path.exists() else None
    duplex.prepare(
        prefix_system_prompt=system_prompt or "Streaming Omni Conversation.",
        ref_audio=ref_audio,
        prompt_wav_path=str(ref_audio_path),
    )
    return duplex


class LocalCausalChunkAdapter(FDAdapter):
    """Base class for synchronous local models driven by fixed input chunks in stream time."""

    realtime = False
    native_sample_rate = 16000
    output_sample_rate = 24000
    chunk_seconds = 1.0
    pad_final_chunk = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.clock = StreamClock()
        self.config: Optional[SessionConfig] = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._chunk_index = 0
        self._closed = False
        self._last_is_listen: Optional[bool] = None
        self.compute: List[Dict[str, Any]] = []

    @property
    def chunk_samples(self) -> int:
        """Samples per model step at the native rate."""
        return int(round(self.chunk_seconds * self.native_sample_rate))

    async def start_session(self, config: SessionConfig) -> None:
        """Start the stream clock and prepare a fresh model session."""
        self.config = config
        self.sr_in = int(config.sample_rate_in)
        self.clock.start()
        self.log_event("session_start", {
            "model": self.name,
            "input_sample_rate": self.sr_in,
            "native_sample_rate": self.native_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "chunk_seconds": self.chunk_seconds,
        })
        self._load_or_prepare(config)

    async def send_audio(self, pcm: np.ndarray, timestamp: float) -> None:
        """Buffer input and run one model step for every complete chunk."""
        if self._closed:
            raise RuntimeError(f"{self.name} session is closed")
        x = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self.input_log.append((float(timestamp), int(x.size)))
        if x.size == 0:
            return
        x = resample(x, self.sr_in, self.native_sample_rate)
        self._buf = np.concatenate([self._buf, x])
        while self._buf.size >= self.chunk_samples:
            chunk = self._buf[: self.chunk_samples]
            self._buf = self._buf[self.chunk_samples:]
            self._run_one_chunk(chunk, final=False)

    async def receive_audio(self) -> None:
        """Nothing to drain: output is produced synchronously in :meth:`send_audio`."""
        self.tick()

    async def close_session(self) -> None:
        """Run the zero-padded final partial chunk and close."""
        if not self._closed and self.pad_final_chunk and self._buf.size > 0:
            chunk = np.pad(self._buf, (0, self.chunk_samples - self._buf.size))
            self._buf = np.zeros(0, dtype=np.float32)
            self._run_one_chunk(chunk, final=True)
        self._closed = True
        self.tick()
        self.log_event("session_close", {"chunks_processed": self._chunk_index})

    def _run_one_chunk(self, chunk: np.ndarray, final: bool) -> None:
        idx = self._chunk_index
        stream_start = self.clock.stamp()
        wall0 = time.perf_counter()
        result: Optional[LocalChunkResult] = None
        try:
            result = self._process_model_chunk(chunk.astype(np.float32, copy=False), final=final)
        except Exception as exc:
            self.log_event("model_error", {"chunk": idx, "error": repr(exc)})
            raise
        finally:
            wall_s = time.perf_counter() - wall0
        stream_dur = chunk.size / float(self.native_sample_rate)
        self.clock.advance(stream_dur)
        stream_end = self.clock.stamp()

        raw = result.raw
        is_listen = bool(raw.get("is_listen", False)) if raw else None
        text = raw.get("text", "") if raw else ""
        end_of_turn = bool(raw.get("end_of_turn", False)) if raw else False
        out = result.output_audio
        out_samples = int(np.asarray(out).size) if out is not None else 0
        rec = {
            "chunk": idx,
            "stream_start_s": round(stream_start, 4),
            "stream_end_s": round(stream_end, 4),
            "stream_dur_s": round(stream_dur, 4),
            "wall_s": round(wall_s, 4),
            "rtf": round(wall_s / stream_dur, 4) if stream_dur else None,
            "is_listen": is_listen,
            "end_of_turn": end_of_turn,
            "text": text,
            "output_samples": out_samples,
        }
        for key in ("cost_all", "cost_llm", "cost_tts", "cost_token2wav", "n_tokens", "n_tts_tokens", "current_time"):
            if isinstance(raw, dict) and key in raw:
                try:
                    rec[key] = float(raw[key]) if key.startswith("cost") else raw[key]
                except Exception:
                    rec[key] = str(raw[key])
        self.compute.append(rec)
        self.log_event("compute", rec)

        if end_of_turn:
            self.log_event("end_of_turn", {"chunk": idx, "text": text})
        if self._last_is_listen is False and is_listen is True and self.playout.speaking:
            self.on_cancel("model_yield", {"chunk": idx, "transition": "speak_to_listen"})

        if out is not None and is_listen is not True:
            wav = np.asarray(out, dtype=np.float32).reshape(-1)
            if wav.size and float(np.max(np.abs(wav))) > 1e-5:
                self.on_output_audio(int(wav.size), int(result.output_sample_rate))
        self._last_is_listen = is_listen
        self.tick()
        self._chunk_index += 1

    def trace(self) -> Dict[str, Any]:
        """Base trace plus per-chunk compute records and the overall real-time factor."""
        tr = super().trace()
        total_wall = sum(float(x.get("wall_s") or 0.0) for x in self.compute)
        total_stream = sum(float(x.get("stream_dur_s") or 0.0) for x in self.compute)
        tr["compute_summary"] = {
            "chunks": len(self.compute),
            "total_wall_s": round(total_wall, 4),
            "total_stream_s": round(total_stream, 4),
            "rtf": round(total_wall / total_stream, 4) if total_stream else None,
        }
        tr["compute"] = self.compute
        return tr

    @abstractmethod
    def _load_or_prepare(self, config: SessionConfig) -> None:
        """Load (cached) weights and start a fresh duplex session."""

    @abstractmethod
    def _process_model_chunk(self, chunk: np.ndarray, final: bool = False) -> LocalChunkResult:
        """Feed one 16 kHz chunk and return the model's step output."""
