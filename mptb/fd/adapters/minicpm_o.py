"""MiniCPM-o-4.5 local full-duplex adapter (``minicpm-o-4.5``; Table 4, Appendix F).

Checkpoint: ``openbmb/MiniCPM-o-4_5`` at revision :data:`REVISION` (override with the ``model_dir``
adapter argument or ``$MPTB_MINICPM_O_DIR``, either a local directory or a repo id). Reference voice:
``assets/HT_ref_audio.wav`` of the checkpoint. Duplex mode with audio generation, 1.0 s chunks,
``max_new_speak_tokens_per_chunk=20``, sampling decode (default sampling parameters).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import SessionConfig
from .local_base import LocalCausalChunkAdapter, LocalChunkResult, load_duplex_model, resolve_checkpoint

REPO_ID = "openbmb/MiniCPM-o-4_5"
REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"


class MiniCPMOAdapter(LocalCausalChunkAdapter):
    """MiniCPM-o-4.5 in duplex mode, driven in 1.0 s causal chunks."""

    name = "minicpm-o-4.5"
    native_sample_rate = 16000
    output_sample_rate = 24000
    chunk_seconds = 1.0

    def __init__(self, model_dir: Optional[str] = None, ref_audio: Optional[str] = None, **kw: Any):
        super().__init__(**kw)
        self.model_spec = model_dir or os.environ.get("MPTB_MINICPM_O_DIR") or REPO_ID
        self.ref_audio = ref_audio

    def _load_or_prepare(self, config: SessionConfig) -> None:
        model_dir = resolve_checkpoint(self.model_spec, REVISION if self.model_spec == REPO_ID else None)
        ref_audio_path = Path(self.ref_audio) if self.ref_audio else model_dir / "assets" / "HT_ref_audio.wav"
        self.log_event("model_load_start", {"model": self.model_spec, "ref_audio": ref_audio_path.name})
        self.model = load_duplex_model(f"minicpm::{model_dir}", model_dir, ref_audio_path, config.system_prompt)
        self.ref_audio_path = str(ref_audio_path)
        self.log_event("model_load_done", {"model": self.model_spec})

    def _process_model_chunk(self, chunk: np.ndarray, final: bool = False) -> LocalChunkResult:
        self.model.streaming_prefill(
            audio_waveform=chunk,
            frame_list=[],
            max_slice_nums=1,
            batch_vision_feed=False,
        )
        result: Dict[str, Any] = self.model.streaming_generate(
            prompt_wav_path=self.ref_audio_path,
            max_new_speak_tokens_per_chunk=20,
            decode_mode="sampling",
        )
        return LocalChunkResult(raw=result, output_audio=result.get("audio_waveform"),
                                output_sample_rate=self.output_sample_rate)
