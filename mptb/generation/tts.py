"""Stage 3 -- speech synthesis with Kokoro-82M (paper Section 3.1 step 3).

Each utterance is synthesised independently with Kokoro-82M (``kokoro==0.9.4``, Hugging Face
``hexgrad/Kokoro-82M`` at revision :data:`KOKORO_REVISION`). Abstract voice ids (v01..v24)
map to Kokoro voices through ``configs/voice_binding_kokoro.json`` (gender-matched; prefix
``a``/``b`` = American/British English, which selects the G2P language). Leading/trailing
silence is trimmed at -40 dBFS with a 30 ms margin.

Synthesis is deterministic: the torch RNG is re-seeded per utterance from a hash of
(voice, speed, text). An optional content-hash disk cache makes re-runs cheap; it stores the
float32 waveform and does not change the output.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_BINDING = str(Path(__file__).resolve().parents[2] / "configs" / "voice_binding_kokoro.json")
# Hugging Face revision of hexgrad/Kokoro-82M used for the released corpus.
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
_KOKORO_FILES = ["config.json", "kokoro-v1_0.pth", "voices/*.pt"]


@dataclass
class TurnAudio:
    """One synthesised utterance: float32 mono audio and per-word (text, start_s, end_s)."""
    audio: np.ndarray
    sample_rate: int
    words: List[Tuple[str, float, float]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return len(self.audio) / self.sample_rate


def _trim_bounds(audio: np.ndarray, sr: int, threshold_db: float = -40.0,
                 margin_ms: float = 30.0) -> Tuple[int, int]:
    """Voiced region ``[start, end)`` in samples (10 ms RMS frames above ``threshold_db``), padded by ``margin_ms``."""
    if audio.size == 0:
        return 0, 0
    hop = max(1, int(sr * 0.01))
    n = len(audio) // hop
    if n == 0:
        return 0, len(audio)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    voiced = np.flatnonzero(db > threshold_db)
    if voiced.size == 0:
        return 0, len(audio)
    margin = int(sr * margin_ms / 1000.0)
    start = max(0, voiced[0] * hop - margin)
    end = min(len(audio), (voiced[-1] + 1) * hop + margin)
    return start, end


class KokoroTTS:
    """Kokoro-82M synthesiser for abstract voice ids.

    Args:
        binding_path: voice binding JSON (``binding: {voice_id: {kokoro_voice, ...}}``, ``repo_id``).
        device: torch device for the model.
        revision: Hugging Face revision of the Kokoro repository (pinned by default).
        cache_dir: optional directory for the synthesis cache (default ``$MPTB_TTS_CACHE`` or none).
    """

    def __init__(self, binding_path: str = DEFAULT_BINDING, device: str = "cuda",
                 sample_rate: int = 24000, speed: float = 1.0, base_seed: int = 0,
                 revision: str = KOKORO_REVISION, cache_dir: Optional[str] = None,
                 trim_silence: bool = True):
        cfg = json.loads(Path(binding_path).read_text())
        self.binding: Dict[str, dict] = cfg["binding"]
        self.repo_id: str = cfg.get("repo_id", "hexgrad/Kokoro-82M")
        self.revision = revision
        self.sample_rate = sample_rate
        self.speed = speed
        self.base_seed = base_seed
        self.device = device
        self.trim_silence = trim_silence
        self._pipes: Dict[str, object] = {}
        self._model = None
        self._snapshot: Optional[Path] = None
        cache_dir = cache_dir or os.environ.get("MPTB_TTS_CACHE")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def kokoro_voice(self, voice_id: str) -> str:
        """Kokoro voice name bound to an abstract voice id (e.g. ``v01`` -> ``af_alloy``)."""
        return self.binding[voice_id]["kokoro_voice"]

    def _lang_for(self, kokoro_voice: str) -> str:
        return "b" if kokoro_voice[0] == "b" else "a"

    def _snapshot_dir(self) -> Path:
        if self._snapshot is None:
            from huggingface_hub import snapshot_download
            self._snapshot = Path(snapshot_download(self.repo_id, revision=self.revision,
                                                    allow_patterns=_KOKORO_FILES))
        return self._snapshot

    def _pipe_for(self, lang_code: str):
        if lang_code not in self._pipes:
            from kokoro import KModel, KPipeline
            if self._model is None:
                snap = self._snapshot_dir()
                self._model = KModel(repo_id=self.repo_id, config=str(snap / "config.json"),
                                     model=str(snap / "kokoro-v1_0.pth")).to(self.device).eval()
            self._pipes[lang_code] = KPipeline(lang_code=lang_code, repo_id=self.repo_id,
                                               model=self._model, device=self.device)
        return self._pipes[lang_code]

    def _cache_key(self, text: str, kv: str) -> str:
        blob = f"{self.repo_id}|{kv}|{self.speed}|{self.base_seed}|{self.trim_silence}|{text}"
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

    def _cache_load(self, key: str) -> Optional[TurnAudio]:
        if not self.cache_dir:
            return None
        npy = self.cache_dir / f"{key}.npy"
        meta = self.cache_dir / f"{key}.json"
        if npy.exists() and meta.exists():
            m = json.loads(meta.read_text())
            return TurnAudio(audio=np.load(npy), sample_rate=m["sample_rate"],
                             words=[tuple(w) for w in m["words"]])
        return None

    def _cache_save(self, key: str, ta: TurnAudio) -> None:
        if not self.cache_dir:
            return
        np.save(self.cache_dir / f"{key}.npy", ta.audio.astype(np.float32))
        (self.cache_dir / f"{key}.json").write_text(json.dumps(
            {"sample_rate": ta.sample_rate, "words": [list(w) for w in ta.words]}))

    def _seed_for(self, text: str, kv: str) -> int:
        key = f"{self.base_seed}|{kv}|{self.speed}|{text}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2 ** 31)

    def synth(self, text: str, voice_id: str) -> TurnAudio:
        """Synthesise ``text`` with abstract voice ``voice_id`` (trimmed, with word timings)."""
        kv = self.kokoro_voice(voice_id)
        ck = self._cache_key(text, kv)
        hit = self._cache_load(ck)
        if hit is not None:
            return hit
        import torch
        torch.manual_seed(self._seed_for(text, kv))
        pipe = self._pipe_for(self._lang_for(kv))
        voice_file = str(self._snapshot_dir() / "voices" / f"{kv}.pt")
        chunks: List[np.ndarray] = []
        words: List[Tuple[str, float, float]] = []
        offset = 0.0
        for r in pipe(text, voice=voice_file, speed=self.speed):
            if r.audio is not None:
                a = r.audio.detach().cpu().numpy().astype(np.float32)
                for tok in (r.tokens or []):
                    if tok.start_ts is not None and (tok.text or "").strip():
                        words.append((tok.text, float(tok.start_ts) + offset,
                                      float(tok.end_ts) + offset))
                offset += len(a) / self.sample_rate
                chunks.append(a)
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        if self.trim_silence:
            s, e = _trim_bounds(audio, self.sample_rate)
            audio = audio[s:e]
            shift = s / self.sample_rate
            words = [(w, max(0.0, st - shift), max(0.0, en - shift)) for (w, st, en) in words]
        ta = TurnAudio(audio=audio, sample_rate=self.sample_rate, words=words)
        self._cache_save(ck, ta)
        return ta

    def duration_fn(self, voice_map: Dict[str, str]) -> Callable[[str, str], float]:
        """A Stage-2 ``duration_fn(text, speaker)`` resolving speakers via ``{speaker: voice_id}``."""
        def _fn(text: str, speaker: str) -> float:
            return self.synth(text, voice_map[speaker]).duration_s
        return _fn
