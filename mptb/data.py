"""Access to a local copy of the MultiPartyTurnBench dataset release.

The dataset (Hugging Face: ``MultiPartyTurnBench``) is laid out as::

    <root>/metadata/split.json                       {"dev": [ids], "test": [ids]}
    <root>/<split>/annotations/{events,anchors,scripts,render}/<id>.json
    <root>/<split>/audio/{noisy,clean}/<id>_speaker_<k>.flac

``events`` is the gold event log consumed by :mod:`mptb.eval`. Systems are evaluated on the
``noisy`` channels. The root can be passed explicitly or taken from ``$MPTB_ROOT``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import numpy as np

SPLITS = ("dev", "test")


def default_root() -> Optional[str]:
    """Dataset root from ``$MPTB_ROOT`` (None if unset)."""
    return os.environ.get("MPTB_ROOT")


class Dataset:
    """Read-only view of a downloaded MultiPartyTurnBench release."""

    def __init__(self, root: Union[str, Path, None] = None):
        root = root or default_root()
        if not root:
            raise ValueError("dataset root not given: pass root=... or set MPTB_ROOT")
        self.root = Path(root)
        split_file = self.root / "metadata" / "split.json"
        if not split_file.is_file():
            raise FileNotFoundError(f"{self.root} is not a MultiPartyTurnBench release ({split_file} missing)")
        self._ids: Dict[str, List[str]] = json.loads(split_file.read_text())
        self._split_of = {c: s for s in SPLITS for c in self._ids[s]}

    # ------------------------------------------------------------------ ids
    def ids(self, split: Optional[str] = None) -> List[str]:
        """Sorted conversation ids of one split, or of all splits (dev first) if ``split`` is None."""
        if split is None:
            return [c for s in SPLITS for c in sorted(self._ids[s])]
        return sorted(self._ids[split])

    def split_of(self, cid: str) -> str:
        return self._split_of[cid]

    @property
    def split_map(self) -> Dict[str, str]:
        """{conversation_id: "dev" | "test"}."""
        return dict(self._split_of)

    # ---------------------------------------------------------- annotations
    def _annotation(self, kind: str, cid: str) -> dict:
        return json.loads((self.root / self.split_of(cid) / "annotations" / kind / f"{cid}.json").read_text())

    def gold(self, cid: str) -> dict:
        """Validated gold event log of one conversation."""
        from .eval.schema import assert_valid_gold
        g = self._annotation("events", cid)
        assert_valid_gold(g)
        return g

    def golds(self, split: Optional[str] = None) -> Dict[str, dict]:
        """{conversation_id: gold} for one split (or all)."""
        return {c: self.gold(c) for c in self.ids(split)}

    def render(self, cid: str) -> dict:
        return self._annotation("render", cid)

    def script(self, cid: str) -> dict:
        return self._annotation("scripts", cid)

    # ---------------------------------------------------------------- audio
    def audio_path(self, cid: str, speaker: str, kind: str = "noisy") -> Path:
        return self.root / self.split_of(cid) / "audio" / kind / f"{cid}_{speaker}.flac"

    def audio(self, cid: str, kind: str = "noisy") -> Tuple[Dict[str, "np.ndarray"], int]:
        """Per-speaker mono float32 channels of one conversation and their sample rate (24 kHz).

        ``kind`` is ``"noisy"`` (the evaluation input) or ``"clean"``.
        """
        import soundfile as sf
        audio, sr = {}, None
        for spk in self.gold(cid)["speakers"]:
            a, sr = sf.read(str(self.audio_path(cid, spk, kind)), dtype="float32")
            audio[spk] = a.mean(axis=1) if a.ndim > 1 else a
        return audio, sr
