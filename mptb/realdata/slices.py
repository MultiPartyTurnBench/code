"""Run the offline baselines on a real-audio slice and select their operating point in-slice (paper Table 14).

Both slices are scored with :func:`mptb.eval.scorer.score_track` on per-speaker anchors, streaming one
conversation at a time. For every backend, each threshold of its grid (:data:`BACKENDS`) is scored on the
whole slice, and the operating point per track is the highest recall with FPR <= budget (0.10), else the
lowest-FPR point (reported as not ``qualified``). There is no dev/test split on the real slices, so the
selected numbers are optimistic. Systems come from :mod:`mptb.systems`:

* ``energy``     - RMS energy VAD (EOT + INT), :class:`mptb.systems.vad.VADBaseline`;
* ``smart_turn`` - Smart Turn v2 on energy-VAD candidates (EOT only), :class:`mptb.systems.smart_turn.SmartTurnEOT`;
* ``vap``        - VAP "s vs rest" (EOT + INT), :class:`mptb.systems.vap.VAP`, with the generic pretrained
  VAP checkpoint shipped in the VAP repository (:data:`VAP_SLICE_CHECKPOINT`), which is the checkpoint
  the paper's real-slice VAP rows were computed with;
* ``kyutai``     - Kyutai Semantic-VAD (EOT only), :class:`mptb.systems.kyutai.KyutaiEOT`.

Data layout expected (written by ``scripts/realdata/prepare_turnbench.py`` / ``build_ami_slice.py``)::

    turnbench: <data>/anchors.json, <data>/audio16k/<cid>_speaker_{1,2}.flac
    ami:       <data>/gold/<meeting>.json (audio file names relative to --audio-dir)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..eval.anchors import SpeakerAnchors, derive_anchors
from ..eval.scorer import DEV_FPR_BUDGET, TrackResult, percentile, score_track
from .turnbench import SPEAKERS, anchors_from_json

TRACKS = ("eot", "interruption")
SR = 16000
VAP_SLICE_CHECKPOINT = "example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"  # relative to the VAP repo


@dataclass(frozen=True)
class Backend:
    """A baseline as run on the real slices.

    ``thetas`` is the slice threshold grid, ``tracks`` the scored tracks, and ``tie_fpr`` whether equal
    in-budget recall is broken by lower FPR (otherwise the first theta of the grid wins).
    """
    thetas: Tuple[float, ...]
    tracks: Tuple[str, ...] = TRACKS
    tie_fpr: bool = False
    trim_channels: bool = True


BACKENDS: Dict[str, Backend] = {
    "energy": Backend((-50.0, -45.0, -40.0, -35.0, -30.0)),
    "smart_turn": Backend((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), tracks=("eot",)),
    "vap": Backend((0.5, 0.7, 0.8, 0.9, 0.95)),
    "kyutai": Backend((0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), tracks=("eot",), tie_fpr=True,
                      trim_channels=False),
}


def make_predictor(backend: str, device: str = "cuda", smart_turn_dir: str = None, vap_repo: str = None,
                   vap_checkpoint: str = None):
    """Instantiate a :mod:`mptb.systems` predictor restricted to the slice threshold grid."""
    if backend == "energy":
        from ..systems.vad import VADBaseline
        p = VADBaseline("energy")
    elif backend == "smart_turn":
        from ..systems.smart_turn import SmartTurnEOT
        p = SmartTurnEOT(smart_turn_dir, device)
    elif backend == "vap":
        from ..systems.vap import VAP
        vap_repo = vap_repo or os.environ.get("VAP_REPO")
        if not vap_checkpoint and vap_repo:
            vap_checkpoint = os.path.join(vap_repo, VAP_SLICE_CHECKPOINT)
        p = VAP(device, vap_repo, vap_checkpoint)
    elif backend == "kyutai":
        from ..systems.kyutai import KyutaiEOT
        p = KyutaiEOT(device)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    p.thetas = BACKENDS[backend].thetas
    return p


# --------------------------------------------------------------------------------------------- slices
def _read_mono(path: str) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    return (a.mean(axis=1) if a.ndim > 1 else a), sr


def load_turnbench_slice(data_dir: str) -> Tuple[Dict[str, Dict[str, SpeakerAnchors]],
                                                 Callable[[str, bool], Dict[str, np.ndarray]]]:
    """``(anchors_by_conversation, load_audio)`` for the TurnBench dev slice in ``data_dir``.

    ``load_audio(cid, trim)`` returns the two 16 kHz channels (TurnBench channels are not trimmed).
    """
    cache = json.loads((Path(data_dir) / "anchors.json").read_text())
    anchors = {cid: anchors_from_json(d) for cid, d in cache.items()}
    audio_dir = Path(data_dir) / "audio16k"

    def load_audio(cid: str, trim: bool) -> Dict[str, np.ndarray]:
        return {s: _read_mono(str(audio_dir / f"{cid}_{s}.flac"))[0] for s in SPEAKERS}
    return anchors, load_audio


def load_ami_slice(data_dir: str, audio_dir: str) -> Tuple[Dict[str, Dict[str, SpeakerAnchors]],
                                                           Callable[[str, bool], Dict[str, np.ndarray]]]:
    """``(anchors_by_meeting, load_audio)`` for the AMI slice built in ``data_dir``.

    ``load_audio(meeting, trim)`` returns the 16 kHz headset channels, trimmed to the shortest if ``trim``.
    """
    golds = {}
    for f in sorted((Path(data_dir) / "gold").glob("*.json")):
        g = json.loads(f.read_text())
        golds[g["conversation_id"]] = g
    anchors = {cid: derive_anchors(g) for cid, g in golds.items()}

    def load_audio(cid: str, trim: bool) -> Dict[str, np.ndarray]:
        out = {}
        for s, name in golds[cid]["audio"]["files"].items():
            a, sr = _read_mono(os.path.join(audio_dir, name))
            if sr != SR:
                raise ValueError(f"{name}: expected 16 kHz AMI headset audio, got {sr} Hz")
            out[s] = a
        if trim:
            n = min(len(a) for a in out.values())
            out = {s: a[:n] for s, a in out.items()}
        return out
    return anchors, load_audio


# --------------------------------------------------------------------------------------------- scoring
def select_point(sweep: List[Dict[str, Any]], budget: float, tie_fpr: bool) -> Dict[str, Any]:
    """In-slice operating point: highest recall with FPR <= budget, else the lowest FPR (then highest recall)."""
    pts = [d for d in sweep if d["fpr"] is not None and d["recall"] is not None]
    if not pts:
        return {"theta": None}
    q = [d for d in pts if d["fpr"] <= budget + 1e-9]
    if q:
        chosen = max(q, key=(lambda d: (d["recall"], -d["fpr"])) if tie_fpr else (lambda d: d["recall"]))
    else:
        chosen = min(pts, key=lambda d: (d["fpr"], -d["recall"]))
    chosen = dict(chosen)
    chosen["qualified"] = bool(q)
    return chosen


def run_backend(backend: str, predictor, anchors_by_cid: Dict[str, Dict[str, SpeakerAnchors]],
                load_audio: Callable[[str, bool], Dict[str, np.ndarray]], budget: float = DEV_FPR_BUDGET,
                log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Score ``predictor`` over the slice for every theta; returns the report entry ``{tracks, sweep}``."""
    spec = BACKENDS[backend]
    thetas = list(spec.thetas)
    agg = {th: {tr: TrackResult() for tr in spec.tracks} for th in thetas}
    for cid, spk_anchors in anchors_by_cid.items():
        per_theta = predictor.predict(load_audio(cid, spec.trim_channels), SR)
        for th in thetas:
            p = per_theta[float(th)]
            for s, sa in spk_anchors.items():
                for tr in spec.tracks:
                    agg[th][tr] += score_track(getattr(sa, tr), p.get(s, {}).get(tr, []))
        del per_theta
        log(f"  {backend}: scored {cid}")
    sweep = {tr: [] for tr in spec.tracks}
    for th in thetas:
        for tr in spec.tracks:
            r = agg[th][tr]
            sweep[tr].append({"theta": th, "recall": r.recall, "fpr": r.fpr, "lat_p50": percentile(r.latencies, 50),
                              "n_pos": r.tp + r.fn, "n_neg": r.fp + r.tn})
    entry: Dict[str, Any] = {"tracks": {tr: select_point(sweep[tr], budget, spec.tie_fpr) for tr in spec.tracks},
                             "sweep": sweep}
    if spec.tracks == ("eot",):
        entry["eot_only"] = True
    return entry
