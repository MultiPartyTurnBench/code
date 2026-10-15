"""Run a baseline over the benchmark, choose its operating point on dev, score test (paper Section 4).

Two system families share one results layout (``<out>/<system>/``):

* **Grid systems** (:class:`GridSystem`: acoustic VADs, Smart Turn EOT, Kyutai EOT, VAP, MuVAP) produce
  committed predictions for every threshold of a fixed grid. ``raw/<id>.json`` holds
  ``{theta: {speaker: {"eot": [...], "interruption": [...]}}}``.
* **Score-stream systems** (:class:`ScoreSystem`: Mimi-EP, DualTurn, turn-1-mini, WavLM anchor, ESPnet,
  Kyutai INT, Smart Turn INT) produce causal per-frame scores. ``scores/<id>.npz`` holds, per speaker,
  the commit time ``t`` of each frame, the ``eot``/``int`` scores and the ``rest`` gate (any other speaker
  voice-active, WebRTC aggressiveness 2). They are decoded with the commit rule (rising edge above
  theta, 2 s refractory; INT additionally requires ``rest``). Candidate thresholds are a fixed grid plus
  quantiles of the system's own dev scores (TurnBench's rule; some native scores sit far below 0.5).

``assemble`` applies the operating-point rule per track: the highest dev recall with dev FPR <= 0.10
(else the lowest-FPR point, reported as over budget), then scores the test split once. It writes
``report.json`` and ``predictions/<track>.json`` (a submission at the selected threshold for every dev and
test conversation).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data import Dataset
from ..eval.schema import SUBMISSION_SCHEMA_VERSION
from ..eval.scorer import DEV_FPR_BUDGET, TRACKS, score_dataset, select_operating_point
from .common import SR16, equal_length_channels, refractory, rest_active, rising_edges, webrtc_masks

SCORE_GRID = [round(x, 3) for x in np.arange(0.05, 0.951, 0.05)] + [0.97, 0.98, 0.99, 0.995]
SCORE_QUANTILES = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.965, 0.975, 0.985,
                   0.99, 0.995, 0.998, 0.999]
METRIC_KEYS = ("tp", "fn", "fp", "tn", "recall", "fpr", "latency_p10", "latency_p50", "latency_p90")


@dataclass
class GridSystem:
    """A system evaluated on a fixed threshold grid.

    ``factory()`` returns an object with ``thetas`` and ``predict(audio, sr)``. ``tie_break`` resolves equal
    dev recall among in-budget thresholds: ``"fpr"`` (lower dev FPR) or ``"theta"`` (lower theta).
    """
    name: str
    description: str
    factory: Callable[[], object]
    tracks: Tuple[str, ...] = TRACKS
    tie_break: str = "fpr"
    int_kind: str = "none"


@dataclass
class ScoreSystem:
    """A system producing per-frame scores; ``factory()`` returns an object with ``sr`` and ``scores(ch, sr)``.

    ``transform`` optionally maps the saved arrays to ``eot``/``int`` scores at assembly time (DualTurn).
    """
    name: str
    description: str
    factory: Callable[[], object]
    tracks: Tuple[str, ...] = TRACKS
    int_kind: str = "none"
    transform: Optional[Callable[[dict], dict]] = field(default=None, repr=False)


# --------------------------------------------------------------------------- results I/O
def system_dir(out: Path, name: str) -> Path:
    return Path(out) / name


def _write_json(p: Path, obj, indent=None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.rename(p)


def save_scores(path: Path, per_spk: Dict[str, Dict[str, np.ndarray]], **meta) -> None:
    """Save per-speaker score arrays as ``<key>__<speaker>`` entries of one compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {f"{k}__{s}": np.asarray(v) for s, d in per_spk.items() for k, v in d.items()}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrs)
    tmp.rename(path)


def load_scores(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    z = np.load(path)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for k in z.files:
        if k == "meta":
            continue
        key, s = k.split("__", 1)
        out.setdefault(s, {})[key] = z[k]
    return out


def submission(preds: Dict[str, Dict[str, Dict[str, List[float]]]], cids: Sequence[str]) -> dict:
    """Wrap ``{cid: {speaker: {"eot", "interruption"}}}`` into a schema-v1 submission."""
    return {"schema_version": SUBMISSION_SCHEMA_VERSION,
            "predictions": [{"conversation_id": c, **preds[c]} for c in cids]}


# --------------------------------------------------------------------------- compute
def compute(system, ds: Dataset, out: Path, shard: int = 0, nshards: int = 1, overwrite: bool = False) -> None:
    """Run the model over conversations ``ids[shard::nshards]`` and save raw predictions or scores."""
    import torch
    torch.set_grad_enabled(False)
    d = system_dir(out, system.name)
    grid = isinstance(system, GridSystem)
    target = (lambda c: d / "raw" / f"{c}.json") if grid else (lambda c: d / "scores" / f"{c}.npz")
    cids = ds.ids()[shard::nshards]
    todo = [c for c in cids if overwrite or not target(c).exists()]
    print(f"[{system.name} shard {shard}/{nshards}] {len(todo)}/{len(cids)} conversations to do", flush=True)
    model = system.factory()
    t0 = time.time()
    for i, cid in enumerate(todo):
        audio, sr = ds.audio(cid, "noisy")
        if grid:
            res = model.predict(audio, sr)
            _write_json(target(cid), {str(float(k)): v for k, v in res.items()})
        else:
            ch = equal_length_channels(audio, sr, model.sr)
            msr = model.sr or sr
            ch16 = ch if msr == SR16 else equal_length_channels(audio, sr, SR16)
            masks = webrtc_masks(ch16, SR16)
            per_spk = model.scores(ch, msr)
            for s, dd in per_spk.items():
                t = np.asarray(dd["t"], np.float64)
                dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
                dd["rest"] = rest_active(masks, s, t, dt)
            save_scores(target(cid), per_spk, model=system.name, sr=msr)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"[{system.name} shard {shard}] {i + 1}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- score-stream decoding
def decode(per_spk: Dict[str, Dict[str, np.ndarray]], th_eot: Optional[float], th_int: Optional[float]
           ) -> Dict[str, Dict[str, List[float]]]:
    """Commit rule on per-frame scores: rising edge of ``score > theta`` (INT also gated by ``rest``)."""
    res = {}
    for s, d in per_spk.items():
        t = d["t"].astype(np.float64)
        eot = refractory(rising_edges(d["eot"] > th_eot, t)) if th_eot is not None and "eot" in d else []
        if th_int is not None and "int" in d:
            intr = refractory(rising_edges((d["int"] > th_int) & d["rest"].astype(bool), t))
        else:
            intr = []
        res[s] = {"eot": eot, "interruption": intr}
    return res


def candidates(dev_scores: Dict[str, dict], key: str) -> List[float]:
    """Threshold candidates: :data:`SCORE_GRID` united with quantiles of the dev scores (INT: gated frames)."""
    vals = []
    for per_spk in dev_scores.values():
        for d in per_spk.values():
            if key not in d:
                continue
            v = d[key] if key == "eot" else d[key][d["rest"].astype(bool)]
            vals.append(np.asarray(v, np.float64))
    if not vals:
        return []
    q = [float(x) for x in np.quantile(np.concatenate(vals), SCORE_QUANTILES)]
    return sorted({round(x, 6) for x in SCORE_GRID + q})


def _score_key(track: str) -> str:
    return "eot" if track == "eot" else "int"


def score_sweep(scores: Dict[str, dict], golds: Dict[str, dict], track: str, thetas: Sequence[float]) -> List[dict]:
    """Pooled dev metrics of a score-stream system at each candidate threshold."""
    pts = []
    for th in thetas:
        te, ti = (th, None) if track == "eot" else (None, th)
        sub = submission({c: decode(scores[c], te, ti) for c in golds}, list(golds))
        m = score_dataset(golds, sub)["overall"][track]
        pts.append({"theta": th, "recall": m["recall"], "fpr": m["fpr"],
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]})
    return pts


def select_score_point(pts: List[dict], budget: float) -> Tuple[dict, bool]:
    """Score-stream rule: thresholds that never fire (TP = 0) are excluded; among in-budget points the
    highest recall wins (ties: lower FPR, then lower theta); otherwise the lowest-FPR point."""
    live = [d for d in pts if d["fpr"] is not None and d["recall"] is not None and d["tp"] > 0]
    ok = [d for d in live if d["fpr"] <= budget + 1e-9]
    if ok:
        return max(ok, key=lambda d: (d["recall"], -d["fpr"], -d["theta"])), True
    return min(live, key=lambda d: (d["fpr"], -d["recall"])), False


def select_grid_point(sweep: List[dict], budget: float, tie_break: str) -> Tuple[Optional[dict], bool]:
    """Grid rule: highest dev recall within budget (ties per ``tie_break``), else the lowest-FPR point."""
    pts = [d for d in sweep if d["fpr"] is not None and d["recall"] is not None]
    if not pts:
        return None, False
    q = [d for d in pts if d["fpr"] <= budget + 1e-9]
    if q:
        key = (lambda d: (d["recall"], -d["fpr"])) if tie_break == "fpr" else (lambda d: d["recall"])
        return max(q, key=key), True
    return min(pts, key=lambda d: (d["fpr"], -d["recall"])), False


# --------------------------------------------------------------------------- assemble
def _track_report(chosen: dict, qualified: bool, rep: dict, track: str) -> dict:
    m = rep["overall"][track]
    return {"theta": chosen["theta"], "qualified_on_dev": qualified,
            "dev_recall": chosen["recall"], "dev_fpr": chosen["fpr"],
            "test": {k: m.get(k) for k in METRIC_KEYS},
            "by_num_speakers": {k: {kk: v[track].get(kk) for kk in METRIC_KEYS}
                                for k, v in rep["by_num_speakers"].items()},
            "by_type": {k: {kk: v[track].get(kk) for kk in METRIC_KEYS}
                        for k, v in rep["by_conversation_type"].items()}}


def _log(name: str, track: str, r: dict) -> None:
    m = r["test"]
    print(f"  {name} {track:12s}: theta={r['theta']} qualified={r['qualified_on_dev']} | dev r={r['dev_recall']:.3f} "
          f"fpr={r['dev_fpr']:.3f} | TEST r={m['recall']:.3f} fpr={m['fpr']:.3f} TP={m['tp']} FN={m['fn']} "
          f"FP={m['fp']} TN={m['tn']} p50={m['latency_p50']}", flush=True)


def assemble(system, ds: Dataset, out: Path, budget: float = DEV_FPR_BUDGET, extra: Optional[dict] = None) -> dict:
    """Choose the operating point per track on dev, score test, write the report and predictions."""
    d = system_dir(out, system.name)
    golds = ds.golds()
    dev = {c: g for c, g in golds.items() if ds.split_of(c) == "dev"}
    test = {c: g for c, g in golds.items() if ds.split_of(c) == "test"}
    tracks, sweeps, preds = {}, {}, {}

    if isinstance(system, GridSystem):
        missing = [c for c in golds if not (d / "raw" / f"{c}.json").exists()]
        if missing:
            raise SystemExit(f"{system.name}: {len(missing)} conversations not computed, e.g. {missing[:3]}")
        raw = {c: json.loads((d / "raw" / f"{c}.json").read_text()) for c in golds}
        thetas = [float(k) for k in raw[next(iter(raw))]]

        def at(th, cids):
            return {c: raw[c][str(float(th))] for c in cids}

        op = select_operating_point(dev, {th: submission(at(th, dev), list(dev)) for th in thetas}, budget=budget)
        test_reports = {}
        for tr in system.tracks:
            sweeps[tr] = op[tr]["sweep"]
            chosen, qualified = select_grid_point(op[tr]["sweep"], budget, system.tie_break)
            th = chosen["theta"]
            if th not in test_reports:
                test_reports[th] = score_dataset(test, submission(at(th, test), list(test)))
            tracks[tr] = _track_report(chosen, qualified, test_reports[th], tr)
            preds[tr] = {c: {s: {"eot": v["eot"] if tr == "eot" else [],
                                 "interruption": v["interruption"] if tr == "interruption" else []}
                             for s, v in raw[c][str(float(th))].items()} for c in golds}
    else:
        missing = [c for c in golds if not (d / "scores" / f"{c}.npz").exists()]
        if missing:
            raise SystemExit(f"{system.name}: {len(missing)} conversations not computed, e.g. {missing[:3]}")
        scores = {c: load_scores(d / "scores" / f"{c}.npz") for c in golds}
        if system.transform is not None:
            scores = {c: system.transform(p) for c, p in scores.items()}
        for tr in system.tracks:
            cand = candidates({c: scores[c] for c in dev}, _score_key(tr))
            if not cand:
                continue
            sweeps[tr] = score_sweep(scores, dev, tr, cand)
            chosen, qualified = select_score_point(sweeps[tr], budget)
            th = chosen["theta"]
            te, ti = (th, None) if tr == "eot" else (None, th)
            preds[tr] = {c: decode(scores[c], te, ti) for c in golds}
            rep = score_dataset(test, submission({c: preds[tr][c] for c in test}, list(test)))
            tracks[tr] = _track_report(chosen, qualified, rep, tr)

    for tr, r in tracks.items():
        _log(system.name, tr, r)
        _write_json(d / "predictions" / f"{tr}.json", submission(preds[tr], list(golds)))
    report = {"system": system.name, "description": system.description, "int_kind": system.int_kind,
              "dev_fpr_budget": budget, "n_dev": len(dev), "n_test": len(test),
              "tracks": tracks, "dev_sweep": sweeps, **(extra or {})}
    _write_json(d / "report.json", report, indent=2)
    print(f"wrote {d / 'report.json'}", flush=True)
    return report


def select_dualturn_mapping(ds: Dataset, out: Path, name: str = "dualturn", budget: float = DEV_FPR_BUDGET
                            ) -> Tuple[str, bool, str]:
    """Choose DualTurn's EOT head (raw or inverted) and INT head on the *dev* split only.

    For each candidate, the best in-budget dev recall (score-stream rule) is computed; the candidate with
    the highest such recall is kept for each track.
    """
    from .dualturn import EOT_CANDIDATES, INT_CANDIDATES
    d = system_dir(out, name)
    dev = {c: ds.gold(c) for c in ds.ids("dev")}
    raw = {c: load_scores(d / "scores" / f"{c}.npz") for c in dev}

    def best(head: str, inverted: bool, track: str) -> float:
        key = _score_key(track)
        view = {c: {s: {"t": v["t"], "rest": v["rest"], key: (1 - v[head] if inverted else v[head])}
                    for s, v in raw[c].items()} for c in dev}
        pts = score_sweep(view, dev, track, candidates(view, key))
        ok = [p for p in pts if p["tp"] > 0 and p["fpr"] <= budget]
        return max((p["recall"], -p["fpr"]) for p in ok)[0] if ok else -1.0

    eot = max(EOT_CANDIDATES, key=lambda hc: best(hc[0], hc[1], "eot"))
    int_head = max(INT_CANDIDATES, key=lambda h: best(h, False, "interruption"))
    return eot[0], eot[1], int_head
