"""MultiPartyTurnBench scorer — TurnBench protocol, generalized to N speakers.

Port of the TurnBench scoring protocol (Jiang et al., 2026). The scorer consumes a
validated gold conversation and a submission (committed event times per speaker and
track) and produces recall, FPR and signed-latency percentiles for the EOT and INT tracks.

Matching (per speaker, per track):
  * For each gold positive at time t, the TP is the EARLIEST UNCLAIMED prediction
    in the window [t - tau_pre, t + tau_max]  (tau_pre=0.25, tau_max=3.0).
    Positives are processed in ascending time; matching is one-to-one.
  * Predictions inside an EXCLUDED interval are dropped (ignored) before matching.
  * Recall = TP / (TP + FN).
  * FPR    = FP / (FP + TN), computed PER NEGATIVE SPAN: a span is one FP if any
    remaining (unclaimed) prediction lands in it, else one TN. Predictions that
    are neither a TP nor inside a negative span are ignored.
  * Latency = signed (t_pred - t_gold) over matched TPs; report p10/p50/p90.
    Negative latency == the model committed before the gold boundary.

The rising-edge / 2 s-refractory commit rule belongs to the inference harness
(:mod:`mptb.systems.decode`), not to the scorer: submissions already contain committed
event times. Operating-point selection (threshold sweep within an FPR
budget) is `select_operating_point`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .anchors import TrackAnchors, derive_anchors

TAU_PRE = 0.25   # matching tolerance before the gold boundary (s)
TAU_MAX = 3.0    # latency deadline after the gold boundary (s)
DEV_FPR_BUDGET = 0.10
TEST_FPR_CEILING = 0.15
TRACKS = ("eot", "interruption")


# ----------------------------------------------------------------------------
# Confusion counts + metrics
# ----------------------------------------------------------------------------

@dataclass
class TrackResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    latencies: List[float] = field(default_factory=list)

    def __iadd__(self, other: "TrackResult") -> "TrackResult":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        self.latencies.extend(other.latencies)
        return self

    @property
    def recall(self) -> Optional[float]:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def fpr(self) -> Optional[float]:
        d = self.fp + self.tn
        return self.fp / d if d else None

    def metrics(self) -> Dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "n_positives": self.tp + self.fn,
            "n_negative_spans": self.fp + self.tn,
            "recall": self.recall,
            "fpr": self.fpr,
            "latency_p10": percentile(self.latencies, 10),
            "latency_p50": percentile(self.latencies, 50),
            "latency_p90": percentile(self.latencies, 90),
            "n_matched": len(self.latencies),
        }


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (matches numpy's default 'linear')."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return float(xs[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


# ----------------------------------------------------------------------------
# Core per-track matching
# ----------------------------------------------------------------------------

def _in_any_span(p: float, spans: Iterable[Tuple[float, float]]) -> bool:
    return any(s <= p <= e for (s, e) in spans)


def score_track(anchors: TrackAnchors, preds: Sequence[float],
                tau_pre: float = TAU_PRE, tau_max: float = TAU_MAX) -> TrackResult:
    preds_sorted = sorted(float(p) for p in preds)
    # 1. drop predictions that fall inside an excluded interval
    kept = [p for p in preds_sorted if not _in_any_span(p, anchors.excluded_spans)]
    claimed = [False] * len(kept)

    res = TrackResult()
    # 2. greedy earliest-first matching of positives -> earliest unclaimed pred in window
    for t in sorted(anchors.positives):
        lo, hi = t - tau_pre, t + tau_max
        chosen = None
        for i, p in enumerate(kept):
            if p > hi:
                break
            if claimed[i] or p < lo:
                continue
            chosen = i
            break
        if chosen is None:
            res.fn += 1
        else:
            claimed[chosen] = True
            res.tp += 1
            res.latencies.append(kept[chosen] - t)

    # 3. negative spans: each span is 1 FP if any remaining unclaimed pred lands in it
    remaining = [kept[i] for i in range(len(kept)) if not claimed[i]]
    for span in anchors.negative_spans:
        if _in_any_span_pred(remaining, span):
            res.fp += 1
        else:
            res.tn += 1
    return res


def _in_any_span_pred(preds: Sequence[float], span: Tuple[float, float]) -> bool:
    s, e = span
    return any(s <= p <= e for p in preds)


# ----------------------------------------------------------------------------
# Conversation- and dataset-level scoring
# ----------------------------------------------------------------------------

def _pred_for(sub_conv: Dict[str, Any], speaker: str, track: str) -> List[float]:
    field_name = "eot" if track == "eot" else "interruption"
    spk = sub_conv.get(speaker, {}) if sub_conv else {}
    vals = spk.get(field_name, []) if isinstance(spk, dict) else []
    return [float(v) for v in vals]


def score_conversation(gold: Dict[str, Any],
                       sub_conv: Optional[Dict[str, Any]]) -> Dict[str, TrackResult]:
    """Score one conversation; returns {'eot': TrackResult, 'interruption': TrackResult}
    pooled over all speakers of the conversation."""
    anchors = derive_anchors(gold)
    out = {t: TrackResult() for t in TRACKS}
    for spk, sa in anchors.items():
        out["eot"] += score_track(sa.eot, _pred_for(sub_conv, spk, "eot"))
        out["interruption"] += score_track(sa.interruption,
                                            _pred_for(sub_conv, spk, "interruption"))
    return out


def score_dataset(golds: Dict[str, Dict[str, Any]],
                  submission: Dict[str, Any]) -> Dict[str, Any]:
    """Score a whole dataset.

    golds: {conversation_id: gold_dict}
    submission: validated submission object.
    Returns pooled metrics plus per-N and per-conversation-type breakouts.
    """
    sub_by_cid = {p["conversation_id"]: p for p in submission.get("predictions", [])}

    overall = {t: TrackResult() for t in TRACKS}
    by_n: Dict[int, Dict[str, TrackResult]] = {}
    by_type: Dict[str, Dict[str, TrackResult]] = {}

    for cid, gold in golds.items():
        conv_res = score_conversation(gold, sub_by_cid.get(cid))
        n = gold["num_speakers"]
        ctype = gold["conversation_type"]
        by_n.setdefault(n, {t: TrackResult() for t in TRACKS})
        by_type.setdefault(ctype, {t: TrackResult() for t in TRACKS})
        for t in TRACKS:
            overall[t] += conv_res[t]
            by_n[n][t] += conv_res[t]
            by_type[ctype][t] += conv_res[t]

    def pack(group: Dict[str, TrackResult]) -> Dict[str, Any]:
        return {t: group[t].metrics() for t in TRACKS}

    return {
        "n_conversations": len(golds),
        "overall": pack(overall),
        "by_num_speakers": {str(k): pack(v) for k, v in sorted(by_n.items())},
        "by_conversation_type": {k: pack(v) for k, v in sorted(by_type.items())},
    }


# ----------------------------------------------------------------------------
# Operating-point selection (threshold sweep within an FPR budget)
# ----------------------------------------------------------------------------

def select_operating_point(golds: Dict[str, Dict[str, Any]],
                           submissions_by_theta: Dict[float, Dict[str, Any]],
                           budget: float = DEV_FPR_BUDGET) -> Dict[str, Any]:
    """Given a family of submissions at different thresholds theta, pick — per
    track, independently — the theta with the highest recall whose FPR <= budget.

    Returns {track: {theta, recall, fpr, ...}} plus the full sweep for auditing.
    """
    sweep: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TRACKS}
    for theta, sub in submissions_by_theta.items():
        report = score_dataset(golds, sub)
        for t in TRACKS:
            m = report["overall"][t]
            sweep[t].append({"theta": theta, "recall": m["recall"], "fpr": m["fpr"],
                             "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]})

    chosen: Dict[str, Any] = {}
    for t in TRACKS:
        pts = sorted(sweep[t], key=lambda d: d["theta"])
        eligible = [d for d in pts
                    if d["fpr"] is not None and d["fpr"] <= budget + 1e-9]
        if eligible:
            # highest recall within budget; tie-break: lower FPR, then lower theta
            best = max(eligible, key=lambda d: (d["recall"] if d["recall"] is not None else -1,
                                                -d["fpr"], -d["theta"]))
        else:
            best = None
        chosen[t] = {"selected": best, "budget": budget, "sweep": pts}
    return chosen
