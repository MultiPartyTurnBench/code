"""Corpus statistics written to ``metadata/dataset_stats.json`` (paper Section 3.2 and Appendix D).

All statistics are computed from the gold event logs; anchor counts use
:func:`mptb.eval.anchors.derive_anchors`, so they match the scorer's denominators exactly.
Standard library only.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from ..eval.anchors import derive_anchors

_VOCAL_LABELS = ("TURN", "CLAIM_FLOOR", "VOCALIZE_NO_CLAIM")


def overlap_seconds(events: List[Dict[str, Any]]) -> float:
    """Total time (s) during which at least two distinct speakers vocalize.

    Vocalizations are the [onset, offset] spans of TURN / CLAIM_FLOOR / VOCALIZE_NO_CLAIM
    events (events without an offset count as instantaneous).
    """
    spans = []
    for ev in events:
        on = ev.get("onset")
        off = ev.get("offset", on)
        if on is None or off is None:
            continue
        if ev.get("label") in _VOCAL_LABELS:
            spans.append((float(on), float(off), ev.get("speaker")))
    pts = []
    for on, off, spk in spans:
        pts.append((on, 1, spk))
        pts.append((off, -1, spk))
    pts.sort()
    active: Dict[str, int] = defaultdict(int)
    ov = 0.0
    last = None
    for t, d, spk in pts:
        if last is not None and sum(1 for v in active.values() if v > 0) >= 2:
            ov += t - last
        active[spk] += d
        last = t
    return ov


def conv_stats(gold: Dict[str, Any]) -> Dict[str, float]:
    """Per-conversation counts: duration, anchors, TURN/backchannel/interruption counts,
    speaker changes, mean TURN length, overlap seconds and total events."""
    events = gold["events"]
    anchors = derive_anchors(gold)
    turns = sorted([e for e in events if e.get("label") == "TURN"], key=lambda e: e.get("onset", 0))
    n_turns = len(turns)
    changes = 0
    prev = None
    for t in turns:
        h = t.get("speaker")
        if prev is not None and h != prev:
            changes += 1
        prev = h
    turn_len = sum(float(e.get("offset", 0)) - float(e.get("onset", 0)) for e in turns) / max(n_turns, 1)
    return dict(
        dur=float(gold["duration_s"]),
        eot_pos=sum(len(a.eot.positives) for a in anchors.values()),
        eot_neg=sum(len(a.eot.negative_spans) for a in anchors.values()),
        int_pos=sum(len(a.interruption.positives) for a in anchors.values()),
        int_neg=sum(len(a.interruption.negative_spans) for a in anchors.values()),
        n_turns=n_turns,
        n_bc=sum(1 for e in events if e.get("label") == "VOCALIZE_NO_CLAIM"),
        n_int=sum(1 for e in events if e.get("label") == "CLAIM_FLOOR"
                  and e.get("resolved_as") == "INTERRUPTION"),
        changes=changes, turn_len=turn_len, overlap=overlap_seconds(events),
        n_events=len(events))


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    """Aggregate :func:`conv_stats` rows into one ``dataset_stats.json`` group (rounded)."""
    n = len(rows)
    if n == 0:
        return {}
    dur_min = sum(r["dur"] for r in rows) / 60.0

    def tot(k: str) -> float:
        return sum(r[k] for r in rows)

    return dict(
        n_conv=n,
        hours=round(sum(r["dur"] for r in rows) / 3600.0, 3),
        mean_dur_s=round(sum(r["dur"] for r in rows) / n, 1),
        mean_turn_len_s=round(sum(r["turn_len"] for r in rows) / n, 2),
        speaker_changes_per_dialog=round(tot("changes") / n, 1),
        overlap_s_per_dialog=round(tot("overlap") / n, 1),
        events_per_min=round(tot("n_events") / dur_min, 2),
        bc_per_min=round(tot("n_bc") / dur_min, 2),
        int_per_min=round(tot("n_int") / dur_min, 2),
        eot_anchors=tot("eot_pos"),
        eot_negatives=tot("eot_neg"),
        int_anchors=tot("int_pos"),
        int_negatives=tot("int_neg"),
    )


def dataset_stats(golds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The ``metadata/dataset_stats.json`` object: overall, by party count N and by type."""
    per_conv = {cid: conv_stats(g) for cid, g in golds.items()}
    by_n: Dict[int, list] = defaultdict(list)
    by_type: Dict[str, list] = defaultdict(list)
    for cid, g in golds.items():
        by_n[g["num_speakers"]].append(per_conv[cid])
        by_type[g["conversation_type"]].append(per_conv[cid])
    return {
        "overall": aggregate(list(per_conv.values())),
        "by_N": {str(n): aggregate(by_n[n]) for n in sorted(by_n)},
        "by_type": {t: aggregate(by_type[t]) for t in sorted(by_type)},
    }
