"""Corpus-level consistency checks for a generated corpus (run before packaging or using it).

* :func:`check_oracle_round_trip` -- an oracle submission that fires exactly at every gold
  positive must score recall 1.0 with zero false positives on both tracks (validates the
  events -> anchors -> scorer chain);
* :func:`check_qc`               -- every conversation passes acoustic QC (:mod:`.qc`);
* :func:`check_balance`          -- every (N x conversation type) cell is populated, equally;
* :func:`check_split`            -- dev and test use disjoint voices and are each balanced.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from ..eval.anchors import derive_anchors
from ..eval.schema import CONVERSATION_TYPES, N_MAX, N_MIN, SUBMISSION_SCHEMA_VERSION
from ..eval.scorer import score_dataset

_CELLS = [(n, t) for n in range(N_MIN, N_MAX + 1) for t in CONVERSATION_TYPES]


def perfect_submission(golds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Submission that predicts exactly every gold positive anchor (oracle upper bound)."""
    preds = []
    for cid, gold in golds.items():
        pr: Dict[str, Any] = {"conversation_id": cid}
        for spk, sa in derive_anchors(gold).items():
            pr[spk] = {"eot": sorted(sa.eot.positives),
                       "interruption": sorted(sa.interruption.positives)}
        preds.append(pr)
    return {"schema_version": SUBMISSION_SCHEMA_VERSION, "predictions": preds}


def check_oracle_round_trip(golds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Score :func:`perfect_submission`; passes iff recall == 1 and FP == 0 on both tracks."""
    rep = score_dataset(golds, perfect_submission(golds))
    tracks, ok = {}, True
    for tr in ("eot", "interruption"):
        m = rep["overall"][tr]
        ok = ok and m["recall"] == 1.0 and m["fp"] == 0 and m["fpr"] in (0.0, None)
        tracks[tr] = {"recall": m["recall"], "fpr": m["fpr"], "fp": m["fp"],
                      "n_positives": m["n_positives"]}
    return {"pass": ok, "tracks": tracks}


def check_qc(qc_report: Dict[str, Any]) -> Dict[str, Any]:
    """Pass iff every conversation in a ``qc_report.json`` passed."""
    failed = [r["conversation_id"] for r in qc_report["results"] if not r["passed"]]
    return {"pass": not failed, "n_passed": qc_report["n_passed"],
            "n_conversations": qc_report["n_conversations"], "failed": failed}


def check_balance(golds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Pass iff every (N, type) cell is populated; also reports whether counts are equal."""
    cells = Counter((g["num_speakers"], g["conversation_type"]) for g in golds.values())
    missing = [f"N{n}/{t}" for (n, t) in _CELLS if cells[(n, t)] == 0]
    return {"pass": not missing, "balanced": len({cells[c] for c in _CELLS}) == 1,
            "missing_cells": missing,
            "per_cell_counts": {f"N{n}/{t}": cells[(n, t)] for (n, t) in _CELLS}}


def check_split(golds: Dict[str, Dict[str, Any]], split: Dict[str, str]) -> Dict[str, Any]:
    """Pass iff dev/test voice sets are disjoint and each split is balanced over cells."""
    voices = {"dev": set(), "test": set()}
    cells = {"dev": Counter(), "test": Counter()}
    for cid, g in golds.items():
        s = split[cid]
        voices[s].update(g["metadata"]["voices"].values())
        cells[s][(g["num_speakers"], g["conversation_type"])] += 1
    overlap = sorted(voices["dev"] & voices["test"])
    balanced = {s: len({cells[s][c] for c in _CELLS}) == 1 for s in cells}
    return {"pass": not overlap and all(balanced.values()), "voice_overlap": overlap,
            "n_voices": {s: len(v) for s, v in voices.items()},
            "n_conversations": {s: sum(c.values()) for s, c in cells.items()},
            "balanced": balanced}


def check_corpus(golds: Dict[str, Dict[str, Any]], qc_report: Dict[str, Any],
                 split: Dict[str, str]) -> Dict[str, Any]:
    """Run all checks; ``report["pass"]`` is True iff every check passes."""
    report = {"n_conversations": len(golds),
              "oracle_round_trip": check_oracle_round_trip(golds),
              "acoustic_qc": check_qc(qc_report),
              "balance": check_balance(golds),
              "split": check_split(golds, split)}
    report["pass"] = all(report[k]["pass"] for k in
                         ("oracle_round_trip", "acoustic_qc", "balance", "split"))
    return report
