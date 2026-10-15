"""Tests of the full-duplex analysis and offline-detector scoring on tiny synthetic trial logs."""
from __future__ import annotations

import csv
import json

import pytest

from mptb.fd import analysis as A
from mptb.fd.offline import hit, negative_span_rel, score_detector


def _trial(model, sid, run, fam, fired_, ok=True, n=2, cancel=0):
    proto = A.PROTOCOL[fam]
    if not ok:
        return {"model": model, "scenario_id": sid, "run": run, "protocol": proto, "event_type": fam,
                "ok": False, "invalid_reason": "agent_never_spoke", "correct": None, "latency_ms": None,
                "detail": {}, "N": n}
    pos = A.POSITIVE[fam]
    if fam == "true_eot":
        d = {"delta_start_s": 0.5 if fired_ else None}
    elif fam == "hold_pause":
        d = {"false_start": fired_}
    else:
        d = {"stopped_in_window": fired_, "provider_cancel_events": cancel, "resumed_within_2s": False}
    return {"model": model, "scenario_id": sid, "run": run, "protocol": proto, "event_type": fam, "ok": True,
            "invalid_reason": None, "correct": fired_ if pos else not fired_, "latency_ms": 500.0, "detail": d,
            "N": n}


def _rows():
    rows = []
    for fam, pattern in (("true_eot", [1, 1, 0]), ("hold_pause", [0, 0, 1]),
                         ("true_int", [1, 0, 0]), ("backchannel", [1, 1, 1])):
        for s in range(4):
            for r, f in enumerate(pattern):
                rows.append(_trial("m", f"{fam}_{s}", r, fam, bool(f), n=2 + s % 3, cancel=int(f)))
    rows.append(_trial("m", "true_int_9", 0, "true_int", None, ok=False))
    return rows


def test_model_metrics_headline():
    mm = A.model_metrics(_rows())
    h = mm["headline"]
    assert mm["n_trials_attempted"] == 49 and mm["n_trials_valid"] == 48
    assert h["S_correct_start"] == pytest.approx(2 / 3) and h["false_start_rate"] == pytest.approx(1 / 3)
    assert h["start_balanced"] == pytest.approx(2 / 3)
    assert h["stop_bias"] == pytest.approx(1 / 3 + 1 - 1)
    assert h["bargein_rate_backchannel"] == 1.0
    assert mm["families"]["true_eot"]["run_consistency"] == 0.0
    assert mm["invalid_reasons"] == {"agent_never_spoke": 1}


def test_bacc_bootstrap_point_and_determinism():
    ci1, ci2 = A.bacc_bootstrap(_rows(), b=500), A.bacc_bootstrap(_rows(), b=500)
    assert ci1 == ci2
    st = ci1["m"]["START"]
    assert st["bacc"] == pytest.approx(2 / 3) and st["n_pos"] == 12 and st["lo"] <= st["bacc"] <= st["hi"]


def test_spearman_and_contingency():
    assert A.spearman([1, 2, 3, None], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert A.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    rows = _rows()
    offline = {"m": {r["scenario_id"]: True for r in rows}}
    c = A.contingency(rows, offline)
    assert c["all"]["n_detector_correct"] == 48 and c["all"]["n_detector_wrong"] == 0


def test_trials_csv(tmp_path):
    A.write_trials_csv(tmp_path / "t.csv", _rows())
    with open(tmp_path / "t.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 49 and list(rows[0]) == A.TRIAL_COLUMNS
    assert {r["fired"] for r in rows} == {"True", "False", ""}


def test_analyze_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows()))
    res = A.analyze(runs, {}, tmp_path / "out")
    assert set(res["bacc_ci"]["m"]) == {"START", "STOP"}
    assert (tmp_path / "out" / "fd_association.json").exists()


def test_offline_hit_and_scoring():
    assert hit([10.1], 10.0, None) and not hit([13.5], 10.0, None)
    assert hit([12.0], 10.0, [10.0, 11.0]) and not hit([9.8], 10.0, [10.0, 11.0])
    scen = [{"scenario_id": "a", "protocol": "START", "event_type": "true_eot", "context_start_s": 0.0,
             "event_time": 10.0, "negative_span": None},
            {"scenario_id": "b", "protocol": "START", "event_type": "hold_pause", "context_start_s": 5.0,
             "event_time": 10.0, "negative_span": [10.0, 11.0]}]
    assert negative_span_rel(scen[1]) == [5.0, 6.0]
    raw = {"a": {"ok": True, "event_time_rel": 10.0, "commits": [10.5], "vad_stops": [], "vad_starts": []},
           "b": {"ok": True, "event_time_rel": 5.0, "commits": [5.5], "vad_stops": [], "vad_starts": []}}
    s = score_detector(scen, raw)
    assert s["eot"]["tp"] == 1 and s["eot"]["fp"] == 1 and s["E_eot"] == 0.5 and s["I_int"] is None
    assert s["per_event"] == {"a": True, "b": False}
