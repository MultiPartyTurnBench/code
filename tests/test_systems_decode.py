"""Commit rule, activity gates and operating-point rules of the baseline harness (no models needed)."""
import numpy as np

from mptb.systems.common import refractory, rest_active, rising_edges
from mptb.systems.harness import candidates, decode, select_grid_point, select_score_point
from mptb.systems.vad import masks_to_predictions


def test_refractory_and_rising_edges():
    assert refractory([0.0, 1.0, 2.0, 2.5, 4.6]) == [0.0, 2.0, 4.6]
    t = np.arange(6) * 0.5
    assert list(rising_edges(np.array([1, 1, 0, 1, 0, 1], bool), t)) == [0.0, 1.5, 2.5]


def test_rest_active_gate():
    masks = {"a": np.zeros(10, bool), "b": np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], bool)}
    g = rest_active(masks, "a", np.array([0.04, 0.08, 0.12, 0.2]), 0.04)
    assert g.tolist() == [False, True, True, False]


def test_masks_to_predictions():
    m1 = np.array([1] * 20 + [0] * 30, bool)
    m2 = np.array([0] * 10 + [1] * 40, bool)
    p = masks_to_predictions({"s1": m1, "s2": m2})
    assert p["s1"]["eot"] == [0.68]           # offset at frame 20 (0.40 s); committed after the 0.3 s hangover
    assert p["s2"]["interruption"] == [0.3]   # onset at frame 10 (0.20 s) while s1 is voiced; +0.1 s confirmation


def test_decode_gates_int_on_rest():
    per = {"s": {"t": np.array([0.1, 0.2, 0.3]), "eot": np.array([0.0, 0.9, 0.9]),
                 "int": np.array([0.9, 0.9, 0.0]), "rest": np.array([False, True, True])}}
    d = decode(per, 0.5, 0.5)
    assert d["s"] == {"eot": [0.2], "interruption": [0.2]}


def test_selection_rules():
    sweep = [{"theta": 0.5, "recall": 0.8, "fpr": 0.10, "tp": 8}, {"theta": 0.6, "recall": 0.8, "fpr": 0.05, "tp": 8},
             {"theta": 0.7, "recall": 0.5, "fpr": 0.20, "tp": 5}]
    assert select_grid_point(sweep, 0.10, "fpr")[0]["theta"] == 0.6
    assert select_grid_point(sweep, 0.10, "theta")[0]["theta"] == 0.5
    pts = [{"theta": 0.9, "recall": 0.0, "fpr": 0.0, "tp": 0}, {"theta": 0.5, "recall": 0.4, "fpr": 0.3, "tp": 4}]
    chosen, ok = select_score_point(pts, 0.10)
    assert chosen["theta"] == 0.5 and not ok


def test_candidates_include_grid_and_quantiles():
    dev = {"c": {"s": {"t": np.arange(5.0), "eot": np.array([0.001, 0.002, 0.003, 0.004, 0.005]),
                       "rest": np.ones(5, bool)}}}
    c = candidates(dev, "eot")
    assert 0.05 in c and 0.995 in c and min(c) < 0.01
