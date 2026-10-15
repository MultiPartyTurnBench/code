"""Scorer, anchors and schema on a tiny synthetic conversation."""
import pytest

from mptb.eval import derive_anchors, score_dataset, select_operating_point, validate_gold, validate_submission
from mptb.eval.scorer import TrackResult, percentile, score_track
from mptb.eval.anchors import TrackAnchors
from fixtures import tiny_gold


def sub(preds, cid="test_00001"):
    return {"schema_version": 1, "predictions": [{"conversation_id": cid, **preds}]}


def test_gold_valid_and_anchors():
    g = tiny_gold()
    assert validate_gold(g) == []
    a = derive_anchors(g)
    assert a["speaker_1"].eot.positives == [8.0]
    assert a["speaker_1"].eot.negative_spans == [(4.0, 5.0)]
    assert a["speaker_2"].eot.excluded_spans == [(13.75, 14.25)]
    assert a["speaker_1"].interruption.positives == [12.0]
    assert a["speaker_1"].interruption.excluded_spans == [(10.0, 10.6)]
    assert a["speaker_2"].interruption.negative_spans == [(2.0, 2.5)]


def test_oracle_scores_perfectly():
    s = sub({"speaker_1": {"eot": [8.0], "interruption": [12.0]}, "speaker_2": {"eot": [], "interruption": []}})
    assert validate_submission(s) == []
    m = score_dataset({"test_00001": tiny_gold()}, s)["overall"]
    assert (m["eot"]["recall"], m["eot"]["fpr"]) == (1.0, 0.0)
    assert (m["interruption"]["recall"], m["interruption"]["fpr"]) == (1.0, 0.0)


def test_false_positive_window_and_exclusion():
    s = sub({"speaker_1": {"eot": [4.5, 8.2], "interruption": [10.3]},
             "speaker_2": {"eot": [], "interruption": [2.2]}})
    m = score_dataset({"test_00001": tiny_gold()}, s)["overall"]
    assert m["eot"]["tp"] == 1 and m["eot"]["fp"] == 1 and m["eot"]["latency_p50"] == pytest.approx(0.2)
    assert m["interruption"]["fn"] == 1 and m["interruption"]["fp"] == 1  # 10.3 excluded, 2.2 hits the backchannel


def test_matching_is_earliest_unclaimed_within_window():
    a = TrackAnchors(positives=[1.0, 2.0])
    r = score_track(a, [0.8, 1.9, 1.95])
    assert (r.tp, r.fn) == (2, 0) and r.latencies == pytest.approx([-0.2, -0.1])
    r = score_track(a, [0.8, 1.1, 5.1])  # 1.1 is before 2.0 - 0.25 and 5.1 after 2.0 + 3.0
    assert (r.tp, r.fn) == (1, 1)
    assert score_track(a, [5.5]).tp == 0 and score_track(a, [0.7]).tp == 0


def test_percentile_and_operating_point():
    assert percentile([0.0, 1.0], 50) == 0.5 and percentile([], 50) is None
    g = {"test_00001": tiny_gold()}
    subs = {0.3: sub({"speaker_1": {"eot": [4.5, 8.0], "interruption": []}}),
            0.7: sub({"speaker_1": {"eot": [8.0], "interruption": []}})}
    op = select_operating_point(g, subs, budget=0.10)
    assert op["eot"]["selected"]["theta"] == 0.7
    assert TrackResult(tp=1, fn=1).recall == 0.5
