"""Tests of the TurnBench dev consensus and anchor rebuild (mptb.realdata.turnbench) on hand-built tracks."""
from __future__ import annotations

import pytest

from mptb.eval.scorer import score_track
from mptb.realdata.turnbench import (CANONICAL_MAP, ConsensusEvent, RealEvent, anchors_for_conversation,
                                     anchors_from_json, anchors_to_json, build_consensus, derive_real_anchors,
                                     to_canonical)


def _ev(start, end, canon, ann="a"):
    return RealEvent(start, end, canon, ann)


def test_canonical_map_is_17_to_7():
    assert len(CANONICAL_MAP) == 17
    assert set(CANONICAL_MAP.values()) == {"TURN", "INTERRUPTION", "NON_FLOOR_TAKING_INTERRUPTION",
                                           "BACKCHANNEL", "LAUGHTER", "AWKWARD_SILENCE", "NONCONTENT"}
    assert to_canonical("Floor-taking Competitive Interruption") == "INTERRUPTION"
    with pytest.raises(KeyError):
        to_canonical("Bogus Label")


def test_three_way_consensus_median_endpoints():
    tracks = {"a": [_ev(1.00, 2.00, "TURN", "a")], "b": [_ev(1.05, 2.10, "TURN", "b")],
              "c": [_ev(1.10, 1.95, "TURN", "c")]}
    cons, excl, kept, total = build_consensus(tracks)
    assert len(cons) == 1 and cons[0].n_agree == 3
    assert (cons[0].start, cons[0].end) == (1.05, 2.00)
    assert (kept, total, excl) == (3, 3, [])


def test_two_of_three_majority_and_no_majority_span():
    tracks = {"a": [_ev(1.0, 2.0, "TURN", "a")], "b": [_ev(1.05, 2.05, "TURN", "b")],
              "c": [_ev(5.0, 6.0, "BACKCHANNEL", "c")]}
    cons, excl, kept, total = build_consensus(tracks)
    assert len(cons) == 1 and cons[0].n_agree == 2 and cons[0].canon == "TURN"
    assert (kept, total) == (2, 3)
    assert excl == [(5.0, 6.0)]


def test_b_c_pair_without_a():
    tracks = {"a": [], "b": [_ev(1.0, 2.0, "BACKCHANNEL", "b")], "c": [_ev(1.1, 2.1, "BACKCHANNEL", "c")]}
    cons, excl, kept, _ = build_consensus(tracks)
    assert len(cons) == 1 and cons[0].canon == "BACKCHANNEL" and kept == 2 and excl == []


def test_endpoint_tolerance_is_200ms():
    tracks = {"a": [_ev(1.0, 2.0, "TURN", "a")], "b": [_ev(1.0, 2.5, "TURN", "b")], "c": []}
    cons, excl, kept, _ = build_consensus(tracks)
    assert cons == [] and kept == 0
    assert set(excl) == {(1.0, 2.0), (1.0, 2.5)}


def test_outvoted_minority_is_absorbed_not_excluded():
    tracks = {"a": [_ev(3.0, 3.6, "INTERRUPTION", "a")], "b": [_ev(3.05, 3.62, "INTERRUPTION", "b")],
              "c": [_ev(3.02, 3.7, "NON_FLOOR_TAKING_INTERRUPTION", "c")]}
    cons, excl, kept, _ = build_consensus(tracks)
    assert len(cons) == 1 and cons[0].canon == "INTERRUPTION"
    assert excl == [] and kept == 2


def test_eot_turn_view_positive_and_midturn_negative():
    s1 = [ConsensusEvent(0.0, 2.0, "TURN", 3), ConsensusEvent(2.5, 4.0, "TURN", 3)]
    s2 = [ConsensusEvent(5.0, 7.0, "TURN", 3)]
    anchors = derive_real_anchors({"speaker_1": s1, "speaker_2": s2}, {"speaker_1": [(9.0, 9.5)], "speaker_2": []})
    eot1 = anchors["speaker_1"].eot
    assert eot1.positives == [4.0]
    assert eot1.negative_spans == [(2.0, 2.5)]
    assert eot1.excluded_spans == [(9.0, 9.5)] == anchors["speaker_1"].interruption.excluded_spans
    assert anchors["speaker_2"].eot.positives == [7.0]


def test_int_label_view_anchor_types():
    s1 = [ConsensusEvent(1.0, 1.6, "INTERRUPTION", 2), ConsensusEvent(3.0, 3.4, "BACKCHANNEL", 3),
          ConsensusEvent(5.0, 5.5, "NON_FLOOR_TAKING_INTERRUPTION", 2), ConsensusEvent(7.0, 7.2, "NONCONTENT", 3),
          ConsensusEvent(9.0, 9.3, "LAUGHTER", 3)]
    it = derive_real_anchors({"speaker_1": s1}, {"speaker_1": []})["speaker_1"].interruption
    assert it.positives == [1.0]
    assert it.negative_spans == [(3.0, 3.4), (7.0, 7.2)]
    assert it.excluded_spans == [(5.0, 5.5)]


def test_conversation_pipeline_oracle_and_json_round_trip():
    a = {"a": [_ev(0.0, 2.0, "TURN"), _ev(4.0, 6.0, "INTERRUPTION"), _ev(8.0, 8.3, "BACKCHANNEL")]}
    tracks1 = {k: [RealEvent(e.start + d, e.end + d, e.canon, k) for e in a["a"]]
               for k, d in (("a", 0.0), ("b", 0.05), ("c", -0.05))}
    tracks2 = {k: [RealEvent(2.2 + d, 3.8 + d, "TURN", k)] for k, d in (("a", 0.0), ("b", 0.1), ("c", 0.0))}
    conv = {"conversation_id": "x", "conversation_type": "Casual", "tracks": {"speaker_1": tracks1, "speaker_2": tracks2}}
    anchors, stats = anchors_for_conversation(conv)
    assert stats == {"kept": 12, "total": 12, "canon_counts": {"TURN": 2, "INTERRUPTION": 1, "BACKCHANNEL": 1}}
    for sa in anchors.values():
        for ta in (sa.eot, sa.interruption):
            r = score_track(ta, list(ta.positives))
            if ta.positives:
                assert r.recall == 1.0 and r.fpr in (0.0, None)
    assert anchors_from_json(anchors_to_json(anchors)) == anchors
