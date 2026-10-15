"""Tests of scenario selection on a synthetic gold event log."""
from __future__ import annotations

from mptb.fd import scenarios as S


def _gold():
    ev = [
        {"id": "e1", "label": "TURN", "speaker": "speaker_1", "onset": 7.0, "offset": 15.0, "floor_outcome": "yield"},
        {"id": "e2", "label": "TURN", "speaker": "speaker_2", "onset": 15.5, "offset": 24.0, "floor_outcome": "hold"},
        {"id": "e3", "label": "TURN", "speaker": "speaker_2", "onset": 25.0, "offset": 30.0, "floor_outcome": "yield"},
        {"id": "e4", "label": "VOCALIZE_NO_CLAIM", "speaker": "speaker_1", "onset": 20.0, "offset": 20.4,
         "sublabel": "BACKCHANNEL"},
        {"id": "e5", "label": "CLAIM_FLOOR", "speaker": "speaker_1", "onset": 28.0, "offset": 29.0,
         "resolved_as": "INTERRUPTION"},
    ]
    return {"conversation_id": "dev_00001", "num_speakers": 2, "conversation_type": "casual",
            "speakers": ["speaker_1", "speaker_2"], "events": ev}


def _render():
    return {"clips": [{"id": "c1", "speaker": "speaker_1", "start_s": 20.0, "end_s": 20.4},
                      {"id": "c2", "speaker": "speaker_1", "start_s": 28.1, "end_s": 29.0}]}


def test_candidates_cover_all_four_families():
    c = {x["family"]: x for x in S.candidates(_gold(), _render())}
    assert set(c) == set(S.FAMILIES)
    assert c["hold_pause"]["negative_span"] == [24.0, 25.0]
    assert c["true_int"]["injected_clip"]["id"] == "c2"
    assert c["backchannel"]["sublabel"] == "BACKCHANNEL"


def test_clip_match_tolerance():
    assert S.clip_for_event(_render(), "speaker_1", 28.3)["id"] == "c2"
    assert S.clip_for_event(_render(), "speaker_1", 25.0) is None


def test_pick_balanced_respects_quota_and_conversation_cap():
    pool = [{"family": "true_eot", "N": 2, "conversation_type": t, "conversation_id": f"c{i % 3}"}
            for i, t in enumerate(["casual", "narrative"] * 5)]
    quota = {(f, n): 0 for f in S.FAMILIES for n in S.NS}
    quota[("true_eot", 2)] = 5
    chosen = S.pick_balanced(pool, quota, max_per_conv=1, seed=1)
    assert len(chosen) == 3 and len({c["conversation_id"] for c in chosen}) == 3
    assert chosen == S.pick_balanced(pool, quota, max_per_conv=1, seed=1)
