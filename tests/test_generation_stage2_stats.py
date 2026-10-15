"""Tests for Stage 2 orchestration, corpus checks and corpus statistics (stdlib only, synthetic script)."""
from __future__ import annotations

import json

from mptb.eval.schema import validate_gold
from mptb.generation.checks import check_balance, check_oracle_round_trip, check_qc, check_split
from mptb.generation.stage2_orchestrate import build_gold, orchestrate, render_record
from mptb.generation.stats import dataset_stats, overlap_seconds

SCRIPT = {
    "conversation_id": "conv_00001", "num_speakers": 3, "speakers": ["speaker_1", "speaker_2", "speaker_3"],
    "conversation_type": "casual",
    "turns": [
        {"speaker": "speaker_1", "text": "a" * 60, "event_type": "TURN", "timing_hint": "after_pause"},
        {"speaker": "speaker_2", "text": "mm-hm", "event_type": "VOCALIZE_NO_CLAIM", "sublabel": "BACKCHANNEL",
         "target": "speaker_1", "timing_hint": "during_holder"},
        {"speaker": "speaker_1", "text": "b" * 60, "event_type": "TURN", "timing_hint": "latched"},
        {"speaker": "speaker_3", "text": "c" * 20, "event_type": "CLAIM_FLOOR",
         "resolved_hint": "NON_FLOOR_TAKING_INTERRUPTION", "target": "speaker_1", "timing_hint": "overlap_onset"},
        {"speaker": "speaker_2", "text": "d" * 40, "event_type": "TURN", "timing_hint": "after_pause"},
        {"speaker": "speaker_3", "text": "e" * 30, "event_type": "CLAIM_FLOOR", "resolved_hint": "INTERRUPTION",
         "target": "speaker_2", "timing_hint": "mid_utterance"},
        {"speaker": "speaker_3", "text": "f" * 50, "event_type": "TURN", "timing_hint": "latched"},
        {"speaker": "speaker_1", "text": "haha", "event_type": "VOCALIZE_NO_CLAIM", "sublabel": "LAUGHTER",
         "target": "speaker_3", "timing_hint": "during_holder"},
        {"speaker": "speaker_1", "text": "g" * 40, "event_type": "TURN", "timing_hint": "long_pause"},
    ],
}
META = {"domain": "book club", "voices": {"speaker_1": "v01", "speaker_2": "v03", "speaker_3": "v05"},
        "voice_styles": {}, "noise_profile": {}, "sample_rate": 24000, "tone": "casual and friendly",
        "generator_model": "stub", "engine": "kokoro"}


def dur(text: str, speaker: str) -> float:
    """Deterministic stand-in for TTS: 0.1 s per character."""
    return 0.1 * len(text)


def _gold(seed=3, cid="conv_00001", ctype="casual"):
    script = dict(SCRIPT, conversation_id=cid, conversation_type=ctype)
    return json.loads(json.dumps(build_gold(script, orchestrate(script, dur, seed=seed), META)))


def test_orchestration_is_deterministic_and_valid():
    r1, r2 = orchestrate(SCRIPT, dur, seed=3), orchestrate(SCRIPT, dur, seed=3)
    assert render_record("conv_00001", r1) == render_record("conv_00001", r2)
    gold = _gold()
    assert validate_gold(gold) == []
    labels = [(e["label"], e.get("floor_outcome") or e.get("resolved_as") or e.get("sublabel")) for e in gold["events"]]
    for expected in [("TURN", "yield"), ("TURN", "interrupted"), ("CLAIM_FLOOR", "INTERRUPTION"),
                     ("CLAIM_FLOOR", "NON_FLOOR_TAKING_INTERRUPTION"), ("VOCALIZE_NO_CLAIM", "BACKCHANNEL"),
                     ("VOCALIZE_NO_CLAIM", "LAUGHTER")]:
        assert expected in labels
    assert gold["events"] == sorted(gold["events"], key=lambda e: (e["onset"], e["speaker"]))


def test_interruption_truncates_holder_and_no_self_overlap():
    res = orchestrate(SCRIPT, dur, seed=3)
    cut = [c for c in res.clips if c.truncated]
    assert len(cut) == 1 and cut[0].speaker == "speaker_2" and cut[0].dur < dur("d" * 40, "")
    for spk in SCRIPT["speakers"]:
        own = sorted((c.start_s, c.end_s) for c in res.clips if c.speaker == spk)
        assert all(b[0] >= a[1] for a, b in zip(own, own[1:]))
    rr = render_record("conv_00001", res)
    assert rr["clips"][0] == {"id": "c000", "speaker": "speaker_1", "start_s": 0.3, "end_s": 6.3,
                              "kind": "holder", "overlaps": False, "truncated": False, "src_index": 0}


def test_oracle_round_trip_and_checks():
    golds = {f"conv_{i:05d}": _gold(seed=i, cid=f"conv_{i:05d}") for i in range(1, 4)}
    assert check_oracle_round_trip(golds)["pass"]
    qc = {"n_conversations": 3, "n_passed": 2, "n_failed": 1,
          "results": [{"conversation_id": c, "passed": c != "conv_00002"} for c in golds]}
    assert check_qc(qc)["failed"] == ["conv_00002"]
    assert not check_balance(golds)["pass"]
    sp = check_split(golds, {"conv_00001": "dev", "conv_00002": "test", "conv_00003": "test"})
    assert sp["voice_overlap"] == ["v01", "v03", "v05"] and not sp["pass"]


def test_overlap_seconds():
    ev = [{"label": "TURN", "speaker": "a", "onset": 0.0, "offset": 2.0},
          {"label": "VOCALIZE_NO_CLAIM", "speaker": "b", "onset": 1.5, "offset": 3.0},
          {"label": "CLAIM_FLOOR", "speaker": "c", "onset": 1.0},
          {"label": "TURN", "speaker": "a", "onset": 2.5, "offset": 4.0}]
    assert abs(overlap_seconds(ev) - 1.0) < 1e-9


def test_dataset_stats_structure_and_counts():
    golds = {"conv_00001": _gold(seed=1, cid="conv_00001"),
             "conv_00002": _gold(seed=2, cid="conv_00002", ctype="narrative")}
    st = dataset_stats(golds)
    assert list(st) == ["overall", "by_N", "by_type"] and list(st["by_type"]) == ["casual", "narrative"]
    assert st["overall"]["n_conv"] == 2 and st["by_N"]["3"]["n_conv"] == 2
    assert st["overall"]["int_anchors"] == 2
