"""Tests for spec sampling, the voice-disjoint split and Stage 1 script handling (stdlib only, stub LLM)."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from mptb.eval.schema import CONVERSATION_TYPES
from mptb.generation.split import partition_voices
from mptb.generation.stage1_script import (SYSTEM_PROMPT, generate_script, normalize_script,
                                           validate_script)
from mptb.generation.strata import (ConvSpec, compute_targets, load_manifest, load_voice_pool,
                                    sample_stratified, save_manifest)


def _spec(n=3, ctype="casual", seed=5, dur=180.0):
    pool = load_voice_pool()
    voices = {f"speaker_{k + 1}": dict(pool[k]) for k in range(n)}
    return ConvSpec("conv_00001", n, list(voices), "book club", ctype, dur, voices,
                    {"ambient": "cafe", "snr_db": 30.0, "discrete_events": [], "background_speech": False}, seed)


def test_voice_partition_is_disjoint_and_gender_balanced():
    pool = load_voice_pool()
    dev, test = partition_voices(pool, n_dev_per_gender=4, seed=23)
    assert len(dev) == 8 and len(test) == 16
    assert Counter(v["gender"] for v in dev) == {"F": 4, "M": 4}
    assert not {v["voice_id"] for v in dev} & {v["voice_id"] for v in test}
    assert partition_voices(pool, 4, 23) == (dev, test)


def test_stratified_sampling_balanced_and_deterministic(tmp_path):
    pool = load_voice_pool()
    specs = sample_stratified(2, 23, pool, id_prefix="dev_")
    assert len(specs) == 2 * 4 * len(CONVERSATION_TYPES)
    cells = Counter((s.num_speakers, s.conversation_type) for s in specs)
    assert set(cells.values()) == {2}
    for s in specs:
        ids = [v["voice_id"] for v in s.voices.values()]
        assert len(set(ids)) == s.num_speakers == len(s.speakers)
        assert 150.0 <= s.target_duration_s <= 240.0
        assert 20.0 <= s.noise_profile["snr_db"] <= 38.0
    assert specs[0].conversation_id == "dev_00001" and specs[0].seed == 23 * 100003 + 1
    assert [s.to_dict() for s in sample_stratified(2, 23, pool, id_prefix="dev_")] == [s.to_dict() for s in specs]
    save_manifest(specs, str(tmp_path / "m.jsonl"))
    assert [s.to_dict() for s in load_manifest(str(tmp_path / "m.jsonl"))] == [s.to_dict() for s in specs]


def test_compute_targets_formula():
    t = compute_targets(_spec(n=4, ctype="argumentative", dur=180.0))
    assert t["n_turns"] == 18                      # max(8, round(180 / 10.0))
    assert t["n_interruptions"] == round(2.48 * 3.0 * 1.2)
    assert t["n_backchannels"] == round(3.09 * 3.0 * 1.2)
    assert t["n_scale"] == 1.2 and t["duration_min"] == 3.0


def _valid_turns(n=3, n_turns=8):
    turns = [{"speaker": f"speaker_{i % n + 1}", "text": f"utterance number {i}", "event_type": "TURN",
              "timing_hint": "after_pause"} for i in range(n_turns)]
    turns.insert(2, {"speaker": "speaker_3", "text": "mm-hm", "event_type": "VOCALIZE_NO_CLAIM",
                     "sublabel": "BACKCHANNEL", "target": "speaker_2", "timing_hint": "during_holder"})
    turns.insert(4, {"speaker": "speaker_1", "text": "wait, I disagree with that", "event_type": "CLAIM_FLOOR",
                     "resolved_hint": "INTERRUPTION", "target": "speaker_3", "timing_hint": "overlap_onset"})
    return turns


def test_normalize_long_backchannel_to_failed_claim():
    obj = {"turns": [{"speaker": "speaker_2", "text": "oh yes that is so true", "event_type": "VOCALIZE_NO_CLAIM",
                      "sublabel": "BACKCHANNEL", "target": "speaker_1", "timing_hint": "during_holder"},
                     {"speaker": "speaker_2", "text": "haha that is really very funny", "event_type": "VOCALIZE_NO_CLAIM",
                      "sublabel": "LAUGHTER", "target": "speaker_1", "timing_hint": "during_holder"}]}
    assert normalize_script(obj) == 1
    t = obj["turns"][0]
    assert (t["event_type"], t["resolved_hint"], t["timing_hint"]) == (
        "CLAIM_FLOOR", "NON_FLOOR_TAKING_INTERRUPTION", "overlap_onset")
    assert "sublabel" not in t and obj["turns"][1]["event_type"] == "VOCALIZE_NO_CLAIM"
    assert normalize_script(obj) == 0


def test_validate_script_errors():
    spec = _spec()
    errors, _ = validate_script({"turns": _valid_turns()}, spec)
    assert errors == []
    bad = _valid_turns()
    bad[0] = dict(bad[0], event_type="CLAIM_FLOOR")
    errors, _ = validate_script({"turns": bad}, spec)
    assert any("first utterance must be TURN" in e for e in errors)
    errors, _ = validate_script({"turns": _valid_turns()[:3]}, spec)
    assert any("too few TURN" in e for e in errors)


class StubLLM:
    """Returns queued responses from chat_json and records the request arguments."""

    model = "stub-model"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def chat_json(self, messages, **kw):
        self.calls.append((json.loads(json.dumps(messages)), kw))
        return self.responses.pop(0)


def test_generate_script_retries_with_feedback():
    spec = _spec()
    llm = StubLLM([{"turns": _valid_turns()[:3]}, {"turns": _valid_turns()}])
    script = generate_script(llm, spec, verbose=False)
    assert len(llm.calls) == 2
    (m0, kw0), (m1, kw1) = llm.calls
    assert m0[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert kw0 == {"temperature": 0.9, "max_tokens": 8000, "seed": spec.seed, "model": None, "use_cache": True}
    assert kw1["seed"] == spec.seed + 1 and kw1["use_cache"] is False
    assert m1[-1]["role"] == "user" and "fix them" in m1[-1]["content"]
    assert script["generator_model"] == "stub-model" and script["turns"] == _valid_turns()
    assert script["tone"] == generate_script(StubLLM([{"turns": _valid_turns()}]), spec, verbose=False)["tone"]


def test_generate_script_gives_up():
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        generate_script(StubLLM([{"turns": []}] * 3), _spec(), verbose=False)
