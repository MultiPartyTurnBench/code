"""Tests of the AMI floor-state derivation (mptb.realdata.ami) on a tiny hand-written annotation fixture."""
from __future__ import annotations

from mptb.eval.anchors import derive_anchors
from mptb.eval.schema import validate_gold
from mptb.realdata.ami import build_ami_gold, headset_files

NITE = 'xmlns:nite="http://nite.sourceforge.net/"'


def _segments(path, channel, spans):
    segs = "".join(f'<segment nite:id="s{i}" channel="{channel}" transcriber_start="{a}" transcriber_end="{b}"/>'
                   for i, (a, b) in enumerate(spans))
    path.write_text(f'<?xml version="1.0"?><nite:root {NITE}>{segs}</nite:root>')


def _words(path, words):
    punc = ' punc="true"'
    ws = "".join(f'<w nite:id="w{i}" starttime="{a}" endtime="{b}"{punc if p else ""}>x</w>'
                 for i, (a, b, p) in enumerate(words))
    path.write_text(f'<?xml version="1.0"?><nite:root {NITE}>{ws}</nite:root>')


def _fixture(tmp_path):
    """B (channel 0) holds with a pause, A backchannels, C interrupts B, A attempts, then A takes a yield."""
    (tmp_path / "segments").mkdir()
    (tmp_path / "words").mkdir()
    _segments(tmp_path / "segments/M.B.segments.xml", 0, [(0.0, 3.0), (3.5, 10.0)])
    _segments(tmp_path / "segments/M.A.segments.xml", 1, [(5.0, 5.5), (11.0, 11.6), (12.5, 15.0)])
    _segments(tmp_path / "segments/M.C.segments.xml", 2, [(8.5, 12.0)])
    _words(tmp_path / "words/M.A.words.xml", [(5.1, 5.3, False), (5.3, 5.3, True), (11.1, 11.3, False),
                                              (11.3, 11.5, False), (12.6, 13.0, False)])
    return str(tmp_path)


def test_floor_state_machine_events(tmp_path):
    ann = _fixture(tmp_path)
    files = headset_files(ann, "M")
    assert files == {"A": "M.Headset-1.wav", "B": "M.Headset-0.wav", "C": "M.Headset-2.wav"}
    gold = build_ami_gold(ann, "M", files, 16.0)
    assert validate_gold(gold) == []
    assert gold["speakers"] == ["speaker_1", "speaker_2", "speaker_3"]  # ordered by headset channel
    assert gold["metadata"]["agents"] == {"speaker_1": "B", "speaker_2": "A", "speaker_3": "C"}
    assert gold["audio"]["files"]["speaker_1"] == "M.Headset-0.wav"
    got = [(e["speaker"], e["label"], e["onset"], e["offset"], e.get("floor_outcome") or e.get("resolved_as")
            or e.get("sublabel")) for e in gold["events"]]
    assert got == [
        ("speaker_1", "TURN", 0.0, 3.0, "hold"),
        ("speaker_2", "VOCALIZE_NO_CLAIM", 5.0, 5.5, "BACKCHANNEL"),
        ("speaker_3", "CLAIM_FLOOR", 8.5, 12.0, "INTERRUPTION"),
        ("speaker_1", "TURN", 3.5, 10.0, "interrupted"),
        ("speaker_2", "CLAIM_FLOOR", 11.0, 11.6, "NON_FLOOR_TAKING_INTERRUPTION"),
        ("speaker_3", "TURN", 8.5, 12.0, "yield"),
        ("speaker_2", "TURN", 12.5, 15.0, "yield"),
    ]
    assert gold["events"][2]["floor_transfer_time"] == 10.0


def test_anchors_from_derived_gold(tmp_path):
    ann = _fixture(tmp_path)
    a = derive_anchors(build_ami_gold(ann, "M", headset_files(ann, "M"), 16.0))
    assert a["speaker_1"].eot.negative_spans == [(3.0, 3.5)]
    assert a["speaker_1"].eot.excluded_spans == [(9.75, 10.25)]
    assert a["speaker_3"].interruption.positives == [8.5] and a["speaker_3"].eot.positives == [12.0]
    assert a["speaker_2"].interruption.negative_spans == [(5.0, 5.5)]
    assert a["speaker_2"].interruption.excluded_spans == [(11.0, 11.6)]
    assert a["speaker_2"].eot.positives == [15.0]


def test_long_overlap_with_many_words_is_not_a_backchannel(tmp_path):
    (tmp_path / "segments").mkdir()
    (tmp_path / "words").mkdir()
    _segments(tmp_path / "segments/M.A.segments.xml", 0, [(0.0, 10.0)])
    _segments(tmp_path / "segments/M.B.segments.xml", 1, [(2.0, 3.0)])
    _words(tmp_path / "words/M.B.words.xml", [(2.0 + 0.2 * i, 2.1 + 0.2 * i, False) for i in range(4)])
    gold = build_ami_gold(str(tmp_path), "M", {}, 10.0)
    assert [e.get("resolved_as") for e in gold["events"] if e["label"] == "CLAIM_FLOOR"] == \
        ["NON_FLOOR_TAKING_INTERRUPTION"]
