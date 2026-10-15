"""Tiny synthetic gold conversations shared by the unit tests (no dataset files needed)."""


def tiny_gold(cid="test_00001", ctype="casual"):
    """Two speakers: S1 holds (pause 4.0-5.0), yields at 8.0; S2 backchannels, fails a claim, interrupts at 12.0."""
    return {
        "conversation_id": cid, "conversation_type": ctype, "num_speakers": 2,
        "speakers": ["speaker_1", "speaker_2"], "duration_s": 20.0,
        "audio": {"mode": "channel_separated", "files": {"speaker_1": "a.flac", "speaker_2": "b.flac"}},
        "metadata": {},
        "events": [
            {"id": "e1", "speaker": "speaker_1", "label": "TURN", "onset": 0.0, "offset": 4.0,
             "current_holder": "speaker_1", "floor_outcome": "hold"},
            {"id": "e2", "speaker": "speaker_1", "label": "TURN", "onset": 5.0, "offset": 8.0,
             "current_holder": "speaker_1", "floor_outcome": "yield", "next_holder": "speaker_2"},
            {"id": "e3", "speaker": "speaker_2", "label": "VOCALIZE_NO_CLAIM", "sublabel": "BACKCHANNEL",
             "onset": 2.0, "offset": 2.5, "current_holder": "speaker_1"},
            {"id": "e4", "speaker": "speaker_2", "label": "TURN", "onset": 8.5, "offset": 14.0,
             "current_holder": "speaker_2", "floor_outcome": "interrupted"},
            {"id": "e5", "speaker": "speaker_1", "label": "CLAIM_FLOOR", "resolved_as": "NON_FLOOR_TAKING_INTERRUPTION",
             "onset": 10.0, "offset": 10.6, "current_holder": "speaker_2"},
            {"id": "e6", "speaker": "speaker_1", "label": "CLAIM_FLOOR", "resolved_as": "INTERRUPTION",
             "onset": 12.0, "current_holder": "speaker_2", "next_holder": "speaker_1", "floor_transfer_time": 12.5},
        ],
    }
