"""Gold-annotation and submission schemas, validators and JSON I/O (standard library only).

A *gold* file is the role-relative event log of one conversation
(``<split>/annotations/events/<id>.json`` in the dataset release). A *submission* holds a
system's committed decision times per conversation, speaker channel and track::

    {"schema_version": 1,
     "predictions": [{"conversation_id": "test_00001",
                      "speaker_1": {"eot": [t, ...], "interruption": [t, ...]}, ...}]}

Speaker keys are ``speaker_1`` ... ``speaker_N`` (TurnBench's submission format,
generalised from two to N speakers). Anchor derivation lives in :mod:`mptb.eval.anchors`
and scoring in :mod:`mptb.eval.scorer`; this module only defines structure, validation and I/O.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ----------------------------------------------------------------------------
# Controlled vocabularies
# ----------------------------------------------------------------------------

# TurnBench's six conversation types.
CONVERSATION_TYPES = (
    "casual",
    "task_oriented",
    "instructional",
    "collaborative",
    "argumentative",
    "narrative",
)

# Gold event labels (role-relative).
#   TURN               : a floor-holding speaking span by the current holder.
#   CLAIM_FLOOR        : a non-holder attempting to take the floor.
#   VOCALIZE_NO_CLAIM  : backchannel / laughter — explicitly NOT claiming the floor.
#   NONCONTENT         : injected non-speech noise event (bell/alarm/etc.); scorer-negative.
EVENT_LABELS = ("TURN", "CLAIM_FLOOR", "VOCALIZE_NO_CLAIM", "NONCONTENT")

# How a CLAIM_FLOOR resolves.
RESOLVED_AS = ("INTERRUPTION", "NON_FLOOR_TAKING_INTERRUPTION")

# How a TURN's floor-holding span ends (drives EOT anchor derivation). This is a
# MultiPartyTurnBench extension to make gold anchors unambiguous without fragile
# sequence inference (Stage 2 emits it explicitly):
#   yield       : holder finished and the floor passed cleanly -> EOT positive at offset.
#   interrupted : holder was cut off by a CLAIM_FLOOR/INTERRUPTION -> excluded on EOT track
#                 (the transition is scored as an INT on the interrupter's channel).
#   hold        : holder paused mid-turn and later resumes -> EOT negative span.
FLOOR_OUTCOMES = ("yield", "interrupted", "hold")

# Sublabels for VOCALIZE_NO_CLAIM.
VOCALIZE_SUBLABELS = ("BACKCHANNEL", "LAUGHTER")

AUDIO_MODES = ("channel_separated",)

N_MIN, N_MAX = 2, 5

SUBMISSION_SCHEMA_VERSION = 1


def speaker_key(i: int) -> str:
    """1-indexed speaker key, e.g. speaker_key(1) == 'speaker_1'."""
    return f"speaker_{i}"


# ----------------------------------------------------------------------------
# Lightweight dataclasses for constructing gold (Stage 2 uses these).
# On-disk form is plain JSON dicts; use to_dict()/from_dict() to convert.
# ----------------------------------------------------------------------------

def _round(x: Optional[float], nd: int = 3) -> Optional[float]:
    return None if x is None else round(float(x), nd)


@dataclass
class GoldEvent:
    id: str
    speaker: str
    label: str                       # one of EVENT_LABELS
    onset: float
    current_holder: Optional[str]    # floor holder at event onset (None = genuine silence)
    offset: Optional[float] = None   # required for TURN/VOCALIZE_NO_CLAIM/NONCONTENT
    sublabel: Optional[str] = None   # VOCALIZE_NO_CLAIM only
    resolved_as: Optional[str] = None        # CLAIM_FLOOR only
    next_holder: Optional[str] = None        # set on transitions (yield / INTERRUPTION)
    floor_transfer_time: Optional[float] = None  # CLAIM_FLOOR resolved INTERRUPTION
    floor_outcome: Optional[str] = None      # TURN only (yield/interrupted/hold)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "speaker": self.speaker,
            "label": self.label,
            "onset": _round(self.onset),
            "current_holder": self.current_holder,
        }
        if self.offset is not None:
            d["offset"] = _round(self.offset)
        if self.sublabel is not None:
            d["sublabel"] = self.sublabel
        if self.resolved_as is not None:
            d["resolved_as"] = self.resolved_as
        if self.next_holder is not None:
            d["next_holder"] = self.next_holder
        if self.floor_transfer_time is not None:
            d["floor_transfer_time"] = _round(self.floor_transfer_time)
        if self.floor_outcome is not None:
            d["floor_outcome"] = self.floor_outcome
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldEvent":
        return cls(
            id=d["id"], speaker=d["speaker"], label=d["label"],
            onset=d["onset"], current_holder=d.get("current_holder"),
            offset=d.get("offset"), sublabel=d.get("sublabel"),
            resolved_as=d.get("resolved_as"), next_holder=d.get("next_holder"),
            floor_transfer_time=d.get("floor_transfer_time"),
            floor_outcome=d.get("floor_outcome"),
        )


@dataclass
class GoldConversation:
    conversation_id: str
    conversation_type: str
    num_speakers: int
    speakers: List[str]
    duration_s: float
    audio_files: Dict[str, str]      # speaker_k -> relative path
    metadata: Dict[str, Any]
    events: List[GoldEvent] = field(default_factory=list)
    audio_mode: str = "channel_separated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type,
            "num_speakers": self.num_speakers,
            "speakers": list(self.speakers),
            "duration_s": _round(self.duration_s),
            "audio": {"mode": self.audio_mode, "files": dict(self.audio_files)},
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldConversation":
        audio = d.get("audio", {})
        return cls(
            conversation_id=d["conversation_id"],
            conversation_type=d["conversation_type"],
            num_speakers=d["num_speakers"],
            speakers=list(d["speakers"]),
            duration_s=d["duration_s"],
            audio_files=dict(audio.get("files", {})),
            audio_mode=audio.get("mode", "channel_separated"),
            metadata=d.get("metadata", {}),
            events=[GoldEvent.from_dict(e) for e in d.get("events", [])],
        )


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised by assert_valid_* helpers when validation fails."""


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_gold(obj: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable errors (empty list == valid)."""
    e: List[str] = []
    if not isinstance(obj, dict):
        return ["gold must be a JSON object"]

    for k in ("conversation_id", "conversation_type", "num_speakers", "speakers",
              "duration_s", "audio", "events"):
        if k not in obj:
            e.append(f"missing required field: {k}")
    if e:
        return e

    if not isinstance(obj["conversation_id"], str) or not obj["conversation_id"]:
        e.append("conversation_id must be a non-empty string")

    ctype = obj["conversation_type"]
    if ctype not in CONVERSATION_TYPES:
        e.append(f"conversation_type {ctype!r} not in {CONVERSATION_TYPES}")

    n = obj["num_speakers"]
    if not isinstance(n, int) or isinstance(n, bool) or not (N_MIN <= n <= N_MAX):
        e.append(f"num_speakers must be int in [{N_MIN},{N_MAX}], got {n!r}")

    speakers = obj["speakers"]
    if not isinstance(speakers, list) or not all(isinstance(s, str) for s in speakers):
        e.append("speakers must be a list of strings")
        return e
    spk_set = set(speakers)
    if len(spk_set) != len(speakers):
        e.append("speakers contains duplicates")
    if isinstance(n, int) and len(speakers) != n:
        e.append(f"len(speakers)={len(speakers)} != num_speakers={n}")
    expected = [speaker_key(i) for i in range(1, len(speakers) + 1)]
    if speakers != expected:
        e.append(f"speakers must be exactly {expected} (got {speakers})")

    if not _is_num(obj["duration_s"]) or obj["duration_s"] <= 0:
        e.append("duration_s must be a positive number")
    dur = obj["duration_s"] if _is_num(obj["duration_s"]) else None

    audio = obj["audio"]
    if not isinstance(audio, dict):
        e.append("audio must be an object")
    else:
        if audio.get("mode") not in AUDIO_MODES:
            e.append(f"audio.mode must be one of {AUDIO_MODES}, got {audio.get('mode')!r}")
        files = audio.get("files")
        if not isinstance(files, dict):
            e.append("audio.files must be an object mapping speaker->path")
        else:
            for s in speakers:
                if s not in files:
                    e.append(f"audio.files missing entry for {s}")

    events = obj["events"]
    if not isinstance(events, list):
        e.append("events must be a list")
        return e

    seen_ids: set = set()
    for idx, ev in enumerate(events):
        e.extend(_validate_event(ev, idx, spk_set, dur, seen_ids))
    return e


def _validate_event(ev: Any, idx: int, spk_set: set, dur: Optional[float],
                    seen_ids: set) -> List[str]:
    p = f"events[{idx}]"
    e: List[str] = []
    if not isinstance(ev, dict):
        return [f"{p} must be an object"]

    eid = ev.get("id")
    if not isinstance(eid, str) or not eid:
        e.append(f"{p}.id must be a non-empty string")
    elif eid in seen_ids:
        e.append(f"{p}.id {eid!r} is duplicated")
    else:
        seen_ids.add(eid)

    label = ev.get("label")
    if label not in EVENT_LABELS:
        e.append(f"{p}.label {label!r} not in {EVENT_LABELS}")

    spk = ev.get("speaker")
    # NONCONTENT (noise) events need not belong to a tracked speaker.
    if label != "NONCONTENT":
        if spk not in spk_set:
            e.append(f"{p}.speaker {spk!r} not in speakers {sorted(spk_set)}")

    onset = ev.get("onset")
    if not _is_num(onset):
        e.append(f"{p}.onset must be a number")
    elif onset < 0:
        e.append(f"{p}.onset must be >= 0")
    elif dur is not None and onset > dur + 1e-6:
        e.append(f"{p}.onset {onset} exceeds duration_s {dur}")

    offset = ev.get("offset")
    needs_offset = label in ("TURN", "VOCALIZE_NO_CLAIM", "NONCONTENT")
    if needs_offset:
        if not _is_num(offset):
            e.append(f"{p}.offset required for label {label}")
        elif _is_num(onset) and offset < onset:
            e.append(f"{p}.offset {offset} < onset {onset}")

    ch = ev.get("current_holder")
    if ch is not None and ch not in spk_set:
        e.append(f"{p}.current_holder {ch!r} not in speakers")

    if label == "TURN":
        fo = ev.get("floor_outcome")
        if fo not in FLOOR_OUTCOMES:
            e.append(f"{p}.floor_outcome {fo!r} not in {FLOOR_OUTCOMES} (required for TURN)")
        nh = ev.get("next_holder")
        if fo == "yield" and nh is not None and nh not in spk_set:
            e.append(f"{p}.next_holder {nh!r} not in speakers")

    if label == "CLAIM_FLOOR":
        ra = ev.get("resolved_as")
        if ra not in RESOLVED_AS:
            e.append(f"{p}.resolved_as {ra!r} not in {RESOLVED_AS} (required for CLAIM_FLOOR)")
        if ra == "INTERRUPTION":
            if ev.get("next_holder") not in spk_set:
                e.append(f"{p}.next_holder required (in speakers) for INTERRUPTION")
            if not _is_num(ev.get("floor_transfer_time")):
                e.append(f"{p}.floor_transfer_time required for INTERRUPTION")

    if label == "VOCALIZE_NO_CLAIM":
        sl = ev.get("sublabel")
        if sl not in VOCALIZE_SUBLABELS:
            e.append(f"{p}.sublabel {sl!r} not in {VOCALIZE_SUBLABELS}")

    return e


def validate_submission(obj: Dict[str, Any]) -> List[str]:
    """Validate a model submission. Returns list of errors."""
    e: List[str] = []
    if not isinstance(obj, dict):
        return ["submission must be a JSON object"]
    if obj.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        e.append(f"schema_version must be {SUBMISSION_SCHEMA_VERSION}, got {obj.get('schema_version')!r}")
    preds = obj.get("predictions")
    if not isinstance(preds, list):
        return e + ["predictions must be a list"]
    seen_conv: set = set()
    for i, pr in enumerate(preds):
        p = f"predictions[{i}]"
        if not isinstance(pr, dict):
            e.append(f"{p} must be an object")
            continue
        cid = pr.get("conversation_id")
        if not isinstance(cid, str) or not cid:
            e.append(f"{p}.conversation_id must be a non-empty string")
        elif cid in seen_conv:
            e.append(f"{p}.conversation_id {cid!r} duplicated")
        else:
            seen_conv.add(cid)
        spk_keys = [k for k in pr.keys() if k.startswith("speaker_")]
        if not spk_keys:
            e.append(f"{p} has no speaker_* keys")
        for sk in spk_keys:
            track = pr[sk]
            if not isinstance(track, dict):
                e.append(f"{p}.{sk} must be an object with eot/interruption lists")
                continue
            for field_name in ("eot", "interruption"):
                if field_name not in track:
                    e.append(f"{p}.{sk} missing {field_name} list")
                    continue
                vals = track[field_name]
                if not isinstance(vals, list) or not all(_is_num(v) for v in vals):
                    e.append(f"{p}.{sk}.{field_name} must be a list of numbers")
    return e


def assert_valid_gold(obj: Dict[str, Any]) -> None:
    errs = validate_gold(obj)
    if errs:
        raise SchemaError("invalid gold:\n  - " + "\n  - ".join(errs))


def assert_valid_submission(obj: Dict[str, Any]) -> None:
    errs = validate_submission(obj)
    if errs:
        raise SchemaError("invalid submission:\n  - " + "\n  - ".join(errs))


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------

def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_gold(path: Union[str, Path]) -> Dict[str, Any]:
    obj = load_json(path)
    assert_valid_gold(obj)
    return obj


def load_submission(path: Union[str, Path]) -> Dict[str, Any]:
    obj = load_json(path)
    assert_valid_submission(obj)
    return obj
