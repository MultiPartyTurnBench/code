"""AMI meetings: MultiPartyTurnBench gold derived from AMI's manual annotations (paper Table 14, Appendix H.2).

AMI meetings are four-party, headset-recorded (one channel per participant) design-team discussions. Floor
events are derived from AMI's manual segment and word annotations (``ami_public_manual_1.6.2``), not from a
turn-taking hand annotation. Floor-state machine (thresholds below):

* each participant's ``<segment>`` spans (``transcriber_start``/``transcriber_end``) are the speech units;
* a single floor ``holder`` is tracked; the holder's own segments extend the turn, and a same-speaker gap
  >= :data:`PAUSE_MIN` closes a TURN(hold) at the pause (EOT hard negative);
* a non-holder segment starting after the holder went silent is a floor yield: the holder's TURN(yield)
  closes (EOT positive) and the new speaker takes the floor;
* a non-holder segment overlapping the holder is
    - a BACKCHANNEL (VOCALIZE_NO_CLAIM; INT hard negative) if it is short (<= :data:`BC_MAX_DUR` s,
      <= :data:`BC_MAX_WORDS` words) and the holder keeps talking more than :data:`FLOOR_TAKE_WIN` s;
    - a floor-taking INTERRUPTION (INT positive) if it lasts >= :data:`FLOOR_MIN` s and the holder stops
      within :data:`FLOOR_TAKE_WIN` s of its onset; the holder's TURN closes as ``interrupted``;
    - otherwise a NON_FLOOR_TAKING_INTERRUPTION (excluded on the INT track).

The result is a standard gold dict (:mod:`mptb.eval.schema`), so anchors come from
:func:`mptb.eval.anchors.derive_anchors` exactly as for the synthetic benchmark.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

PAUSE_MIN = 0.20        # same-speaker gap >= this = mid-turn pause (EOT negative); shorter gaps merge
BC_MAX_DUR = 1.20       # backchannel max duration (s)
BC_MAX_WORDS = 3        # backchannel max words
FLOOR_MIN = 1.00        # min interrupter speech for a floor-taking interruption (s)
FLOOR_TAKE_WIN = 2.50   # the holder must fall silent within this of the interrupter onset
OVERLAP_EPS = 0.10      # min overlap for a non-holder start to count as "during" the holder

AMI_SPK = ["A", "B", "C", "D"]   # AMI participant (agent) ids
SLICE_MEETINGS = (               # the paper's 15 four-party meetings (~7.1 h of headset audio)
    "ES2002a", "ES2002b", "ES2002c", "ES2002d", "ES2003a", "ES2004a", "ES2005a",
    "IS1000a", "IS1000b", "IS1001a", "IS1001b", "TS3003a", "TS3003b", "TS3004a", "TS3004b",
)
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Headset-{h}.wav"


def _parse_segments(seg_path: str) -> Tuple[List[Tuple[float, float]], int]:
    """Sorted ``[(start, end)]`` of one participant's segments and its headset channel index."""
    segs: List[Tuple[float, float]] = []
    channel = None
    for s in ET.parse(seg_path).getroot().iter("segment"):
        t0 = float(s.get("transcriber_start"))
        t1 = float(s.get("transcriber_end"))
        if channel is None and s.get("channel") is not None:
            channel = int(s.get("channel"))
        if t1 > t0:
            segs.append((t0, t1))
    segs.sort()
    return segs, (channel if channel is not None else 0)


def _parse_word_times(words_path: str) -> List[Tuple[float, float]]:
    """Sorted word ``[(start, end)]`` (punctuation excluded); empty if the file is missing."""
    out: List[Tuple[float, float]] = []
    if not os.path.exists(words_path):
        return out
    tree = ET.parse(words_path)
    for w in tree.getroot().iter("{http://nite.sourceforge.net/}w"):
        if w.get("punc") == "true":
            continue
        st, en = w.get("starttime"), w.get("endtime")
        if st is None or en is None:
            continue
        try:
            out.append((float(st), float(en)))
        except ValueError:
            continue
    if not out:  # some words files use a bare <w> without namespace
        for w in tree.getroot().iter("w"):
            if w.get("punc") == "true":
                continue
            st, en = w.get("starttime"), w.get("endtime")
            if st and en:
                out.append((float(st), float(en)))
    out.sort()
    return out


def _nwords_in(word_times: List[Tuple[float, float]], t0: float, t1: float) -> int:
    return sum(1 for (a, b) in word_times if a >= t0 - 1e-3 and a <= t1 + 1e-3)


def headset_files(ann_dir: str, meeting: str) -> Dict[str, str]:
    """``{agent: headset WAV file name}``, from the first ``channel`` attribute of each segments file
    (agent index if absent)."""
    files = {}
    for a in AMI_SPK:
        sp = os.path.join(ann_dir, "segments", f"{meeting}.{a}.segments.xml")
        if not os.path.exists(sp):
            continue
        ch = None
        for s in ET.parse(sp).getroot().iter("segment"):
            if s.get("channel") is not None:
                ch = int(s.get("channel"))
                break
        if ch is None:
            ch = AMI_SPK.index(a)
        files[a] = f"{meeting}.Headset-{ch}.wav"
    return files


def build_ami_gold(ann_dir: str, meeting: str, audio_files: Dict[str, str], duration_s: float,
                   conversation_type: str = "task_oriented") -> Dict:
    """One gold dict for an AMI meeting from its segment/word annotations.

    Speakers are numbered by headset channel; ``audio_files`` maps agent id -> audio file name.
    """
    per_spk = {}
    present = []
    for a in AMI_SPK:
        sp = os.path.join(ann_dir, "segments", f"{meeting}.{a}.segments.xml")
        if not os.path.exists(sp):
            continue
        segs, ch = _parse_segments(sp)
        if not segs:
            continue
        wt = _parse_word_times(os.path.join(ann_dir, "words", f"{meeting}.{a}.words.xml"))
        per_spk[a] = dict(segs=segs, channel=ch, words=wt)
        present.append(a)
    present.sort(key=lambda a: per_spk[a]["channel"])
    spk_key = {a: f"speaker_{i+1}" for i, a in enumerate(present)}

    allsegs = []
    for a in present:
        for (t0, t1) in per_spk[a]["segs"]:
            allsegs.append((t0, t1, a))
    allsegs.sort()

    events: List[Dict] = []
    eid = [0]

    def emit(**kw):
        eid[0] += 1
        kw["id"] = f"e{eid[0]:04d}"
        events.append(kw)

    holder = None
    turn_start = None
    turn_last_end = None

    def close_turn(off, outcome, next_holder=None):
        emit(speaker=spk_key[holder], label="TURN", onset=float(turn_start), offset=float(off),
             current_holder=spk_key[holder], floor_outcome=outcome,
             **({"next_holder": spk_key[next_holder]} if next_holder else {}))

    for (t0, t1, a) in allsegs:
        if holder is None:
            holder, turn_start, turn_last_end = a, t0, t1
            continue
        if a == holder:
            if t0 - turn_last_end >= PAUSE_MIN:
                close_turn(turn_last_end, "hold")
                turn_start = t0
            turn_last_end = max(turn_last_end, t1)
            continue
        overlap = t0 < turn_last_end - OVERLAP_EPS
        if not overlap and t0 >= turn_last_end - 1e-6:
            close_turn(turn_last_end, "yield", next_holder=a)
            holder, turn_start, turn_last_end = a, t0, t1
            continue
        dur = t1 - t0
        nw = _nwords_in(per_spk[a]["words"], t0, t1)
        holder_stops_soon = (turn_last_end - t0) <= FLOOR_TAKE_WIN
        if dur <= BC_MAX_DUR and nw <= BC_MAX_WORDS and not holder_stops_soon:
            emit(speaker=spk_key[a], label="VOCALIZE_NO_CLAIM", onset=float(t0), offset=float(t1),
                 current_holder=spk_key[holder], sublabel="BACKCHANNEL")
        elif dur >= FLOOR_MIN and holder_stops_soon:
            emit(speaker=spk_key[a], label="CLAIM_FLOOR", onset=float(t0), offset=float(t1),
                 current_holder=spk_key[holder], resolved_as="INTERRUPTION",
                 next_holder=spk_key[a], floor_transfer_time=float(min(turn_last_end, t1)))
            close_turn(min(turn_last_end, t1), "interrupted", next_holder=a)
            holder, turn_start, turn_last_end = a, t0, t1
        else:
            emit(speaker=spk_key[a], label="CLAIM_FLOOR", onset=float(t0), offset=float(t1),
                 current_holder=spk_key[holder], resolved_as="NON_FLOOR_TAKING_INTERRUPTION")
    if holder is not None:
        close_turn(turn_last_end, "yield")

    return {
        "conversation_id": meeting,
        "conversation_type": conversation_type,
        "num_speakers": len(present),
        "speakers": [spk_key[a] for a in present],
        "duration_s": float(duration_s),
        "audio": {"mode": "channel_separated",
                  "files": {spk_key[a]: audio_files[a] for a in present if a in audio_files}},
        "metadata": {"source": "AMI", "derivation": "alignment-derived (segments/words)",
                     "agents": {spk_key[a]: a for a in present},
                     "channels": {spk_key[a]: per_spk[a]["channel"] for a in present}},
        "events": events,
    }
