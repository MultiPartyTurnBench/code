"""Derive scoring anchors (EOT/INT positives, negative spans, excluded intervals)
from a MultiPartyTurnBench gold conversation.

The rules follow the TurnBench scoring protocol, generalised to N speakers:

EOT track (per speaker s):
  positives      : TURN events by s with floor_outcome == "yield" -> anchor = offset
                   (spec: EOT anchor = turn segment end).
  negative spans : mid-turn pauses where the SAME speaker resumes — a TURN by s
                   with floor_outcome == "hold", spanning [offset, resume_onset]
                   where resume_onset is the next TURN by s that keeps the floor.
  excluded       : TURN events by s with floor_outcome == "interrupted" — the
                   forced stop is not a clean EOT; ignore predictions near it.

INT track (per speaker s, scored on the interrupter's own channel):
  positives      : CLAIM_FLOOR by s resolved INTERRUPTION -> anchor = onset
                   (spec: INT anchor = consensus interrupter onset on that channel).
  negative spans : VOCALIZE_NO_CLAIM by s -> [onset, offset] (backchannel/laughter).
  excluded       : CLAIM_FLOOR by s resolved NON_FLOOR_TAKING_INTERRUPTION ->
                   [onset, offset] (indistinguishable from a real interruption at
                   onset / TurnBench excluded-interval treatment).

NONCONTENT (noise) events are never anchors; they are simply not emitted here
(they matter only as audio the model must not fire on, and any spurious fire
lands "outside any span" -> ignored by the scorer, matching TurnBench).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Half-width of the excluded neighbourhood around an interruption-forced turn stop
# on the EOT track. Equal to the scorer's matching tolerance (0.25 s), so detecting that
# (real but interruption-caused) boundary is neither rewarded nor penalised.
EOT_INTERRUPTED_EXCL_PAD = 0.25


@dataclass
class TrackAnchors:
    positives: List[float] = field(default_factory=list)          # anchor time points
    negative_spans: List[Tuple[float, float]] = field(default_factory=list)
    excluded_spans: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class SpeakerAnchors:
    eot: TrackAnchors = field(default_factory=TrackAnchors)
    interruption: TrackAnchors = field(default_factory=TrackAnchors)


def derive_anchors(gold: Dict[str, Any]) -> Dict[str, SpeakerAnchors]:
    """gold dict (validated) -> {speaker_key: SpeakerAnchors}."""
    speakers: List[str] = list(gold["speakers"])
    events: List[Dict[str, Any]] = list(gold["events"])
    out: Dict[str, SpeakerAnchors] = {s: SpeakerAnchors() for s in speakers}

    # Pre-index TURN events per speaker in temporal order, to resolve "hold"
    # negative spans (pause until the same speaker's next holding turn).
    turns_by_spk: Dict[str, List[Dict[str, Any]]] = {s: [] for s in speakers}
    for ev in events:
        if ev.get("label") == "TURN" and ev.get("speaker") in turns_by_spk:
            turns_by_spk[ev["speaker"]].append(ev)
    for s in speakers:
        turns_by_spk[s].sort(key=lambda e: e["onset"])

    for ev in events:
        label = ev.get("label")
        spk = ev.get("speaker")
        if spk not in out:
            continue  # NONCONTENT / untracked

        if label == "TURN":
            fo = ev.get("floor_outcome")
            off = ev.get("offset")
            if fo == "yield":
                out[spk].eot.positives.append(float(off))
            elif fo == "interrupted":
                # exclude a small neighborhood around the forced stop
                pad = EOT_INTERRUPTED_EXCL_PAD
                out[spk].eot.excluded_spans.append((float(off) - pad, float(off) + pad))
            elif fo == "hold":
                resume = _next_turn_onset_after(turns_by_spk[spk], off)
                if resume is not None and resume > off:
                    out[spk].eot.negative_spans.append((float(off), float(resume)))

        elif label == "CLAIM_FLOOR":
            ra = ev.get("resolved_as")
            if ra == "INTERRUPTION":
                out[spk].interruption.positives.append(float(ev["onset"]))
            elif ra == "NON_FLOOR_TAKING_INTERRUPTION":
                on = float(ev["onset"])
                off = float(ev.get("offset", on))
                out[spk].interruption.excluded_spans.append((on, max(off, on)))

        elif label == "VOCALIZE_NO_CLAIM":
            on = float(ev["onset"])
            off = float(ev.get("offset", on))
            out[spk].interruption.negative_spans.append((on, max(off, on)))

    # sort for determinism
    for s in speakers:
        for tr in (out[s].eot, out[s].interruption):
            tr.positives.sort()
            tr.negative_spans.sort()
            tr.excluded_spans.sort()
    return out


def _next_turn_onset_after(turns: List[Dict[str, Any]], t: float):
    """Onset of the earliest TURN in `turns` whose onset is >= t (the resume)."""
    best = None
    for e in turns:
        on = e["onset"]
        if on >= t - 1e-9 and (best is None or on < best):
            # must be strictly after this pause's start
            if on > t + 1e-9:
                best = on
    return best
