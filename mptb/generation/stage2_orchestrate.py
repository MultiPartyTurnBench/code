"""Stage 2 -- floor-state orchestration (paper Section 3.1 step 2, Appendix C).

Turns a Stage-1 script (utterances tagged TURN / CLAIM_FLOOR / VOCALIZE_NO_CLAIM with
qualitative timing hints) into

1. a placement plan: per-utterance absolute ``[start_s, end_s]`` on the speaker's own channel,
   with overlaps and truncations resolved (the ``render`` annotation; input to assembly), and
2. the gold event log (the ``events`` annotation) from which EOT/INT anchors are derived by
   :func:`mptb.eval.anchors.derive_anchors`. The event log *is* the ground truth; no alignment
   to audio is needed.

The orchestrator is a small discrete-event state machine over an N-speaker floor. It is
audio-agnostic: utterance durations come from an injected ``duration_fn(text, speaker)``
(Stage 3 TTS in the pipeline, a stub in tests). All randomness flows through one
``random.Random(seed)``. Per utterance, in script order:

* TURN by a new speaker   -> the previous holder yields (EOT positive at its segment end),
  after a gap sampled from the timing hint;
* TURN by the same holder -> continuation; with probability ``midpause_prob`` (segments of at
  least ``midpause_min_seg_s``) a mid-turn pause splits the segment into a ``hold`` (EOT
  hard negative) and a resumption;
* CLAIM_FLOOR/INTERRUPTION -> the claimant starts inside the holder's utterance, the holder
  keeps speaking briefly and is then truncated; the floor transfers (INT positive at onset;
  the holder's TURN is ``interrupted`` and excluded from EOT);
* CLAIM_FLOOR/NON_FLOOR_TAKING_INTERRUPTION -> a brief failed overlap (excluded from INT);
* VOCALIZE_NO_CLAIM -> a backchannel or laughter overlapping the holder (INT hard negative).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..eval.schema import GoldConversation, GoldEvent

# duration_fn(text, speaker) -> duration_seconds
DurationFn = Callable[[str, str], float]


@dataclass
class TimingConfig:
    """Stage-2 timing parameters (paper Table "Stage-2 orchestrator parameters").

    Ranges are sampled uniformly. ``midpause_split_frac`` is part of the published
    configuration (``generation/generation_config.json``) but is not used by the orchestrator.
    """
    lead_pad_s: float = 0.3
    # clean floor-transfer gaps by timing_hint (TurnBench median inter-speaker gap
    # ~0.38s; transfers often begin slightly before the prior turn ends).
    yield_gap_after_pause: Tuple[float, float] = (0.20, 0.60)   # uniform
    yield_gap_latched: Tuple[float, float] = (-0.12, 0.06)      # slight overlap ok
    yield_gap_long_pause: Tuple[float, float] = (0.80, 2.00)
    # interruption: how far into the holder's ongoing utterance the claim begins,
    # and how long the holder keeps speaking (overlapping) before ceding (~1.48s median).
    intr_start_frac: Tuple[float, float] = (0.45, 0.85)
    intr_hold_after_s: Tuple[float, float] = (0.40, 1.20)
    # failed claim / backchannel overlap placement within the holder's utterance
    nonfloor_start_frac: Tuple[float, float] = (0.30, 0.80)
    vocalize_start_frac: Tuple[float, float] = (0.20, 0.80)
    # mid-turn pause injection (source of EOT negatives)
    midpause_prob: float = 0.45
    midpause_min_seg_s: float = 4.0
    midpause_dur: Tuple[float, float] = (0.40, 1.20)
    midpause_split_frac: Tuple[float, float] = (0.45, 0.70)
    # continuation gap when the same holder simply proceeds (not a mid-pause)
    cont_gap: Tuple[float, float] = (0.02, 0.18)


@dataclass
class Clip:
    """A placed audio unit on one speaker's channel."""
    id: str
    speaker: str
    text: str
    start_s: float
    end_s: float
    kind: str                       # holder | claim | backchannel | laughter
    overlaps: bool = False
    truncated: bool = False
    src_index: int = -1

    @property
    def dur(self) -> float:
        """Placed duration (s)."""
        return self.end_s - self.start_s


@dataclass
class OrchestrationResult:
    """Output of :func:`orchestrate`: placed clips, gold events and total duration (s)."""
    clips: List[Clip] = field(default_factory=list)
    events: List[GoldEvent] = field(default_factory=list)
    duration_s: float = 0.0


class _Orchestrator:
    """Discrete-event state machine over the floor (see module docstring)."""

    def __init__(self, script: Dict[str, Any], duration_fn: DurationFn,
                 rng: random.Random, cfg: TimingConfig):
        self.script = script
        self.dur_fn = duration_fn
        self.rng = rng
        self.cfg = cfg
        self.speakers = list(script["speakers"])
        self.clips: List[Clip] = []
        self.events: List[GoldEvent] = []
        self._eid = 0
        # floor state
        self.holder: Optional[str] = None
        self.seg_start: float = 0.0          # start of holder's current gold TURN segment
        self.holder_clip: Optional[Clip] = None  # holder's active (last) TURN clip
        self.t: float = cfg.lead_pad_s       # timeline cursor = end of holder speech so far

    # -- helpers -----------------------------------------------------------
    def _next_eid(self) -> str:
        self._eid += 1
        return f"e{self._eid:03d}"

    def _u(self, rng_range: Tuple[float, float]) -> float:
        return self.rng.uniform(*rng_range)

    def _emit_turn(self, speaker: str, onset: float, offset: float,
                   floor_outcome: str, next_holder: Optional[str] = None) -> None:
        self.events.append(GoldEvent(
            id=self._next_eid(), speaker=speaker, label="TURN",
            onset=onset, offset=offset, current_holder=speaker,
            floor_outcome=floor_outcome, next_holder=next_holder))

    # -- per-utterance handlers -------------------------------------------
    def run(self) -> OrchestrationResult:
        turns = self.script["turns"]
        for i, utt in enumerate(turns):
            et = utt["event_type"]
            if et == "TURN":
                self._handle_turn(i, utt)
            elif et == "CLAIM_FLOOR":
                self._handle_claim(i, utt)
            elif et == "VOCALIZE_NO_CLAIM":
                self._handle_vocalize(i, utt)
        # close the final holder segment as a yield (end of conversation EOT positive)
        if self.holder is not None and self.holder_clip is not None:
            self._emit_turn(self.holder, self.seg_start, self.holder_clip.end_s,
                            floor_outcome="yield", next_holder=None)
        duration = max((c.end_s for c in self.clips), default=0.0) + 0.3
        return OrchestrationResult(clips=self.clips, events=self.events, duration_s=duration)

    def _handle_turn(self, i: int, utt: Dict[str, Any]) -> None:
        S = utt["speaker"]
        dur = self.dur_fn(utt["text"], S)
        if self.holder is None:
            start = self.t
        elif S == self.holder:
            # continuation of the same holder — possibly a genuine mid-turn pause
            seg_len = self.holder_clip.end_s - self.seg_start if self.holder_clip else 0.0
            if (seg_len >= self.cfg.midpause_min_seg_s
                    and self.rng.random() < self.cfg.midpause_prob):
                # close current segment as a HOLD -> EOT negative span until resume
                pause = self._u(self.cfg.midpause_dur)
                self._emit_turn(self.holder, self.seg_start, self.holder_clip.end_s,
                                floor_outcome="hold")
                start = self.holder_clip.end_s + pause
                self.seg_start = start
            else:
                start = self.t + self._u(self.cfg.cont_gap)
        else:
            # clean floor transfer: previous holder yields -> EOT positive at seg end
            gap = self._yield_gap(utt.get("timing_hint", "after_pause"))
            self._emit_turn(self.holder, self.seg_start, self.holder_clip.end_s,
                            floor_outcome="yield", next_holder=S)
            start = max(0.0, self.holder_clip.end_s + gap)
            self.seg_start = start

        start = self._no_self_overlap(S, start)
        clip = Clip(id=f"c{i:03d}", speaker=S, text=utt["text"], start_s=start,
                    end_s=start + dur, kind="holder", src_index=i)
        self.clips.append(clip)
        self.holder = S
        self.holder_clip = clip
        self.t = clip.end_s

    def _handle_claim(self, i: int, utt: Dict[str, Any]) -> None:
        C = utt["speaker"]
        resolved = utt["resolved_hint"]
        dur = self.dur_fn(utt["text"], C)
        hc = self.holder_clip
        if hc is None:  # defensive: shouldn't happen (validated in Stage 1)
            return self._handle_turn(i, {**utt, "event_type": "TURN"})

        if resolved == "INTERRUPTION":
            # claim begins within the holder's ongoing utterance
            frac = self._u(self.cfg.intr_start_frac)
            c_start = hc.start_s + frac * hc.dur
            c_start = self._no_self_overlap(C, c_start)
            # holder keeps speaking briefly (overlap) then is cut short
            hold_after = self._u(self.cfg.intr_hold_after_s)
            transfer = min(c_start + hold_after, hc.end_s)
            if transfer <= hc.start_s + 0.05:
                transfer = min(hc.start_s + 0.05, hc.end_s)
            # truncate the holder's clip
            hc.truncated = True
            hc.end_s = transfer
            # close holder's gold TURN as interrupted (EOT excluded)
            self._emit_turn(self.holder, self.seg_start, transfer, floor_outcome="interrupted")
            # interrupter's claim clip (its speech = the claim text; it now holds)
            clip = Clip(id=f"c{i:03d}", speaker=C, text=utt["text"], start_s=c_start,
                        end_s=c_start + dur, kind="holder", overlaps=True, src_index=i)
            self.clips.append(clip)
            # gold INT positive at claim onset; floor transfers at `transfer`
            self.events.append(GoldEvent(
                id=self._next_eid(), speaker=C, label="CLAIM_FLOOR", onset=c_start,
                current_holder=self.holder, resolved_as="INTERRUPTION",
                next_holder=C, floor_transfer_time=transfer))
            # C becomes the new holder; its segment starts at c_start
            self.holder = C
            self.seg_start = c_start
            self.holder_clip = clip
            self.t = clip.end_s
        else:  # NON_FLOOR_TAKING_INTERRUPTION — brief failed overlap, holder keeps floor
            frac = self._u(self.cfg.nonfloor_start_frac)
            c_start = hc.start_s + frac * hc.dur
            c_start = self._no_self_overlap(C, c_start)
            clip = Clip(id=f"c{i:03d}", speaker=C, text=utt["text"], start_s=c_start,
                        end_s=c_start + dur, kind="claim", overlaps=True, src_index=i)
            self.clips.append(clip)
            self.events.append(GoldEvent(
                id=self._next_eid(), speaker=C, label="CLAIM_FLOOR", onset=c_start,
                offset=c_start + dur, current_holder=self.holder,
                resolved_as="NON_FLOOR_TAKING_INTERRUPTION"))
            # holder & cursor unchanged (holder continues)

    def _handle_vocalize(self, i: int, utt: Dict[str, Any]) -> None:
        V = utt["speaker"]
        dur = self.dur_fn(utt["text"], V)
        hc = self.holder_clip
        if hc is None:
            return
        frac = self._u(self.cfg.vocalize_start_frac)
        v_start = hc.start_s + frac * hc.dur
        v_start = self._no_self_overlap(V, v_start)
        sub = utt.get("sublabel", "BACKCHANNEL")
        clip = Clip(id=f"c{i:03d}", speaker=V, text=utt["text"], start_s=v_start,
                    end_s=v_start + dur, kind=sub.lower(), overlaps=True, src_index=i)
        self.clips.append(clip)
        self.events.append(GoldEvent(
            id=self._next_eid(), speaker=V, label="VOCALIZE_NO_CLAIM", sublabel=sub,
            onset=v_start, offset=v_start + dur, current_holder=self.holder))
        # holder & cursor unchanged

    def _yield_gap(self, hint: str) -> float:
        if hint == "latched":
            return self._u(self.cfg.yield_gap_latched)
        if hint == "long_pause":
            return self._u(self.cfg.yield_gap_long_pause)
        return self._u(self.cfg.yield_gap_after_pause)

    def _no_self_overlap(self, speaker: str, start: float, margin: float = 0.08) -> float:
        """A speaker cannot overlap themselves on their own channel."""
        latest = 0.0
        for c in self.clips:
            if c.speaker == speaker:
                latest = max(latest, c.end_s)
        return max(start, latest + margin) if latest else max(start, 0.0)


def orchestrate(script: Dict[str, Any], duration_fn: DurationFn, seed: int = 0,
                cfg: Optional[TimingConfig] = None) -> OrchestrationResult:
    """Resolve a Stage-1 script into placed clips and gold events (deterministic given ``seed``)."""
    rng = random.Random(seed)
    return _Orchestrator(script, duration_fn, rng, cfg or TimingConfig()).run()


def render_record(conversation_id: str, result: OrchestrationResult,
                  sample_rate: int = 24000) -> Dict[str, Any]:
    """The placement plan as stored in the ``render`` annotation (times rounded to 1 ms)."""
    return {
        "conversation_id": conversation_id,
        "duration_s": result.duration_s,
        "sample_rate": sample_rate,
        "clips": [{"id": c.id, "speaker": c.speaker, "start_s": round(c.start_s, 3),
                   "end_s": round(c.end_s, 3), "kind": c.kind, "overlaps": c.overlaps,
                   "truncated": c.truncated, "src_index": c.src_index} for c in result.clips],
    }


def build_gold(script: Dict[str, Any], result: OrchestrationResult,
               spec_meta: Dict[str, Any], audio_ext: str = "flac") -> Dict[str, Any]:
    """Assemble the gold event-log JSON (``events`` annotation) from orchestration output."""
    cid = script["conversation_id"]
    speakers = list(script["speakers"])
    files = {s: f"{cid}_{s}.{audio_ext}" for s in speakers}
    gc = GoldConversation(
        conversation_id=cid,
        conversation_type=script["conversation_type"],
        num_speakers=script["num_speakers"],
        speakers=speakers,
        duration_s=result.duration_s,
        audio_files=files,
        metadata=spec_meta,
        events=sorted(result.events, key=lambda e: (e.onset, e.speaker)),
    )
    return gc.to_dict()
