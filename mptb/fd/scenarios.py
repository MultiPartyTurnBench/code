"""Select the 120 full-duplex probe scenarios from the MPTB gold annotations (Appendix H.3, Table "protocol constants").

Every scenario is one real gold event; no dialogue is invented. The four event families are exactly
MPTB's scoring-anchor classes:

=============  ==========  =============================================  ==============================
family         protocol    gold event                                     correct agent behaviour
=============  ==========  =============================================  ==============================
true_eot       START (+)   TURN with ``floor_outcome == "yield"``         start speaking
hold_pause     START (-)   TURN with ``floor_outcome == "hold"``          stay silent until the resume
true_int       STOP  (+)   CLAIM_FLOOR resolved as INTERRUPTION           stop speaking within 1.5 s
backchannel    STOP  (-)   VOCALIZE_NO_CLAIM                              keep speaking
=============  ==========  =============================================  ==============================

Selection is deterministic (seed :data:`SELECTION_SEED`, sorted iteration): 30 scenarios per family,
10 per family for each N in {2, 3, 4}, spread round-robin over interaction types, with at most
:data:`MAX_PER_CONVERSATION` scenarios from any one conversation.
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..data import Dataset
from ..eval.anchors import derive_anchors

FAMILIES = ["true_eot", "hold_pause", "true_int", "backchannel"]
NS = [2, 3, 4]
SELECTION_SEED = 20260929
PER_FAMILY = 30
MAX_PER_CONVERSATION = 2
# Conversations left out of the candidate pool. Part of the published selection: changing this list
# changes which 120 scenarios are drawn.
EXCLUDED_CONVERSATIONS: Tuple[str, ...] = ("test_00176",)

# Preceding conversation streamed before a START event (bounds session length and API cost).
CONTEXT_LEAD_S = 20.0
# Minimum duration of the target turn, and minimum time into the conversation, for a START event.
MIN_LEAD_S = 6.0
# Silence streamed after a true EOT (the scoring window ends at +3.0 s).
OPEN_FLOOR_S = 4.0
# Minimum pause between a held turn and its resume.
MIN_NEG_SPAN_S = 0.35
# A STOP event must be found within this distance of a rendered clip onset.
CLIP_MATCH_TOL_S = 0.35
MIN_INT_CLIP_S = 0.4
MIN_BACKCHANNEL_S = 0.15


def iter_conversations(ds: Dataset, exclude: Iterable[str] = EXCLUDED_CONVERSATIONS
                       ) -> Iterator[Tuple[dict, dict]]:
    """Yield ``(gold, render)`` for every conversation with N in {2, 3, 4}, in id order."""
    exclude = set(exclude)
    for cid in ds.ids():
        gold = ds.gold(cid)
        if gold["num_speakers"] not in NS or cid in exclude:
            continue
        yield gold, ds.render(cid)


def _turns_by_speaker(gold: dict) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = defaultdict(list)
    for e in gold["events"]:
        if e.get("label") == "TURN":
            by[e["speaker"]].append(e)
    for s in by:
        by[s].sort(key=lambda e: e["onset"])
    return by


def clip_for_event(render: dict, speaker: str, onset: float) -> Optional[dict]:
    """The rendered clip of ``speaker`` whose start is closest to ``onset`` (None if none within tolerance)."""
    best, best_d = None, 1e9
    for c in render["clips"]:
        if c["speaker"] != speaker:
            continue
        d = abs(float(c["start_s"]) - float(onset))
        if d < best_d:
            best, best_d = c, d
    if best is None or best_d > CLIP_MATCH_TOL_S:
        return None
    return best


def candidates(gold: dict, render: dict) -> List[Dict[str, Any]]:
    """All usable scenario candidates of one conversation (any family)."""
    cid = gold["conversation_id"]
    anchors = derive_anchors(gold)
    turns = _turns_by_speaker(gold)
    out: List[Dict[str, Any]] = []

    for spk, sa in sorted(anchors.items()):
        for ev in gold["events"]:
            if ev.get("label") != "TURN" or ev.get("speaker") != spk:
                continue
            onset, offset = float(ev["onset"]), float(ev.get("offset", -1))
            if offset < 0 or offset - onset < MIN_LEAD_S or onset < MIN_LEAD_S:
                continue
            fo = ev.get("floor_outcome")
            if fo == "yield" and offset in sa.eot.positives:
                out.append(dict(
                    family="true_eot", event_id=ev["id"], target_speaker=spk,
                    event_time=offset, turn_onset=onset,
                    gold_floor_holder_before=spk, gold_floor_holder_after="AGENT",
                    negative_span=None, resume_time=None,
                    gold_agent_behavior="start_responding"))
            elif fo == "hold":
                resume = None
                for t in turns[spk]:
                    if float(t["onset"]) > offset + 1e-9:
                        resume = float(t["onset"])
                        break
                if resume is None or resume - offset < MIN_NEG_SPAN_S:
                    continue
                if (offset, resume) not in [tuple(x) for x in sa.eot.negative_spans]:
                    continue
                out.append(dict(
                    family="hold_pause", event_id=ev["id"], target_speaker=spk,
                    event_time=offset, turn_onset=onset,
                    gold_floor_holder_before=spk, gold_floor_holder_after=spk,
                    negative_span=[offset, resume], resume_time=resume,
                    gold_agent_behavior="remain_silent"))

        for ev in gold["events"]:
            if ev.get("speaker") != spk:
                continue
            lbl = ev.get("label")
            if lbl == "CLAIM_FLOOR" and ev.get("resolved_as") == "INTERRUPTION":
                onset = float(ev["onset"])
                if onset not in sa.interruption.positives:
                    continue
                clip = clip_for_event(render, spk, onset)
                if clip is None or float(clip["end_s"]) - float(clip["start_s"]) < MIN_INT_CLIP_S:
                    continue
                out.append(dict(
                    family="true_int", event_id=ev["id"], target_speaker=spk,
                    event_time=onset, turn_onset=onset,
                    gold_floor_holder_before="AGENT", gold_floor_holder_after=spk,
                    injected_clip=clip, negative_span=None, resume_time=None,
                    gold_agent_behavior="stop_promptly"))
            elif lbl == "VOCALIZE_NO_CLAIM":
                onset = float(ev["onset"])
                offset = float(ev.get("offset", onset))
                if offset - onset < MIN_BACKCHANNEL_S:
                    continue
                clip = clip_for_event(render, spk, onset)
                if clip is None:
                    continue
                out.append(dict(
                    family="backchannel", event_id=ev["id"], target_speaker=spk,
                    event_time=onset, turn_onset=onset,
                    gold_floor_holder_before="AGENT", gold_floor_holder_after="AGENT",
                    injected_clip=clip, negative_span=[onset, offset], resume_time=None,
                    gold_agent_behavior="continue_speaking"))
    by_id = {e["id"]: e for e in gold["events"]}
    for c in out:
        c["conversation_id"] = cid
        c["N"] = gold["num_speakers"]
        c["conversation_type"] = gold["conversation_type"]
        c["sublabel"] = by_id[c["event_id"]].get("sublabel")
    return out


def pick_balanced(pool: Sequence[Dict[str, Any]], quota: Dict[Tuple[str, int], int],
                  max_per_conv: int, seed: int) -> List[Dict[str, Any]]:
    """Greedy family x N quota fill, round-robin over interaction types, capped per conversation."""
    rng = random.Random(seed)
    chosen: List[Dict[str, Any]] = []
    used_conv: Counter = Counter()
    for fam in FAMILIES:
        for n in NS:
            want = quota[(fam, n)]
            cands = [c for c in pool if c["family"] == fam and c["N"] == n]
            rng.shuffle(cands)
            by_type: Dict[str, List[dict]] = defaultdict(list)
            for c in cands:
                by_type[c["conversation_type"]].append(c)
            types = sorted(by_type)
            got, i = 0, 0
            while got < want and any(by_type[t] for t in types):
                t = types[i % len(types)]
                i += 1
                while by_type[t]:
                    c = by_type[t].pop()
                    if used_conv[c["conversation_id"]] >= max_per_conv:
                        continue
                    used_conv[c["conversation_id"]] += 1
                    chosen.append(c)
                    got += 1
                    break
            if got < want:
                print(f"warning: {fam} N={n}: only {got}/{want} candidates available", file=sys.stderr)
    return chosen


def build_scenarios(ds: Dataset, per_family: int = PER_FAMILY,
                    max_per_conv: int = MAX_PER_CONVERSATION, seed: int = SELECTION_SEED,
                    exclude: Iterable[str] = EXCLUDED_CONVERSATIONS) -> List[Dict[str, Any]]:
    """Select and serialise the scenario list (the paper's 120 with the default arguments)."""
    pool: List[Dict[str, Any]] = []
    for gold, render in iter_conversations(ds, exclude):
        pool.extend(candidates(gold, render))

    base, rem = divmod(per_family, len(NS))
    quota = {(fam, n): base + (1 if i < rem else 0) for fam in FAMILIES for i, n in enumerate(NS)}
    chosen = pick_balanced(pool, quota, max_per_conv, seed)
    chosen.sort(key=lambda c: (FAMILIES.index(c["family"]), c["N"], c["conversation_id"], c["event_time"]))

    scenarios = []
    for i, c in enumerate(chosen, 1):
        start = c["family"] in ("true_eot", "hold_pause")
        scenarios.append({
            "scenario_id": f"fdb_{c['family']}_{i:03d}",
            "protocol": "START" if start else "STOP",
            "event_type": c["family"],
            "polarity": "positive" if c["family"] in ("true_eot", "true_int") else "negative",
            "gold_agent_behavior": c["gold_agent_behavior"],
            "conversation_id": c["conversation_id"],
            "N": c["N"],
            "conversation_type": c["conversation_type"],
            "target_speaker": c["target_speaker"],
            "source_event_id": c["event_id"],
            "sublabel": c.get("sublabel"),
            "event_time": round(float(c["event_time"]), 3),
            "turn_onset": round(float(c["turn_onset"]), 3),
            "context_start_s": round(max(0.0, c["turn_onset"] - CONTEXT_LEAD_S), 3),
            "negative_span": [round(x, 3) for x in c["negative_span"]] if c.get("negative_span") else None,
            "resume_time": round(c["resume_time"], 3) if c.get("resume_time") else None,
            "open_floor_s": OPEN_FLOOR_S if c["family"] == "true_eot" else None,
            "injected_clip": c.get("injected_clip"),
            "gold_floor_holder_before": c["gold_floor_holder_before"],
            "gold_floor_holder_after": c["gold_floor_holder_after"],
        })
    return scenarios
