"""Stage 1 -- event-conditioned LLM script generation (paper Section 3.1 step 1, Appendix C).

Given a :class:`~mptb.generation.strata.ConvSpec`, an LLM (GPT-4.1 in the paper) writes a
multi-party dialogue as JSON, one record per utterance. Each utterance carries exactly one
symbolic act from a restricted vocabulary:

* ``TURN``              -- a floor-holding utterance (the default);
* ``CLAIM_FLOOR``       -- an attempt to take the floor from the current holder, with
  ``resolved_hint`` ``INTERRUPTION`` (succeeds) or ``NON_FLOOR_TAKING_INTERRUPTION`` (fails);
* ``VOCALIZE_NO_CLAIM`` -- a backchannel or laughter over the holder (never takes the floor).

Clean hand-offs (holder yields / continues) are not emitted by the LLM; Stage 2 derives them
from the sequence. ``timing_hint`` is a qualitative placement cue, never a timestamp: Stage 2
owns all timing. Scripts are normalised (:func:`normalize_script`), validated
(:func:`validate_script`) and regenerated with error feedback when malformed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .strata import ConvSpec, compute_targets

# Symbolic acts, resolution hints and sublabels the LLM may emit.
STAGE1_TAGS = ("TURN", "CLAIM_FLOOR", "VOCALIZE_NO_CLAIM")
RESOLVED_HINTS = ("INTERRUPTION", "NON_FLOOR_TAKING_INTERRUPTION")
SUBLABELS = ("BACKCHANNEL", "LAUGHTER")

# A backchannel is a brief vocalization ("mm-hm", "yeah", "right", "oh really"). A
# BACKCHANNEL longer than this many words is relabelled as a failed floor claim
# (NON_FLOOR_TAKING_INTERRUPTION) by normalize_script. LAUGHTER is exempt.
MAX_BACKCHANNEL_WORDS = 3
TIMING_HINTS = (
    "after_pause",     # normal gap, clean transition (TURN)
    "latched",         # starts immediately after prev speaker, ~no gap (TURN)
    "long_pause",      # notable silence before this turn (TURN)
    "mid_utterance",   # begins while the holder is mid-sentence (CLAIM_FLOOR)
    "overlap_onset",   # begins overlapping the tail of the holder (CLAIM_FLOOR)
    "during_holder",   # overlaps within the holder's speech (VOCALIZE / failed claim)
)

TONES = ("enthusiastic and energetic", "thoughtful and contemplative",
         "professional and business-like", "casual and friendly",
         "curious and inquisitive")

# One-line style description per conversation_type (drives the tag distribution).
TYPE_STYLE = {
    "casual": "a relaxed casual chat with frequent overlaps, backchannels, and light interruptions",
    "task_oriented": "a focused task-oriented working session; mostly orderly turns with occasional quick interjections",
    "instructional": "one participant explaining/teaching others who ask clarifying questions; mostly clean turns, few interruptions",
    "collaborative": "participants actively building on each other's ideas, with overlaps and frequent floor contests",
    "argumentative": "a heated debate with frequent disagreement, interruptions, and people talking over each other",
    "narrative": "storytelling with personal anecdotes; longer turns, listeners backchanneling, rare interruptions",
}


def _exemplar() -> str:
    """One short N=3 tagged exemplar exercising every tag type at least once."""
    return json.dumps({"turns": [
        {"speaker": "speaker_1", "text": "Okay so for the mitosis question, the key thing is the chromosome number stays the same.", "event_type": "TURN", "timing_hint": "after_pause"},
        {"speaker": "speaker_2", "text": "Well hang on—", "event_type": "CLAIM_FLOOR", "resolved_hint": "NON_FLOOR_TAKING_INTERRUPTION", "target": "speaker_1", "timing_hint": "mid_utterance"},
        {"speaker": "speaker_1", "text": "—right, whereas meiosis halves it, that's the whole point.", "event_type": "TURN", "timing_hint": "latched"},
        {"speaker": "speaker_3", "text": "mm-hm", "event_type": "VOCALIZE_NO_CLAIM", "sublabel": "BACKCHANNEL", "target": "speaker_1", "timing_hint": "during_holder"},
        {"speaker": "speaker_2", "text": "Actually wait, I think you've got prophase backwards, can we go back?", "event_type": "CLAIM_FLOOR", "resolved_hint": "INTERRUPTION", "target": "speaker_1", "timing_hint": "overlap_onset"},
        {"speaker": "speaker_2", "text": "In prophase one the homologs pair up, that's what makes it different.", "event_type": "TURN", "timing_hint": "latched"},
        {"speaker": "speaker_3", "text": "haha right", "event_type": "VOCALIZE_NO_CLAIM", "sublabel": "LAUGHTER", "target": "speaker_2", "timing_hint": "during_holder"},
        {"speaker": "speaker_1", "text": "Fair enough, good catch. So then the diagram on slide four is wrong too.", "event_type": "TURN", "timing_hint": "after_pause"},
    ]}, indent=2)


# Stage-1 system prompt shared by all conversations (released as generation/stage1_system_prompt.txt).
SYSTEM_PROMPT = """You are a meticulous dialogue data generator that writes REALISTIC multi-party spoken conversation transcripts for a turn-taking benchmark. You output STRICT JSON only.

The transcript is an ordered list of utterances. Each utterance is one JSON object with these fields:
- "speaker": one of the given speaker ids (e.g. "speaker_1").
- "text": what is actually said aloud. Natural spoken English (disfluencies, "um", "like", trailing "—" for cut-offs are welcome). No stage directions, no narration, no lists/bullets.
- "event_type": exactly one of:
    - "TURN": a normal floor-holding utterance (the speaker "has the floor"). This is the default.
    - "CLAIM_FLOOR": this speaker jumps in to try to take the floor FROM the current floor-holder while that holder is still speaking. REQUIRES:
        - "resolved_hint": "INTERRUPTION" (the jump-in SUCCEEDS — the holder stops and this speaker takes over; usually followed by a TURN from this same speaker) OR "NON_FLOOR_TAKING_INTERRUPTION" (the jump-in FAILS — it is a brief overlap and the ORIGINAL holder keeps going; usually the holder's next TURN continues their sentence).
        - "target": the speaker id being interrupted (the current holder).
    - "VOCALIZE_NO_CLAIM": a backchannel or laughter over the current holder; it NEVER takes the floor. REQUIRES:
        - "sublabel": "BACKCHANNEL" or "LAUGHTER".
        - "target": the current holder id being backchanneled.
        - A "BACKCHANNEL" MUST be an extremely brief vocalization of at most THREE words — ideally ONE — e.g. "mm-hm", "yeah", "right", "uh-huh", "exactly", "oh really", "no way". It is a listening noise, not a comment. If a listener would say a whole phrase or sentence, that is NOT a backchannel: make it a "CLAIM_FLOOR" instead (resolved_hint "NON_FLOOR_TAKING_INTERRUPTION" if the holder keeps talking, or "INTERRUPTION" if the holder stops). "LAUGHTER" should also be very short (e.g. "haha", "haha right").
- "timing_hint": exactly one of ["after_pause","latched","long_pause","mid_utterance","overlap_onset","during_holder"] — a QUALITATIVE placement cue, never a number:
    - TURN utterances use "after_pause" (normal gap), "latched" (right after, ~no gap), or "long_pause".
    - CLAIM_FLOOR uses "mid_utterance" or "overlap_onset".
    - VOCALIZE_NO_CLAIM uses "during_holder".

Hard rules:
1. The FIRST utterance must be a TURN (someone starts speaking).
2. A CLAIM_FLOOR or VOCALIZE_NO_CLAIM can only occur while someone holds the floor (never first, and its "target" must be the current holder — the most recent TURN speaker).
3. After a successful CLAIM_FLOOR (INTERRUPTION), the interrupter normally continues with one or more TURN utterances (they now hold the floor).
4. After a failed CLAIM_FLOOR (NON_FLOOR_TAKING_INTERRUPTION), the ORIGINAL holder normally continues with a TURN (often finishing their interrupted sentence, starting with "—").
5. EVERY listed speaker must speak at least one TURN.
6. Do NOT emit any tag other than the three above (no HOLDER_YIELDS etc.) — clean turn hand-offs are implicit (a TURN by a new speaker after a TURN means the previous speaker yielded the floor).
7. BACKCHANNELS must be <=3 words (ideally 1). A listener saying a full phrase/sentence is a CLAIM_FLOOR, never a BACKCHANNEL.
8. Output ONLY a JSON object: {"turns": [ ... ]}. No prose before or after.

Here is a short example of the exact output shape (a 3-speaker snippet exercising every tag):
""" + _exemplar()


def build_user_prompt(spec: ConvSpec, targets: Dict[str, Any], tone: str) -> str:
    """Instantiate the Stage-1 user prompt for one spec, its soft targets and sampled tone."""
    speaker_lines = "\n".join(
        f'  - {sk}: {v["gender"]}, {v["style"]}' for sk, v in spec.voices.items())
    style = TYPE_STYLE[spec.conversation_type]
    return f"""Write a {spec.conversation_type} multi-party conversation.

Scenario / domain: {spec.domain}
Conversation style: {style}
Overall tone: {tone}
Number of speakers: {spec.num_speakers}
Speakers (id: gender, speaking style — give each a distinct personality/voice):
{speaker_lines}

Approximate soft targets (aim NEAR these; they are guardrails, not exact quotas):
- Total utterances (TURN count): about {targets['n_turns']} floor-holding turns over ~{targets['duration_min']} minutes (mean turn ~{targets['turn_len_s']:.0f}s of speech, so ~{max(12, int(targets['turn_len_s']*2.5))}-{int(targets['turn_len_s']*4)} words each).
- Interruptions (CLAIM_FLOOR): about {targets['n_interruptions']} across the conversation (mix of INTERRUPTION and NON_FLOOR_TAKING_INTERRUPTION).
- Backchannels/laughter (VOCALIZE_NO_CLAIM): about {targets['n_backchannels']}.

Make it feel like a real, specific conversation about the scenario — concrete details, natural disfluency, distinct personalities. Every speaker participates. Output ONLY the JSON object {{"turns": [...]}}."""


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def validate_script(obj: Dict[str, Any], spec: ConvSpec) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). errors => retry/reject; warnings => log only."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(obj, dict) or "turns" not in obj:
        return (["output missing 'turns'"], [])
    turns = obj["turns"]
    if not isinstance(turns, list) or not turns:
        return (["'turns' must be a non-empty list"], [])

    spk_set = set(spec.speakers)
    min_turns = max(spec.num_speakers * 2, 8)
    n_turn = n_int = n_bc = 0
    speakers_seen_turn: set = set()
    holder: Optional[str] = None

    for i, t in enumerate(turns):
        p = f"turns[{i}]"
        if not isinstance(t, dict):
            errors.append(f"{p} not an object")
            continue
        spk = t.get("speaker")
        if spk not in spk_set:
            errors.append(f"{p}.speaker {spk!r} not in {sorted(spk_set)}")
        et = t.get("event_type")
        if et not in STAGE1_TAGS:
            errors.append(f"{p}.event_type {et!r} not in {STAGE1_TAGS}")
        text = t.get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{p}.text must be a non-empty string")
        th = t.get("timing_hint")
        if th not in TIMING_HINTS:
            errors.append(f"{p}.timing_hint {th!r} not in {TIMING_HINTS}")

        if i == 0 and et != "TURN":
            errors.append(f"{p}: first utterance must be TURN")

        if et == "TURN":
            n_turn += 1
            if spk in spk_set:
                speakers_seen_turn.add(spk)
            holder = spk
        elif et == "CLAIM_FLOOR":
            n_int += 1
            if t.get("resolved_hint") not in RESOLVED_HINTS:
                errors.append(f"{p}.resolved_hint required for CLAIM_FLOOR")
            tgt = t.get("target")
            if tgt not in spk_set:
                errors.append(f"{p}.target {tgt!r} not a valid speaker")
            elif tgt == spk:
                errors.append(f"{p}.target cannot be the speaker themselves")
            if holder is None:
                errors.append(f"{p}: CLAIM_FLOOR before anyone holds the floor")
            # a successful interruption transfers the floor
            if t.get("resolved_hint") == "INTERRUPTION":
                holder = spk
        elif et == "VOCALIZE_NO_CLAIM":
            n_bc += 1
            if t.get("sublabel") not in SUBLABELS:
                errors.append(f"{p}.sublabel required for VOCALIZE_NO_CLAIM")
            tgt = t.get("target")
            if tgt not in spk_set:
                errors.append(f"{p}.target {tgt!r} not a valid speaker")
            elif tgt == spk:
                errors.append(f"{p}.target cannot be the speaker themselves")
            if holder is None:
                errors.append(f"{p}: VOCALIZE_NO_CLAIM before anyone holds the floor")

    if n_turn < min_turns:
        errors.append(f"too few TURN utterances: {n_turn} < {min_turns}")
    missing = spk_set - speakers_seen_turn
    if missing:
        errors.append(f"these speakers never take a TURN: {sorted(missing)}")

    # distribution warnings (soft) — instructional legitimately has few interruptions
    if spec.conversation_type != "instructional" and n_int == 0:
        warnings.append("no CLAIM_FLOOR events (unexpectedly clean for this type)")
    if n_bc == 0:
        warnings.append("no VOCALIZE_NO_CLAIM events")
    return (errors, warnings)


# ----------------------------------------------------------------------------
# Normalization (post-generation, before validation & downstream stages)
# ----------------------------------------------------------------------------

def normalize_script(obj: Dict[str, Any]) -> int:
    """In-place: reclassify any BACKCHANNEL longer than MAX_BACKCHANNEL_WORDS as a
    FAILED barge-in (CLAIM_FLOOR / NON_FLOOR_TAKING_INTERRUPTION) — the holder keeps
    the floor, exactly as a backchannel implied, but it is no longer scored as an
    INT-negative (it becomes an excluded interval, matching TurnBench's treatment of
    non-floor-taking attempts). Idempotent. Returns the number of turns changed.

    Rationale: multi-word 'backchannels' acoustically compete with the holder and are
    failed barge-ins rather than listening noises. LAUGHTER is left untouched."""
    turns = obj.get("turns", []) if isinstance(obj, dict) else []
    changed = 0
    for t in turns:
        if (t.get("event_type") == "VOCALIZE_NO_CLAIM"
                and t.get("sublabel") == "BACKCHANNEL"
                and len(str(t.get("text", "")).split()) > MAX_BACKCHANNEL_WORDS):
            t["event_type"] = "CLAIM_FLOOR"
            t["resolved_hint"] = "NON_FLOOR_TAKING_INTERRUPTION"
            t.pop("sublabel", None)
            if t.get("timing_hint") == "during_holder":
                t["timing_hint"] = "overlap_onset"  # a failed barge-in overlaps the holder
            changed += 1
    if changed and isinstance(obj, dict):
        obj.setdefault("normalizations", []).append(
            {"rule": "long_backchannel_to_failed_bargein",
             "max_words": MAX_BACKCHANNEL_WORDS, "count": changed})
    return changed


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def generate_script(client, spec: ConvSpec, model: Optional[str] = None,
                    max_attempts: int = 3, temperature: float = 0.9,
                    verbose: bool = True) -> Dict[str, Any]:
    """Generate + validate a Stage-1 script for `spec`. Retries with error
    feedback on validation failure. Returns the script object augmented with
    conversation_id, spec metadata, and the sampled tone."""
    targets = compute_targets(spec)
    rng = random.Random(spec.seed)
    tone = rng.choice(TONES)
    user = build_user_prompt(spec, targets, tone)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    last_errors: List[str] = []
    for attempt in range(max_attempts):
        # vary seed across attempts so retries differ; cache keyed on payload
        obj = client.chat_json(messages, temperature=temperature,
                               max_tokens=8000, seed=spec.seed + attempt,
                               model=model, use_cache=(attempt == 0))
        if isinstance(obj, list):
            obj = {"turns": obj}
        normalize_script(obj)  # long backchannels -> failed barge-ins before validating
        errors, warnings = validate_script(obj, spec)
        if not errors:
            script = {
                "conversation_id": spec.conversation_id,
                "num_speakers": spec.num_speakers,
                "speakers": spec.speakers,
                "conversation_type": spec.conversation_type,
                "domain": spec.domain,
                "target_duration_s": spec.target_duration_s,
                "tone": tone,
                "targets": targets,
                "generator_model": model or getattr(client, "model", "?"),
                "turns": obj["turns"],
                "warnings": warnings,
            }
            if verbose and warnings:
                print(f"  [{spec.conversation_id}] warnings: {warnings}")
            return script
        last_errors = errors
        if verbose:
            print(f"  [{spec.conversation_id}] attempt {attempt+1} failed "
                  f"({len(errors)} errors): {errors[:3]}")
        # feed errors back for repair
        messages.append({"role": "assistant", "content": json.dumps(obj)[:6000]})
        messages.append({"role": "user", "content":
                         "That output had these problems; fix them and return the full corrected JSON:\n- "
                         + "\n- ".join(errors[:12])})
    raise RuntimeError(f"[{spec.conversation_id}] failed after {max_attempts} attempts: {last_errors}")


def save_script(script: Dict[str, Any], out_dir: str) -> str:
    """Write ``<out_dir>/<conversation_id>.json`` and return its path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(out_dir) / f"{script['conversation_id']}.json")
    with open(path, "w") as f:
        json.dump(script, f, indent=2)
    return path
