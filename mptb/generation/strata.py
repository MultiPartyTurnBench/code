"""Interaction axes and the stratified conversation-spec sampler (paper Section 3.1, Appendix C).

A *conversation spec* (:class:`ConvSpec`) fixes everything about a conversation except the
words: party count N, conversation type, domain, target duration, the N voices, the acoustic
noise profile and the per-conversation seed. Specs are sampled so that every
(N x conversation type) cell receives the same number of conversations, while domain,
duration, voices and noise are drawn independently per conversation.

One spec per line in ``generation_manifest.jsonl`` is the input to Stage 1 (script
generation) and all later stages. Standard library only; deterministic given the seed.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..eval.schema import CONVERSATION_TYPES, N_MAX, N_MIN, speaker_key

DEFAULT_VOICE_POOL = Path(__file__).resolve().parents[2] / "configs" / "voice_pool.json"

# Scenario / domain vocabulary passed to the script generator (paper Table "Scenario conditioning").
DOMAINS = [
    "tech support call",
    "study group / exam prep",
    "family logistics planning",
    "product design standup",
    "podcast panel / current-events discussion",
    "board game night",
    "job interview",
    "travel itinerary planning",
    "classroom discussion",
    "relationship/interpersonal conflict",
    "restaurant/group dining decision",
    "sports commentary",
    "book club",
    "home renovation planning",
    "medical appointment scheduling (logistics only, non-clinical)",
    "roommate household planning",
    "startup pitch practice",
    "book/movie recommendation swap",
    "volunteer event coordination",
    "group trivia night",
]

# Stage-4 acoustic scene vocabulary (see mptb.generation.noise).
AMBIENTS = [
    "office_hum", "office_hvac", "dorm_room", "cafe", "studio_room_tone",
    "quiet_apartment", "quiet_office", "school_hallway", "living_room_tv_distant",
    "outdoor_park",
]
DISCRETE_EVENTS = [
    "keyboard_clatter", "notification_chime", "dishes_clinking", "laughter_offmic",
    "phone_ring", "door_knock", "chairs_scraping", "school_bell", "alarm_beep",
    "paper_shuffle",
]

# Target conversation duration (s), drawn uniformly per conversation.
TARGET_DURATION_RANGE_S = (150.0, 240.0)

# TurnBench per-type turn-taking statistics (Jiang et al., 2026, Table III). Used only as
# SOFT generation targets in the Stage-1 prompt (paper Table "Generator conditioning").
TABLE_III = {
    "argumentative": {"turn_len_s": 10.0, "bc_per_min": 3.09, "int_per_min": 2.48, "words_per_min": 206, "events_per_min": 17.0},
    "casual":        {"turn_len_s": 8.1,  "bc_per_min": 5.20, "int_per_min": 2.05, "words_per_min": 211, "events_per_min": 20.9},
    "collaborative": {"turn_len_s": 7.9,  "bc_per_min": 3.97, "int_per_min": 2.64, "words_per_min": 207, "events_per_min": 21.4},
    "instructional": {"turn_len_s": 10.5, "bc_per_min": 4.30, "int_per_min": 1.52, "words_per_min": 202, "events_per_min": 17.6},
    "narrative":     {"turn_len_s": 10.2, "bc_per_min": 4.76, "int_per_min": 1.50, "words_per_min": 198, "events_per_min": 18.8},
    "task_oriented": {"turn_len_s": 8.9,  "bc_per_min": 5.00, "int_per_min": 1.75, "words_per_min": 205, "events_per_min": 18.3},
}


@dataclass
class ConvSpec:
    """Specification of one conversation (one line of ``generation_manifest.jsonl``)."""
    conversation_id: str
    num_speakers: int
    speakers: List[str]
    domain: str
    conversation_type: str
    target_duration_s: float
    voices: Dict[str, Dict[str, str]]      # speaker_key -> {voice_id, gender, style}
    noise_profile: Dict[str, Any]          # {ambient, snr_db, discrete_events, background_speech}
    seed: int                              # Stage-1 tone/LLM seed and Stage-2 timing seed

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form (the manifest line)."""
        return asdict(self)


def compute_targets(spec: ConvSpec) -> Dict[str, Any]:
    """Soft Stage-1 targets for a spec, from TurnBench Table III scaled by duration and N.

    n_turns = max(2N, round(D / turn_len)); interruption and backchannel counts are the
    per-minute rates times D (min) times s_N = 1 + 0.1 (N - 2).
    """
    tbl = TABLE_III[spec.conversation_type]
    dur_s = spec.target_duration_s
    dur_min = dur_s / 60.0
    n_scale = 1.0 + 0.10 * (spec.num_speakers - 2)
    turn_len = tbl["turn_len_s"]
    n_turns = max(spec.num_speakers * 2, round(dur_s / turn_len))
    return {
        "turn_len_s": turn_len,
        "n_turns": n_turns,
        "n_interruptions": round(tbl["int_per_min"] * dur_min * n_scale),
        "n_backchannels": round(tbl["bc_per_min"] * dur_min * n_scale),
        "words_per_min": tbl["words_per_min"],
        "duration_min": round(dur_min, 2),
        "n_scale": round(n_scale, 2),
    }


def load_voice_pool(path: Optional[str] = None) -> List[Dict[str, str]]:
    """The 24-voice abstract pool ({voice_id, gender, style}) from ``configs/voice_pool.json``."""
    data = json.loads(Path(path or DEFAULT_VOICE_POOL).read_text())
    return data["voices"]


def _sample_noise(rng: random.Random) -> Dict[str, Any]:
    """Draw one noise profile: ambient scene, ambient SNR in [20, 38] dB, 0-2 discrete
    event labels (k from [0, 1, 1, 2]) and distant background speech with p = 0.2."""
    ambient = rng.choice(AMBIENTS)
    snr = round(rng.uniform(20.0, 38.0), 1)
    k = rng.choice([0, 1, 1, 2])
    discrete = rng.sample(DISCRETE_EVENTS, k) if k else []
    bg_speech = rng.random() < 0.2
    return {"ambient": ambient, "snr_db": snr, "discrete_events": discrete,
            "background_speech": bg_speech}


def sample_stratified(per_cell: int, seed: int, voice_pool: List[Dict[str, str]],
                      id_prefix: str = "conv_", start_index: int = 1) -> List[ConvSpec]:
    """Sample exactly ``per_cell`` specs for every (N, conversation_type) cell.

    Cells are shuffled so N/type are decoupled from the conversation index; domain,
    duration, the N distinct voices (from ``voice_pool``) and the noise profile are drawn
    independently per conversation. The per-conversation seed is ``seed * 100003 + index``.
    Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    if len(voice_pool) < N_MAX:
        raise ValueError(f"voice pool too small ({len(voice_pool)} < {N_MAX})")

    cells = [(n, ct) for n in range(N_MIN, N_MAX + 1) for ct in CONVERSATION_TYPES
             for _ in range(per_cell)]
    rng.shuffle(cells)

    dmin, dmax = TARGET_DURATION_RANGE_S
    specs: List[ConvSpec] = []
    for i, (n, ctype) in enumerate(cells):
        idx = start_index + i
        domain = rng.choice(DOMAINS)
        target = round(rng.uniform(dmin, dmax), 1)
        chosen_voices = rng.sample(voice_pool, n)
        voices = {speaker_key(k + 1): dict(chosen_voices[k]) for k in range(n)}
        noise = _sample_noise(rng)
        specs.append(ConvSpec(
            conversation_id=f"{id_prefix}{idx:05d}", num_speakers=n,
            speakers=[speaker_key(k + 1) for k in range(n)],
            domain=domain, conversation_type=ctype, target_duration_s=target,
            voices=voices, noise_profile=noise,
            seed=(seed * 100003 + idx),
        ))
    return specs


def save_manifest(specs: List[ConvSpec], path: str) -> None:
    """Write specs as JSON lines."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in specs:
            f.write(json.dumps(s.to_dict()) + "\n")


def load_manifest(path: str) -> List[ConvSpec]:
    """Read a JSON-lines manifest into :class:`ConvSpec` objects."""
    with open(path) as f:
        return [ConvSpec(**json.loads(line)) for line in f if line.strip()]
