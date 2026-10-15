"""TurnBench dev set: consensus gold, scoring anchors and 16 kHz audio (paper Table 14, Appendix H.2).

Source: the gated Hugging Face dataset ``mundo-ai/turn-benchmark-dev`` (38 English dyadic conversations,
non-commercial licence). One parquet row per conversation holds two per-speaker 48 kHz FLAC channels
(``speaker_{1,2}_audio``) and three independent annotator tracks per speaker::

    conversation_id : str
    metadata        : {conversation_type, ...}
    speaker_{1,2}_annotation_{a,b,c} : list[{start_s, end_s, label, text}]

``label`` is one of TurnBench's 17 fine labels, collapsed to 7 canonical categories. Gold is rebuilt as in
TurnBench (Section IV-A):

* consensus = 2-of-3 majority on the canonical label with BOTH endpoints within +/-200 ms; the gold
  endpoints are the medians over the agreeing annotators; genuine no-majority spans become excluded
  intervals;
* EOT (turn view): a TURN segment end is a positive if the floor passes to the other speaker (or it is the
  speaker's last turn); if the same speaker resumes first, the gap is a mid-turn-pause negative span;
* INT (label view): a consensus floor-taking INTERRUPTION onset is a positive on the interrupter's channel;
  BACKCHANNEL/NONCONTENT extents are negative spans; NON-FLOOR-TAKING INTERRUPTION extents are excluded.

The anchors are the :class:`mptb.eval.anchors.SpeakerAnchors` consumed by :func:`mptb.eval.scorer.score_track`.
"""
from __future__ import annotations

import glob
import io
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from ..eval.anchors import SpeakerAnchors, TrackAnchors

HF_DATASET = "mundo-ai/turn-benchmark-dev"
SLICE_SIZE = 12          # the paper's slice: the first 12 dev conversations in parquet order
AUDIO_SR = 16000         # sample rate of the extracted per-speaker channels

# 17 fine labels -> 7 canonical categories (TurnBench Table II)
CANONICAL_MAP: Dict[str, str] = {
    "Normal Turn": "TURN",
    "Strong Floor Hold": "TURN",
    "Bounded Response": "TURN",
    "Filler": "TURN",
    "Overlap": "TURN",
    "Floor-taking Competitive Interruption": "INTERRUPTION",
    "Floor-taking Cooperative Interruption": "INTERRUPTION",
    "Non-floor Taking Competitive Interruption": "NON_FLOOR_TAKING_INTERRUPTION",
    "Non-floor Taking Cooperative Interruption": "NON_FLOOR_TAKING_INTERRUPTION",
    "Acknowledgement Backchannel": "BACKCHANNEL",
    "Continuer Backchannel": "BACKCHANNEL",
    "Reaction Backchannel": "BACKCHANNEL",
    "Laughter": "LAUGHTER",
    "Awkward Silence": "AWKWARD_SILENCE",
    "Non-Speech Noise": "NONCONTENT",
    "Channel Bleed": "NONCONTENT",
    "Speech, Non-Linguistic": "NONCONTENT",
}

CONSENSUS_TOL = 0.200    # +/-200 ms endpoint agreement
SPEAKERS = ("speaker_1", "speaker_2")
ANNOTATORS = ("a", "b", "c")


def to_canonical(fine_label: str) -> str:
    """Map a TurnBench fine label to its canonical category (KeyError if unknown)."""
    try:
        return CANONICAL_MAP[fine_label]
    except KeyError:
        raise KeyError(f"unknown fine label {fine_label!r} (not in the 17-label TurnBench set)")


@dataclass
class RealEvent:
    """One annotator event of one speaker (canonical label)."""
    start: float
    end: float
    canon: str
    ann: str  # annotator id 'a' | 'b' | 'c'


@dataclass
class ConsensusEvent:
    """One consensus event; ``n_agree`` is the number of agreeing annotators (2 or 3)."""
    start: float
    end: float
    canon: str
    n_agree: int


# ---------------------------------------------------------------------------------------------- loading
def _dev_parquet_files(parquet_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "data", "dev-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no dev parquet under {parquet_dir}/data/ (download {HF_DATASET} first)")
    return files


def iter_dev_conversations(parquet_dir: str) -> Iterator[Dict[str, Any]]:
    """Yield ``{conversation_id, conversation_type, tracks}`` per conversation, in parquet order.

    ``tracks[speaker][annotator]`` is a list of :class:`RealEvent`. Audio columns are not read.
    """
    import pyarrow.parquet as pq

    ann_cols = [f"{s}_annotation_{a}" for s in SPEAKERS for a in ANNOTATORS]
    cols = ["conversation_id", "metadata"] + ann_cols
    for f in _dev_parquet_files(parquet_dir):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=cols).to_pylist():
                tracks: Dict[str, Dict[str, List[RealEvent]]] = {}
                for s in SPEAKERS:
                    tracks[s] = {}
                    for a in ANNOTATORS:
                        tracks[s][a] = [RealEvent(float(e["start_s"]), float(e["end_s"]), to_canonical(e["label"]), a)
                                        for e in r[f"{s}_annotation_{a}"]]
                yield {"conversation_id": str(r["conversation_id"]),
                       "conversation_type": r["metadata"]["conversation_type"],
                       "tracks": tracks}


def iter_dev_audio(parquet_dir: str, conversation_ids: Iterable[str]) -> Iterator[Tuple[str, Dict[str, np.ndarray]]]:
    """Yield ``(conversation_id, {speaker: mono float32 audio at 16 kHz})`` for the requested conversations.

    Each ``speaker_k_audio`` cell (FLAC bytes) is decoded, down-mixed if needed and resampled to 16 kHz
    with :func:`mptb.systems.common.resample`. Row groups are read one at a time to bound memory.
    """
    import pyarrow.parquet as pq
    import soundfile as sf

    from ..systems.common import resample

    wanted = set(conversation_ids)
    for f in _dev_parquet_files(parquet_dir):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            ids = [str(c) for c in pf.read_row_group(rg, columns=["conversation_id"])["conversation_id"].to_pylist()]
            if not wanted.intersection(ids):
                continue
            table = pf.read_row_group(rg, columns=["conversation_id"] + [f"{s}_audio" for s in SPEAKERS])
            for i, cid in enumerate(ids):
                if cid not in wanted:
                    continue
                out = {}
                for s in SPEAKERS:
                    cell = table[f"{s}_audio"][i].as_py()
                    data = cell["bytes"] if isinstance(cell, dict) else cell
                    a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                    if a.ndim > 1:
                        a = a.mean(axis=1)
                    out[s] = resample(a, sr, AUDIO_SR) if sr != AUDIO_SR else a
                yield cid, out


# ---------------------------------------------------------------------------------------------- consensus
def _endpoints_agree(x: RealEvent, y: RealEvent, tol: float) -> bool:
    return abs(x.start - y.start) <= tol and abs(x.end - y.end) <= tol


def build_consensus(tracks_abc: Dict[str, List[RealEvent]], tol: float = CONSENSUS_TOL
                    ) -> Tuple[List[ConsensusEvent], List[Tuple[float, float]], int, int]:
    """Combine one speaker's three annotator tracks into consensus gold.

    Returns ``(consensus_events, excluded_spans, kept_raw_events, total_raw_events)``. Pass 1 seeds on
    annotator a and takes the closest same-label, endpoint-agreeing match in b and in c; pass 2 pairs the
    remaining b and c events. Leftover events become excluded spans only if they do not overlap a consensus
    event (an outvoted minority label is absorbed, not excluded).
    """
    A, B, C = tracks_abc["a"], tracks_abc["b"], tracks_abc["c"]
    total = len(A) + len(B) + len(C)
    used = {"a": set(), "b": set(), "c": set()}
    consensus: List[ConsensusEvent] = []
    kept = 0

    def best_match(seed: RealEvent, cand: List[RealEvent], who: str) -> Optional[int]:
        best_i, best_cost = None, None
        for i, ev in enumerate(cand):
            if i in used[who] or ev.canon != seed.canon:
                continue
            if not _endpoints_agree(seed, ev, tol):
                continue
            cost = abs(seed.start - ev.start) + abs(seed.end - ev.end)
            if best_cost is None or cost < best_cost:
                best_i, best_cost = i, cost
        return best_i

    for i, a_ev in enumerate(A):
        members = [a_ev]
        j = best_match(a_ev, B, "b")
        k = best_match(a_ev, C, "c")
        if j is not None:
            members.append(B[j])
        if k is not None:
            members.append(C[k])
        if len(members) >= 2:
            used["a"].add(i)
            if j is not None:
                used["b"].add(j)
            if k is not None:
                used["c"].add(k)
            consensus.append(ConsensusEvent(statistics.median([m.start for m in members]),
                                            statistics.median([m.end for m in members]),
                                            a_ev.canon, len(members)))
            kept += len(members)

    for j, b_ev in enumerate(B):
        if j in used["b"]:
            continue
        k = best_match(b_ev, C, "c")
        if k is not None:
            used["b"].add(j)
            used["c"].add(k)
            members = [b_ev, C[k]]
            consensus.append(ConsensusEvent(statistics.median([m.start for m in members]),
                                            statistics.median([m.end for m in members]),
                                            b_ev.canon, 2))
            kept += 2

    consensus.sort(key=lambda e: e.start)

    def overlaps_consensus(st: float, en: float) -> bool:
        return any(not (en < c.start or st > c.end) for c in consensus)

    excluded: List[Tuple[float, float]] = []
    for who, lst in (("a", A), ("b", B), ("c", C)):
        for i, ev in enumerate(lst):
            if i not in used[who] and not overlaps_consensus(ev.start, ev.end):
                excluded.append((ev.start, ev.end))
    excluded.sort()
    return consensus, excluded, kept, total


# ---------------------------------------------------------------------------------------------- anchors
def derive_real_anchors(consensus_by_speaker: Dict[str, List[ConsensusEvent]],
                        excluded_by_speaker: Dict[str, List[Tuple[float, float]]]
                        ) -> Dict[str, SpeakerAnchors]:
    """Per-speaker EOT (turn view) and INT (label view) anchors from consensus gold.

    No-majority spans are excluded on both tracks. LAUGHTER and AWKWARD_SILENCE are neither positive nor
    negative.
    """
    speakers = list(consensus_by_speaker)
    out = {s: SpeakerAnchors() for s in speakers}

    global_turns: List[Tuple[float, float, str]] = []
    for s in speakers:
        for ev in consensus_by_speaker[s]:
            if ev.canon == "TURN":
                global_turns.append((ev.start, ev.end, s))
    global_turns.sort()

    for s in speakers:
        sa = out[s]
        for span in excluded_by_speaker.get(s, []):
            sa.eot.excluded_spans.append(span)
            sa.interruption.excluded_spans.append(span)

        s_turns = [(st, en) for (st, en, sp) in global_turns if sp == s]
        for (st, en) in s_turns:
            nxt = None
            for (nst, nen, nsp) in global_turns:
                if nst >= en - 1e-9 and nst > st + 1e-9:
                    nxt = (nst, nsp)
                    break
            if nxt is None or nxt[1] != s:
                sa.eot.positives.append(en)
            elif nxt[0] > en + 1e-9:
                sa.eot.negative_spans.append((en, nxt[0]))

        for ev in consensus_by_speaker[s]:
            if ev.canon == "INTERRUPTION":
                sa.interruption.positives.append(ev.start)
            elif ev.canon in ("BACKCHANNEL", "NONCONTENT"):
                sa.interruption.negative_spans.append((ev.start, ev.end))
            elif ev.canon == "NON_FLOOR_TAKING_INTERRUPTION":
                sa.interruption.excluded_spans.append((ev.start, ev.end))

        for tr in (sa.eot, sa.interruption):
            tr.positives.sort()
            tr.negative_spans.sort()
            tr.excluded_spans.sort()
    return out


def anchors_for_conversation(conv: Dict[str, Any], tol: float = CONSENSUS_TOL
                             ) -> Tuple[Dict[str, SpeakerAnchors], Dict[str, Any]]:
    """Consensus + anchors for one loaded conversation -> ``(anchors, {kept, total, canon_counts})``."""
    consensus_by_speaker: Dict[str, List[ConsensusEvent]] = {}
    excluded_by_speaker: Dict[str, List[Tuple[float, float]]] = {}
    kept = total = 0
    canon_counts: Dict[str, int] = {}
    for s in conv["tracks"]:
        cons, excl, k, t = build_consensus(conv["tracks"][s], tol)
        consensus_by_speaker[s] = cons
        excluded_by_speaker[s] = excl
        kept += k
        total += t
        for ev in cons:
            canon_counts[ev.canon] = canon_counts.get(ev.canon, 0) + 1
    anchors = derive_real_anchors(consensus_by_speaker, excluded_by_speaker)
    return anchors, {"kept": kept, "total": total, "canon_counts": canon_counts}


def anchors_to_json(anchors: Dict[str, SpeakerAnchors]) -> Dict[str, Any]:
    """Serialise ``{speaker: SpeakerAnchors}`` to plain JSON (the slice's ``anchors.json`` cache)."""
    def track(ta: TrackAnchors) -> Dict[str, Any]:
        return {"positives": list(ta.positives), "negative_spans": [list(x) for x in ta.negative_spans],
                "excluded_spans": [list(x) for x in ta.excluded_spans]}
    return {s: {"eot": track(sa.eot), "interruption": track(sa.interruption)} for s, sa in anchors.items()}


def anchors_from_json(d: Dict[str, Any]) -> Dict[str, SpeakerAnchors]:
    """Inverse of :func:`anchors_to_json`."""
    def track(t: Dict[str, Any]) -> TrackAnchors:
        return TrackAnchors(t["positives"], [tuple(x) for x in t["negative_spans"]],
                            [tuple(x) for x in t["excluded_spans"]])
    return {s: SpeakerAnchors(eot=track(v["eot"]), interruption=track(v["interruption"])) for s, v in d.items()}
