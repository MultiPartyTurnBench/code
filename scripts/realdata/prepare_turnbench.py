#!/usr/bin/env python
"""Prepare the TurnBench dev slice (paper Table 14 "TurnBench dev slice", Appendix H.2): consensus anchors
and 16 kHz per-speaker audio for the first 12 dev conversations.

TurnBench dev (Hugging Face ``mundo-ai/turn-benchmark-dev``) is gated and licensed for non-commercial use.
Accept the gate on the Hub, then download the parquet files, e.g.::

    HF_TOKEN=... huggingface-cli download mundo-ai/turn-benchmark-dev --repo-type dataset --local-dir TB_DEV
    python scripts/realdata/prepare_turnbench.py --parquet-dir TB_DEV --out work/turnbench_slice

Writes ``<out>/anchors.json`` (consensus anchors per conversation) and ``<out>/audio16k/<cid>_speaker_{1,2}.flac``.
Both are derived from the gated data: keep them local and do not redistribute them. Prints the consensus
retention (Kept%) and anchor counts of the slice.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mptb.realdata.turnbench import (AUDIO_SR, SLICE_SIZE, anchors_for_conversation, anchors_to_json,  # noqa: E402
                                     iter_dev_audio, iter_dev_conversations)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True,
                    help="local snapshot of mundo-ai/turn-benchmark-dev (contains data/dev-*.parquet)")
    ap.add_argument("--out", required=True, help="output directory for anchors.json and audio16k/")
    ap.add_argument("--n-convs", type=int, default=SLICE_SIZE,
                    help=f"use the first N dev conversations in parquet order (default {SLICE_SIZE}, the paper's "
                         "slice; 0 = all 38)")
    ap.add_argument("--skip-audio", action="store_true", help="only write anchors.json")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    convs = iter_dev_conversations(a.parquet_dir)
    if a.n_convs:
        convs = islice(convs, a.n_convs)
    cache, kept, total, counts = {}, 0, 0, {}
    for conv in convs:
        anchors, stats = anchors_for_conversation(conv)
        cache[conv["conversation_id"]] = anchors_to_json(anchors)
        kept += stats["kept"]
        total += stats["total"]
        for sa in anchors.values():
            for tr in ("eot", "interruption"):
                ta = getattr(sa, tr)
                for k, v in (("pos", ta.positives), ("neg", ta.negative_spans), ("excl", ta.excluded_spans)):
                    counts[f"{tr}_{k}"] = counts.get(f"{tr}_{k}", 0) + len(v)
    (out / "anchors.json").write_text(json.dumps(cache))
    print(f"{len(cache)} conversations; consensus Kept {100 * kept / max(total, 1):.1f}%; anchors {counts}")

    if a.skip_audio:
        return
    import soundfile as sf
    adir = out / "audio16k"
    adir.mkdir(exist_ok=True)
    for cid, channels in iter_dev_audio(a.parquet_dir, cache):
        for s, x in channels.items():
            sf.write(str(adir / f"{cid}_{s}.flac"), x, AUDIO_SR)
        print(f"  audio {cid}")
    print(f"wrote {out}/anchors.json and {adir}/")


if __name__ == "__main__":
    main()
