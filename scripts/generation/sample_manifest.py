#!/usr/bin/env python
"""Step 1 of corpus generation: sample conversation specs with a voice-disjoint dev/test split.

Partitions the 24-voice pool (gender-balanced, 4 F + 4 M dev voices), then samples each split
stratified by (N x conversation type) from its own voices. Writes to ``--out``:

* ``generation_manifest.jsonl`` -- one conversation spec per line (mptb.generation.strata.ConvSpec);
* ``split.json``                -- {"split": {id: "dev"|"test"}, "dev_voice_ids", "test_voice_ids", "n_dev", "n_test"}.

The defaults reproduce the MultiPartyTurnBench v1 specs (48 dev + 192 test conversations).
Example:

    python scripts/generation/sample_manifest.py --out corpus/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.generation.split import partition_voices  # noqa: E402
from mptb.generation.strata import load_voice_pool, sample_stratified, save_manifest  # noqa: E402


def main() -> int:
    """Sample the manifest and split (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="working corpus directory to create/fill")
    ap.add_argument("--seed", type=int, default=23,
                    help="base seed: voice partition and dev sampling use SEED, test sampling SEED+1 (default 23)")
    ap.add_argument("--dev-per-cell", type=int, default=2,
                    help="dev conversations per (N x type) cell (default 2 -> 48)")
    ap.add_argument("--test-per-cell", type=int, default=8,
                    help="test conversations per (N x type) cell (default 8 -> 192)")
    ap.add_argument("--n-dev-per-gender", type=int, default=4,
                    help="dev-pool voices per gender (default 4 -> 8 dev / 16 test voices)")
    ap.add_argument("--voice-pool", default=None,
                    help="voice pool JSON (default: configs/voice_pool.json)")
    args = ap.parse_args()

    pool = load_voice_pool(args.voice_pool)
    dev_voices, test_voices = partition_voices(pool, n_dev_per_gender=args.n_dev_per_gender,
                                               seed=args.seed)
    dev = sample_stratified(args.dev_per_cell, args.seed, dev_voices, id_prefix="dev_")
    test = sample_stratified(args.test_per_cell, args.seed + 1, test_voices, id_prefix="test_")
    out = Path(args.out)
    save_manifest(dev + test, str(out / "generation_manifest.jsonl"))
    (out / "split.json").write_text(json.dumps({
        "split": {**{s.conversation_id: "dev" for s in dev}, **{s.conversation_id: "test" for s in test}},
        "dev_voice_ids": sorted(v["voice_id"] for v in dev_voices),
        "test_voice_ids": sorted(v["voice_id"] for v in test_voices),
        "n_dev": len(dev), "n_test": len(test),
    }, indent=2))
    print(f"dev={len(dev)} ({len(dev_voices)} voices) test={len(test)} ({len(test_voices)} voices) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
