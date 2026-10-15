#!/usr/bin/env python
"""Regenerate selected conversations with a fresh script (e.g. ones that fail acoustic QC).

The spec seed of each conversation is increased by ``--seed-bump`` (and written back to
``generation_manifest.jsonl``, so the manifest always holds the seed actually used), its old
artifacts are deleted, and Stages 1-4 are re-run: the LLM writes a new script with the same N,
type, domain, voices and noise profile. In MultiPartyTurnBench v1 this was applied once:
``--cids test_00176 --seed-bump 777`` (a near-silent speaker failed QC). Example:

    python scripts/generation/regenerate_conversations.py --out corpus/v1 --musan /data/musan \\
        --cids test_00176 --seed-bump 777
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.generation.noise import index_musan, noise_seed  # noqa: E402
from mptb.generation.pipeline import generate_conversation  # noqa: E402
from mptb.generation.strata import load_manifest, save_manifest  # noqa: E402


def main() -> int:
    """Bump seeds, delete artifacts and regenerate (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="working corpus directory")
    ap.add_argument("--musan", required=True, help="MUSAN root directory")
    ap.add_argument("--cids", nargs="+", required=True, help="conversation ids to regenerate")
    ap.add_argument("--seed-bump", type=int, default=777, help="added to each spec seed (default 777)")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="Stage-4 seed (default: keep the one in noise_seeds.json)")
    ap.add_argument("--model", default="gpt-4.1", help="Stage-1 LLM (default gpt-4.1)")
    ap.add_argument("--device", default="cuda", help="torch device for Kokoro (default cuda)")
    args = ap.parse_args()

    from mptb.generation.llm import ChatClient
    from mptb.generation.tts import KokoroTTS

    root = Path(args.out)
    manifest = root / "generation_manifest.jsonl"
    specs = load_manifest(str(manifest))
    by_id = {s.conversation_id: s for s in specs}
    missing = [c for c in args.cids if c not in by_id]
    if missing:
        sys.exit(f"not in the manifest: {missing}")
    for cid in args.cids:
        by_id[cid].seed += args.seed_bump
    save_manifest(specs, str(manifest))

    seeds_path = root / "noise_seeds.json"
    seeds = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
    llm = ChatClient(model=args.model)
    tts = KokoroTTS(device=args.device)
    musan = index_musan(args.musan)
    for cid in args.cids:
        spec = by_id[cid]
        for p in (root / "scripts" / f"{cid}.json", root / "gold" / f"{cid}.json",
                  root / "render" / f"{cid}.json"):
            p.unlink(missing_ok=True)
        for kind in ("clean", "noisy"):
            for f in (root / "audio" / kind).glob(f"{cid}_*"):
                f.unlink()
        if args.noise_seed is not None:
            seeds[cid] = args.noise_seed
        seeds.setdefault(cid, noise_seed(cid, spec.seed))
        seeds_path.write_text(json.dumps(seeds, indent=1, sort_keys=True))
        generate_conversation(spec, tts, llm, str(root), musan, seeds[cid], gen_model=args.model)
        print(f"  [{cid}] regenerated with seed {spec.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
