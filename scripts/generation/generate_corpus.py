#!/usr/bin/env python
"""Step 2 of corpus generation: Stages 1-4 for every spec in ``<out>/generation_manifest.jsonl``.

Stage 1 calls an OpenAI-compatible LLM (``OPENAI_API_KEY``, optional ``OPENAI_BASE_URL``) unless a
script already exists in ``<out>/scripts/``; Stages 2-4 are deterministic given the spec seed,
the Kokoro-82M revision and the Stage-4 noise seed. The run is checkpointed: completed
conversations are skipped, so it can be restarted after a crash. Requires a GPU-capable Kokoro
environment (requirements/generation.txt) and MUSAN (OpenSLR 17) for Stage 4.

To reproduce the released noise realizations, pass the dataset's seeds:
``--noise-seeds <dataset>/metadata/noise_seeds.json``. Example:

    python scripts/generation/generate_corpus.py --out corpus/v1 --musan /data/musan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.generation.noise import index_musan  # noqa: E402
from mptb.generation.pipeline import generate_corpus  # noqa: E402
from mptb.generation.strata import load_manifest  # noqa: E402


def merge_noise_seeds(out: Path, seeds_file: str) -> None:
    """Merge a seeds file (released ``{"seeds": {...}}`` or a flat ``{id: seed}``) into ``<out>/noise_seeds.json``."""
    new = json.loads(Path(seeds_file).read_text())
    new = new.get("seeds", new)
    path = out / "noise_seeds.json"
    seeds = json.loads(path.read_text()) if path.exists() else {}
    seeds.update({k: int(v) for k, v in new.items()})
    path.write_text(json.dumps(seeds, indent=1, sort_keys=True))


def main() -> int:
    """Run the generation pipeline (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="working corpus directory containing generation_manifest.jsonl")
    ap.add_argument("--musan", required=True, help="MUSAN root directory (with noise/ and speech/)")
    ap.add_argument("--cids", nargs="*", help="only these conversation ids (default: all in the manifest)")
    ap.add_argument("--noise-seeds", default=None,
                    help="JSON with Stage-4 seeds to use (e.g. the dataset's metadata/noise_seeds.json)")
    ap.add_argument("--model", default="gpt-4.1", help="Stage-1 LLM (default gpt-4.1)")
    ap.add_argument("--llm-cache", default=None, help="optional directory caching LLM responses")
    ap.add_argument("--tts-cache", default=None,
                    help="optional directory caching synthesized utterances (default $MPTB_TTS_CACHE)")
    ap.add_argument("--device", default="cuda", help="torch device for Kokoro (default cuda)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate audio and annotations of completed conversations (scripts are kept)")
    args = ap.parse_args()

    from mptb.generation.tts import KokoroTTS

    out = Path(args.out)
    specs = load_manifest(str(out / "generation_manifest.jsonl"))
    if args.cids:
        wanted = set(args.cids)
        specs = [s for s in specs if s.conversation_id in wanted]
    if args.noise_seeds:
        merge_noise_seeds(out, args.noise_seeds)
    need_llm = any(not (out / "scripts" / f"{s.conversation_id}.json").exists() for s in specs)
    llm = None
    if need_llm:
        from mptb.generation.llm import ChatClient
        llm = ChatClient(model=args.model, cache_dir=args.llm_cache)
    tts = KokoroTTS(device=args.device, cache_dir=args.tts_cache)
    done = generate_corpus(specs, tts, llm, str(out), index_musan(args.musan),
                           gen_model=args.model, overwrite=args.overwrite)
    print(f"generated {len(done)}/{len(specs)} conversations -> {out}")
    return 0 if len(done) == len(specs) else 1


if __name__ == "__main__":
    sys.exit(main())
