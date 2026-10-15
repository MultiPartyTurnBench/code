#!/usr/bin/env python
"""Step 3 of the full-duplex study: run every scenario x agent x repeat (1,800 trials for 5 agents x 3 runs).

Resumable: re-run the same command to fill in missing trials. API keys come from OPENAI_API_KEY,
GEMINI_API_KEY and STEPFUN_API_KEY; local checkpoints from the Hugging Face Hub (or $MPTB_MINICPM_O_DIR /
$MPTB_VENUS_DIR). Live API runs cost money.

  python scripts/fd/run_experiment.py --models gemini-3.8-live --runs 3 --concurrency 6
  python scripts/fd/run_experiment.py --models minicpm-o-4.5 realtime-venus --runs 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.fd.adapters import available  # noqa: E402
from mptb.fd.audio_packs import load_packs  # noqa: E402
from mptb.fd.runner import run_model, select_scenarios  # noqa: E402


async def main_async(args: argparse.Namespace) -> None:
    """Run the selected agents one after another."""
    scenarios = {s["scenario_id"]: s for s in json.loads(Path(args.scenarios).read_text())}
    audio_root = Path(args.audio)
    packs = load_packs(audio_root)
    sids = select_scenarios(scenarios, packs, args.only, args.event_types, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        path = await run_model(model, scenarios, packs, audio_root, out, sids, runs=args.runs,
                               concurrency=args.concurrency, timeout=args.timeout, retries=args.retries,
                               backoff=args.backoff, sleep=args.sleep, log_every=args.log_every,
                               adapter_kwargs=json.loads(args.adapter_kwargs), resume=args.resume)
        print(f"  -> {path}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, choices=available(), help="agents to run")
    ap.add_argument("--runs", type=int, default=3, help="repeats per scenario")
    ap.add_argument("--scenarios", default="outputs/fd/scenarios.json", help="scenario list")
    ap.add_argument("--audio", default="outputs/fd/audio", help="audio pack directory")
    ap.add_argument("--out", default="outputs/fd/runs", help="output directory (<model>.jsonl per agent)")
    ap.add_argument("--only", nargs="*", help="run only these scenario ids")
    ap.add_argument("--event-types", nargs="*", help="run only these families (true_eot hold_pause true_int backchannel)")
    ap.add_argument("--limit", type=int, help="run only the first N scenarios (sorted by id)")
    ap.add_argument("--timeout", type=float, default=240.0, help="per-trial timeout (s)")
    ap.add_argument("--retries", type=int, default=2, help="retries of a trial whose session raised")
    ap.add_argument("--backoff", type=float, default=2.0, help="base retry back-off (s), doubled per attempt")
    ap.add_argument("--sleep", type=float, default=0.0, help="pause after each trial (s)")
    ap.add_argument("--log-every", type=int, default=5, help="progress line every N trials")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel sessions for live API agents (local agents always run serially)")
    ap.add_argument("--adapter-kwargs", default="{}",
                    help='JSON keyword arguments for the adapter, e.g. \'{"model_dir": "/path/to/ckpt"}\'')
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="do not skip trials already logged")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
