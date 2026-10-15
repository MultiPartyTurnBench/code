#!/usr/bin/env python
"""Step 4 of the full-duplex study: score the agents as offline EOT / INT detectors (Appendix G).

Detector sessions are cached in --raw/<model>.jsonl; with --score-only (or a complete cache) the scores
are recomputed from the cache without contacting any model. Writes --out as
{model: {E_eot, I_int, eot, int, ..., per_event}} (merged into an existing file).

  python scripts/fd/run_offline_detector.py --models gemini-3.8-live step-audio-3-realtime \
      --root MultiPartyTurnBench --out outputs/fd/offline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.fd import offline as O  # noqa: E402
from mptb.fd.adapters import available  # noqa: E402


async def main_async(args: argparse.Namespace) -> None:
    """Collect (unless --score-only) and score every requested agent."""
    scenarios = json.loads(Path(args.scenarios).read_text())
    if args.limit:
        scenarios = scenarios[:args.limit]
    raw_dir = Path(args.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {}
    for model in args.models:
        raw_path = raw_dir / f"{model}.jsonl"
        if args.score_only:
            done = O.load_raw(raw_path)
        else:
            from mptb.data import Dataset
            done = await O.collect(model, scenarios, Dataset(args.root), raw_path, timeout=args.timeout,
                                   adapter_kwargs=json.loads(args.adapter_kwargs))
        report[model] = O.score_detector(scenarios, done)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        r = report[model]
        print(f"  {model}: E_eot={r['E_eot']} I_int={r['I_int']} ({r['n_sessions']} sessions)")
    print(f"-> {out}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, choices=available(), help="agents to score")
    ap.add_argument("--root", default=None, help="dataset root (default: $MPTB_ROOT)")
    ap.add_argument("--scenarios", default="outputs/fd/scenarios.json", help="scenario list")
    ap.add_argument("--raw", default="outputs/fd/offline_raw", help="raw detector-session cache directory")
    ap.add_argument("--out", default="outputs/fd/offline.json", help="offline score file (merged per model)")
    ap.add_argument("--score-only", action="store_true", help="rescore from the raw cache; never run a model")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-session timeout (s)")
    ap.add_argument("--adapter-kwargs", default="{}", help="JSON keyword arguments for the adapter")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N scenarios (0 = all)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
