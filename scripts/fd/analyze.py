#!/usr/bin/env python
"""Step 5 of the full-duplex study: per-trial table, per-agent metrics, bACC bootstrap CIs and the
offline-online association (inputs of Table 4, Appendix F tables and Appendix G).

Writes to --out: fd_trials.csv, fd_by_model.json, fd_bacc_ci.json, fd_association.json.

  python scripts/fd/analyze.py --runs outputs/fd/runs --offline outputs/fd/offline.json --out outputs/fd/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.fd.analysis import analyze  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="outputs/fd/runs", help="directory of <model>.jsonl trial logs")
    ap.add_argument("--offline", nargs="*", default=["outputs/fd/offline.json"],
                    help="offline detector score file(s) from run_offline_detector.py (merged; missing files skipped)")
    ap.add_argument("--out", default="outputs/fd/analysis", help="output directory")
    args = ap.parse_args()

    offline = {}
    for p in args.offline:
        if Path(p).exists():
            offline.update(json.loads(Path(p).read_text()))
    res = analyze(Path(args.runs), offline, Path(args.out))
    for m, mm in res["by_model"].items():
        h, ci = mm["headline"], res["bacc_ci"][m]
        print(f"  {m:24s} valid {mm['n_trials_valid']}/{mm['n_trials_attempted']}  "
              f"START bACC {ci['START']['bacc']:.3f} [{ci['START']['lo']:.3f}, {ci['START']['hi']:.3f}]  "
              f"STOP bACC {ci['STOP']['bacc']:.3f} [{ci['STOP']['lo']:.3f}, {ci['STOP']['hi']:.3f}]  "
              f"STOP bias {h['stop_bias']:+.2f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
