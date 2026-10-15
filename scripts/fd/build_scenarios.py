#!/usr/bin/env python
"""Step 1 of the full-duplex study: select the 120 probe scenarios from the released gold annotations.

Deterministic (selection seed 20260929); with the default arguments it reproduces the paper's scenario
list exactly. See mptb/fd/scenarios.py for the selection rules.

  python scripts/fd/build_scenarios.py --root MultiPartyTurnBench --out outputs/fd/scenarios.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.data import Dataset  # noqa: E402
from mptb.fd import scenarios as S  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="dataset root (default: $MPTB_ROOT)")
    ap.add_argument("--out", default="outputs/fd/scenarios.json", help="output scenario list (JSON)")
    ap.add_argument("--per-family", type=int, default=S.PER_FAMILY, help="scenarios per event family")
    ap.add_argument("--max-per-conv", type=int, default=S.MAX_PER_CONVERSATION,
                    help="maximum scenarios drawn from one conversation")
    ap.add_argument("--seed", type=int, default=S.SELECTION_SEED, help="selection seed")
    args = ap.parse_args()

    scen = S.build_scenarios(Dataset(args.root), args.per_family, args.max_per_conv, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scen, indent=1))
    print(f"{len(scen)} scenarios -> {out}")
    print(" by family:", dict(Counter(s["event_type"] for s in scen)))
    print(" by N     :", dict(Counter(s["N"] for s in scen)))
    print(" by type  :", dict(Counter(s["conversation_type"] for s in scen)))
    print(" conversations:", len({s["conversation_id"] for s in scen}))


if __name__ == "__main__":
    main()
