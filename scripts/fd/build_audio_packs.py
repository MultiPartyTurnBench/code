#!/usr/bin/env python
"""Step 2 of the full-duplex study: render the audio pack (stream / inject / metadata) of every scenario.

Uses the released noisy audio and Kokoro TTS for the spoken STOP questions (GPU recommended).
See mptb/fd/audio_packs.py.

  python scripts/fd/build_audio_packs.py --root MultiPartyTurnBench \
      --scenarios outputs/fd/scenarios.json --out outputs/fd/audio
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.data import Dataset  # noqa: E402
from mptb.fd.audio_packs import write_packs  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="dataset root (default: $MPTB_ROOT)")
    ap.add_argument("--scenarios", default="outputs/fd/scenarios.json", help="scenario list from build_scenarios.py")
    ap.add_argument("--out", default="outputs/fd/audio", help="output directory (one sub-directory per scenario)")
    ap.add_argument("--only", nargs="*", help="build only these scenario ids")
    ap.add_argument("--device", default="cuda", help="torch device for Kokoro TTS")
    args = ap.parse_args()

    scen = json.loads(Path(args.scenarios).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    packs = write_packs(scen, Dataset(args.root), out, only=args.only, device=args.device)
    print(f"{len(packs)} packs -> {out}")


if __name__ == "__main__":
    main()
