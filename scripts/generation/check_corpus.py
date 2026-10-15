#!/usr/bin/env python
"""Step 3 of corpus generation: acoustic QC and corpus-level checks of a working corpus.

Writes ``<out>/qc_report.json`` (released as ``metadata/qc_report.json``) and
``<out>/corpus_checks.json`` and exits non-zero unless every check passes:
oracle round trip (recall 1, no false positives on both tracks), QC passed for every
conversation, every (N x type) cell populated, and dev/test voice-disjoint and balanced.
Example:

    python scripts/generation/check_corpus.py --out corpus/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.generation.checks import check_corpus  # noqa: E402
from mptb.generation.qc import qc_conversation, summarize  # noqa: E402


def main() -> int:
    """Run QC and checks (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="working corpus directory (gold/, render/, audio/clean/, split.json)")
    ap.add_argument("--quiet", action="store_true", help="do not print per-conversation QC lines")
    args = ap.parse_args()

    root = Path(args.out)
    split = json.loads((root / "split.json").read_text())["split"]
    cids = sorted(split, key=lambda c: (split[c] != "dev", c))
    golds = {c: json.loads((root / "gold" / f"{c}.json").read_text()) for c in cids}
    results = []
    for cid in cids:
        r = qc_conversation(golds[cid], json.loads((root / "render" / f"{cid}.json").read_text()),
                            str(root / "audio" / "clean"))
        results.append(r)
        if not args.quiet:
            print(f"  [{cid}] {'PASS' if r['passed'] else 'FAIL'} overlap={r['overlap_ok']}/{r['overlap_events']}"
                  + ("" if r["passed"] else f" | {r['fails']}"))
    qc = summarize(results)
    (root / "qc_report.json").write_text(json.dumps(qc, indent=2))
    report = check_corpus(golds, qc, split)
    (root / "corpus_checks.json").write_text(json.dumps(report, indent=2))
    rt = report["oracle_round_trip"]["tracks"]
    print(f"oracle round trip : {'PASS' if report['oracle_round_trip']['pass'] else 'FAIL'} "
          f"(EOT recall={rt['eot']['recall']}, INT recall={rt['interruption']['recall']})")
    print(f"acoustic QC       : {qc['n_passed']}/{qc['n_conversations']} passed")
    print(f"cell balance      : {'PASS' if report['balance']['pass'] else 'FAIL'} "
          f"(equal counts: {report['balance']['balanced']})")
    print(f"dev/test split    : {'PASS' if report['split']['pass'] else 'FAIL'} "
          f"(voices {report['split']['n_voices']}, overlap {report['split']['voice_overlap']})")
    print(f"ALL CHECKS        : {'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
