#!/usr/bin/env python
"""Score a MultiPartyTurnBench submission (standard library only).

Usage (``--root`` is the downloaded dataset folder; it can also be set with ``MPTB_ROOT``):
  python scripts/evaluate.py score  --root MultiPartyTurnBench --split test --submission test_predictions.json
  python scripts/evaluate.py select --root MultiPartyTurnBench --split dev \
      --submission 0.5=dev_0.5.json 0.7=dev_0.7.json 0.9=dev_0.9.json

Submission format (JSON)::

  {"schema_version": 1,
   "predictions": [
     {"conversation_id": "test_00001",
      "speaker_1": {"eot": [t, ...], "interruption": [t, ...]},
      "speaker_2": {"eot": [...],    "interruption": [...]},
      ...}]}

Times are causal decision times in seconds on the conversation timeline, one list per speaker
channel and track. A conversation missing from the submission counts as "no predictions".

``score`` reports pooled recall, FPR and signed latency (p10/p50/p90, ms) per track, with breakdowns
by number of speakers and interaction type, and flags a test FPR above the 0.15 ceiling.
``select`` applies the benchmark's operating-point rule: per track, the threshold with the highest
recall subject to dev FPR <= 0.10. Freeze it and run ``score`` on the test split.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mptb.data import Dataset  # noqa: E402
from mptb.eval.schema import validate_submission  # noqa: E402
from mptb.eval.scorer import (DEV_FPR_BUDGET, TEST_FPR_CEILING, score_dataset,  # noqa: E402
                              select_operating_point)


def load_submission(path: str) -> dict:
    sub = json.loads(Path(path).read_text())
    errs = validate_submission(sub)
    if errs:
        sys.exit(f"invalid submission {path}:\n  " + "\n  ".join(errs[:20]))
    return sub


def fmt(m: dict) -> str:
    r = "  -  " if m["recall"] is None else f"{100 * m['recall']:5.1f}"
    f = "  -  " if m["fpr"] is None else f"{100 * m['fpr']:5.1f}"
    lat = "" if m["latency_p50"] is None else (
        f"  latency p10/p50/p90 {1000 * m['latency_p10']:.0f}/{1000 * m['latency_p50']:.0f}/"
        f"{1000 * m['latency_p90']:.0f} ms")
    return f"recall {r}%  FPR {f}%  (TP {m['tp']} FN {m['fn']} FP {m['fp']} TN {m['tn']})" + lat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["score", "select"])
    ap.add_argument("--root", help="dataset root (default: $MPTB_ROOT)")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--submission", nargs="+", required=True,
                    help="score: one JSON file; select: theta=path pairs")
    ap.add_argument("--out", help="write the full JSON report here")
    a = ap.parse_args()
    try:
        golds = Dataset(a.root).golds(a.split)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))

    if a.mode == "score":
        rep = score_dataset(golds, load_submission(a.submission[0]))
        print(f"{a.split}: {rep['n_conversations']} conversations")
        for t in ("eot", "interruption"):
            m = rep["overall"][t]
            flag = ""
            if a.split == "test" and m["fpr"] is not None and m["fpr"] > TEST_FPR_CEILING:
                flag = f"  [over the {TEST_FPR_CEILING:.2f} test FPR ceiling]"
            print(f"  {t.upper():13s} {fmt(m)}{flag}")
        for key, title in (("by_num_speakers", "N"), ("by_conversation_type", "type")):
            print(f"  by {title}:")
            for g, v in rep[key].items():
                print(f"    {g:14s} EOT {fmt(v['eot'])}")
                print(f"    {'':14s} INT {fmt(v['interruption'])}")
    else:
        subs = {}
        for item in a.submission:
            th, path = item.split("=", 1)
            subs[float(th)] = load_submission(path)
        rep = select_operating_point(golds, subs, budget=DEV_FPR_BUDGET)
        for t in ("eot", "interruption"):
            sel = rep[t]["selected"]
            print(f"  {t.upper():13s} " + ("no threshold meets the dev FPR budget" if sel is None else
                  f"theta={sel['theta']}  recall {100 * sel['recall']:.1f}%  FPR {100 * sel['fpr']:.1f}%"))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
