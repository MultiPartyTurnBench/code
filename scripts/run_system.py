#!/usr/bin/env python
"""Run one offline baseline on MultiPartyTurnBench (paper Table 3).

Two steps, as in the paper:

  # 1. inference over all 240 conversations (shard across GPUs if you like)
  CUDA_VISIBLE_DEVICES=0 python scripts/run_system.py --system vap --root MultiPartyTurnBench --out results --shard 0 --nshards 2
  CUDA_VISIBLE_DEVICES=1 python scripts/run_system.py --system vap --root MultiPartyTurnBench --out results --shard 1 --nshards 2
  # 2. dev operating point (FPR <= 0.10) per track, then a single test evaluation
  python scripts/run_system.py --system vap --root MultiPartyTurnBench --out results --assemble

Outputs go to ``<out>/<system>/``: ``raw/`` or ``scores/`` (step 1), ``report.json`` and
``predictions/<track>.json`` (step 2). Systems: see ``--help`` or ``mptb/systems/__init__.py``.

Some systems need external code checkouts (see README): ``VAP_REPO`` (vap), ``MUVAP_REPO`` (muvap),
``TURNBENCH_REPO`` (mimi_ep). Model weights are downloaded from the Hugging Face Hub.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mptb.data import Dataset  # noqa: E402
from mptb.eval.scorer import DEV_FPR_BUDGET  # noqa: E402
from mptb.systems import SYSTEMS  # noqa: E402
from mptb.systems import harness  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS),
                    help="; ".join(f"{k}: {v.description}" for k, v in SYSTEMS.items()))
    ap.add_argument("--root", help="dataset root (default: $MPTB_ROOT)")
    ap.add_argument("--out", required=True, help="results directory (one sub-folder per system)")
    ap.add_argument("--shard", type=int, default=0, help="process conversations ids[shard::nshards]")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="recompute conversations that already have outputs")
    ap.add_argument("--assemble", action="store_true", help="select the dev operating point and score test")
    ap.add_argument("--budget", type=float, default=DEV_FPR_BUDGET, help="dev FPR budget (default 0.10)")
    ap.add_argument("--dualturn-map", metavar="EOT_HEAD,raw|inv,INT_HEAD",
                    help="dualturn only: fix the head mapping instead of choosing it on dev")
    a = ap.parse_args()

    ds = Dataset(a.root)
    system = SYSTEMS[a.system]
    out = Path(a.out)
    if not a.assemble:
        harness.compute(system, ds, out, a.shard, a.nshards, a.overwrite)
        return
    extra = None
    if a.system == "dualturn":
        from mptb.systems.dualturn import mapping_view
        if a.dualturn_map:
            eot_head, inv, int_head = a.dualturn_map.split(",")
            mapping = (eot_head, inv == "inv", int_head)
        else:
            mapping = harness.select_dualturn_mapping(ds, out, budget=a.budget)
        print(f"dualturn mapping: EOT = {'1 - ' if mapping[1] else ''}{mapping[0]}, INT = {mapping[2]}")
        system.transform = mapping_view(*mapping)
        extra = {"dualturn_mapping": {"eot_head": mapping[0], "eot_inverted": mapping[1], "int_head": mapping[2]}}
    harness.assemble(system, ds, out, budget=a.budget, extra=extra)


if __name__ == "__main__":
    main()
