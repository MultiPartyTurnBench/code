#!/usr/bin/env python
"""Run baselines on a real-audio slice and write its report (paper Table 14, Appendix G.1/H.2).

Each backend's threshold grid is scored on the whole slice with the MultiPartyTurnBench scorer and the
operating point is selected in-slice (highest recall with FPR <= 0.10). Results are merged into ``--report``
(one entry per backend), so backends can run in separate environments (Kyutai needs ``moshi``). Only
aggregate metrics are written. Examples::

    python scripts/realdata/run_slice.py --slice turnbench --data work/turnbench_slice \
        --backends smart_turn vap --vap-repo VoiceActivityProjection --report results/realdata/turnbench_report.json
    python scripts/realdata/run_slice.py --slice ami --data work/ami_slice --audio-dir ami_audio \
        --backends energy smart_turn vap --vap-repo VoiceActivityProjection --report results/realdata/ami_report.json
    python scripts/realdata/run_slice.py --slice ami --data work/ami_slice --audio-dir ami_audio \
        --backends kyutai --report results/realdata/ami_report.json      # in the Kyutai environment
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mptb.eval.scorer import DEV_FPR_BUDGET  # noqa: E402
from mptb.realdata.slices import (BACKENDS, load_ami_slice, load_turnbench_slice, make_predictor,  # noqa: E402
                                  run_backend)

CORPUS = {"turnbench": ("turnbench_dev_real", 2), "ami": ("ami_real", 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", required=True, choices=sorted(CORPUS), help="which real-audio slice")
    ap.add_argument("--data", required=True,
                    help="slice directory from prepare_turnbench.py (turnbench) or build_ami_slice.py (ami)")
    ap.add_argument("--audio-dir", help="ami only: directory with the <meeting>.Headset-<k>.wav files")
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS),
                    help="baselines to run (default: all)")
    ap.add_argument("--report", required=True, help="report JSON to create or update")
    ap.add_argument("--budget", type=float, default=DEV_FPR_BUDGET, help="FPR budget (default 0.10)")
    ap.add_argument("--device", default="cuda", help="torch device for the learned backends")
    ap.add_argument("--smart-turn-dir", help="local Smart Turn v2 snapshot (default: download from the Hub)")
    ap.add_argument("--vap-repo", help="VoiceActivityProjection checkout (default: $VAP_REPO)")
    ap.add_argument("--vap-checkpoint",
                    help="VAP weights (default: <vap-repo>/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt)")
    a = ap.parse_args()

    if a.slice == "turnbench":
        anchors, load_audio = load_turnbench_slice(a.data)
    else:
        if not a.audio_dir:
            ap.error("--audio-dir is required for --slice ami")
        anchors, load_audio = load_ami_slice(a.data, a.audio_dir)
    corpus, n_spk = CORPUS[a.slice]
    print(f"{a.slice}: {len(anchors)} conversations (N={n_spk})")

    rp = Path(a.report)
    report = json.loads(rp.read_text()) if rp.exists() else {}
    report.update({"corpus": corpus, "n_conv": len(anchors), "budget": a.budget, "num_speakers": n_spk})
    report.setdefault("backends", {})
    for backend in a.backends:
        predictor = make_predictor(backend, a.device, a.smart_turn_dir, a.vap_repo, a.vap_checkpoint)
        entry = run_backend(backend, predictor, anchors, load_audio, a.budget)
        for tr, c in entry["tracks"].items():
            if c.get("theta") is not None:
                print(f"  {backend} {tr:12s}: theta={c['theta']} qualified={c['qualified']} "
                      f"recall={c['recall']:.3f} fpr={c['fpr']:.3f} lat50={c['lat_p50']}")
        report["backends"][backend] = entry
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2))
        del predictor
    print(f"report: {rp}")


if __name__ == "__main__":
    main()
