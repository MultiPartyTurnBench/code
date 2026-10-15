#!/usr/bin/env python
"""Build the AMI four-party slice (paper Table 14 "AMI meetings", Appendix H.2): derive a gold event log per
meeting from AMI's manual annotations, validate it and check the oracle round trip.

Inputs (AMI Meeting Corpus, CC BY 4.0):

* manual annotations ``ami_public_manual_1.6.2.zip`` (unzip; ``segments/`` and ``words/`` are used),
  from https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
* the 16 kHz individual-headset WAVs ``<meeting>.Headset-{0..3}.wav`` of each meeting, from
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.Headset-<k>.wav

Example::

    python scripts/realdata/build_ami_slice.py --annotations ami_manual --audio-dir ami_audio --out work/ami_slice

Writes ``<out>/gold/<meeting>.json`` (audio file names relative to ``--audio-dir``) and ``<out>/manifest.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mptb.eval.anchors import derive_anchors  # noqa: E402
from mptb.eval.schema import validate_gold  # noqa: E402
from mptb.eval.scorer import TrackResult, score_track  # noqa: E402
from mptb.realdata.ami import SLICE_MEETINGS, build_ami_gold, headset_files  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="unzipped ami_public_manual_1.6.2 (has segments/, words/)")
    ap.add_argument("--audio-dir", required=True, help="directory with <meeting>.Headset-<k>.wav")
    ap.add_argument("--out", required=True, help="output directory (gold/, manifest.jsonl)")
    ap.add_argument("--meetings", nargs="+", default=list(SLICE_MEETINGS),
                    help="meeting ids (default: the paper's 15 meetings)")
    a = ap.parse_args()

    import soundfile as sf
    out = Path(a.out)
    (out / "gold").mkdir(parents=True, exist_ok=True)
    manifest, tot, hours = [], {"eot_pos": 0, "eot_neg": 0, "int_pos": 0, "int_neg": 0}, 0.0
    for m in a.meetings:
        files = headset_files(a.annotations, m)
        dur = min(sf.info(os.path.join(a.audio_dir, f)).duration for f in files.values())
        gold = build_ami_gold(a.annotations, m, files, dur)
        errs = validate_gold(gold)
        if errs:
            print(f"  {m}: SCHEMA ERRORS {errs[:5]}")
            continue
        anchors = derive_anchors(gold)
        oracle = {tr: TrackResult() for tr in ("eot", "interruption")}
        for sa in anchors.values():
            for tr in oracle:
                oracle[tr] += score_track(getattr(sa, tr), sorted(getattr(sa, tr).positives))
        ep = sum(len(x.eot.positives) for x in anchors.values())
        en = sum(len(x.eot.negative_spans) for x in anchors.values())
        ip = sum(len(x.interruption.positives) for x in anchors.values())
        inn = sum(len(x.interruption.negative_spans) for x in anchors.values())
        tot["eot_pos"] += ep
        tot["eot_neg"] += en
        tot["int_pos"] += ip
        tot["int_neg"] += inn
        hours += dur / 3600
        (out / "gold" / f"{m}.json").write_text(json.dumps(gold, indent=1))
        manifest.append({"conversation_id": m, "num_speakers": gold["num_speakers"],
                         "conversation_type": gold["conversation_type"], "duration_s": round(dur, 1),
                         "gold": f"gold/{m}.json", "audio": gold["audio"]["files"]})
        e, i = oracle["eot"], oracle["interruption"]
        print(f"  {m}: {dur / 60:.1f} min  EOT +{ep}/-{en}  INT +{ip}/-{inn}  "
              f"oracle EOT R={e.recall:.2f}/FPR={e.fpr:.2f}  INT R={i.recall:.2f}/FPR={i.fpr:.2f}")
    with open(out / "manifest.jsonl", "w") as f:
        for r in manifest:
            f.write(json.dumps(r) + "\n")
    print(f"built {len(manifest)} meetings ({hours:.2f} h) -> {out}; anchors {tot}")


if __name__ == "__main__":
    main()
