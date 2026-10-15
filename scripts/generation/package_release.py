#!/usr/bin/env python
"""Step 4 of corpus generation: package a working corpus into the public dataset layout.

Input (``--corpus``, produced by the other scripts in this folder): ``split.json``,
``noise_seeds.json``, ``qc_report.json``, ``scripts/``, ``render/``, ``gold/``, ``audio/{clean,noisy}/``.
Output (``--out``)::

    metadata/{split.json, conversations.csv, voices.csv, dataset_stats.json, qc_report.json, noise_seeds.json}
    generation/{stage1_system_prompt.txt, stage1_user_prompt_template.txt, generation_config.json}
    <split>/annotations/{events,anchors,scripts,render}/<id>.json
    <split>/audio/{noisy,clean}/<id>_speaker_<k>.flac       (copied byte-for-byte)
    <split>/manifest.jsonl, <split>/conversations.csv
    viewer/{index.html, data.js, mix/<id>.mp3}               (with --viewer; mixdowns need ffmpeg)
    SHA256SUMS

Dataset cards (README/DATASHEET/LICENSE) and the Hugging Face preview parquet files are not
generated here. Example:

    python scripts/generation/package_release.py --corpus corpus/v1 --out release/MultiPartyTurnBench --viewer
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mptb.eval.anchors import derive_anchors  # noqa: E402
from mptb.generation import noise as NZ  # noqa: E402
from mptb.generation.stage1_script import MAX_BACKCHANNEL_WORDS, SYSTEM_PROMPT, TONES, TYPE_STYLE  # noqa: E402
from mptb.generation.stage2_orchestrate import TimingConfig  # noqa: E402
from mptb.generation.stats import dataset_stats  # noqa: E402
from mptb.generation.strata import (AMBIENTS, DISCRETE_EVENTS, DOMAINS, TABLE_III,  # noqa: E402
                                    TARGET_DURATION_RANGE_S, load_voice_pool)

REPO = Path(__file__).resolve().parents[2]
VIEWER_HTML = REPO / "mptb" / "generation" / "viewer.html"
BINDING = REPO / "configs" / "voice_binding_kokoro.json"

NOISE_SEEDS_DESCRIPTION = (
    "Seed of the Stage-4 noise mixer for each conversation. Applying the noise stage to "
    "<split>/audio/clean/ with this seed and the noise profile in "
    "<split>/annotations/events/<id>.json (metadata.noise_profile) reproduces <split>/audio/noisy/.")

USER_PROMPT_TEMPLATE = """Write a {conversation_type} multi-party conversation.

Scenario / domain: {domain}
Conversation style: {style string for the type, see generation_config.json}
Overall tone: {tone}
Number of speakers: {N}
Speakers (id: gender, speaking style — give each a distinct personality/voice):
  - speaker_k: {gender}, {style}        (one line per speaker)

Approximate soft targets (aim NEAR these; they are guardrails, not exact quotas):
- Total utterances (TURN count): about {n_turns} floor-holding turns over ~{duration_min} minutes (mean turn ~{turn_len_s}s of speech, so ~{max(12, int(2.5*turn_len_s))}-{int(4*turn_len_s)} words each).
- Interruptions (CLAIM_FLOOR): about {n_interruptions} across the conversation (mix of INTERRUPTION and NON_FLOOR_TAKING_INTERRUPTION).
- Backchannels/laughter (VOCALIZE_NO_CLAIM): about {n_backchannels}.

Make it feel like a real, specific conversation about the scenario — concrete details, natural disfluency, distinct personalities. Every speaker participates. Output ONLY the JSON object {"turns": [...]}.

----
Repair message appended on validation failure (up to 3 attempts in total):
That output had these problems; fix them and return the full corrected JSON:
- {error 1}
- ... (up to 12 errors)
"""


def write_json(path: Path, obj: Any, indent: int = 1) -> None:
    """Write UTF-8 JSON with a trailing newline (the release's formatting)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False) + "\n")


def anchors_record(gold: Dict[str, Any]) -> Dict[str, Any]:
    """Per-speaker EOT/INT anchors (positives, negative and excluded spans) of one conversation."""
    out = {}
    for s, a in derive_anchors(gold).items():
        out[s] = {tr: {"positives": t.positives, "negative_spans": t.negative_spans,
                       "excluded_spans": t.excluded_spans}
                  for tr, t in (("eot", a.eot), ("interruption", a.interruption))}
    return out


def generation_config() -> Dict[str, Any]:
    """Machine-readable summary of every generation constant (``generation/generation_config.json``)."""
    return {
        "stage1_script_generation": {
            "model": "gpt-4.1", "temperature": 0.9, "max_tokens": 8000, "max_attempts": 3,
            "response_format": "json", "event_types": ["TURN", "CLAIM_FLOOR", "VOCALIZE_NO_CLAIM"],
            "max_backchannel_words": MAX_BACKCHANNEL_WORDS,
            "long_backchannel_rule": "a BACKCHANNEL with more than 3 words is relabelled as CLAIM_FLOOR / "
                                     "NON_FLOOR_TAKING_INTERRUPTION before validation",
            "type_style_strings": TYPE_STYLE, "tones": list(TONES), "domains": DOMAINS,
            "soft_targets_per_type": TABLE_III,
            "targets_rule": "n_turns=max(2N, round(D/turn_len_s)); n_interruptions=round(int_per_min*D_min*s_N); "
                            "n_backchannels=round(bc_per_min*D_min*s_N); s_N=1+0.1*(N-2)",
            "target_duration_s_uniform": list(TARGET_DURATION_RANGE_S)},
        "stage2_orchestration_timing": asdict(TimingConfig()),
        "stage3_tts": {"engine": "Kokoro-82M (hexgrad/Kokoro-82M)", "sample_rate": 24000, "speed": 1.0,
                       "trim_silence_dbfs": -40.0, "trim_margin_ms": 30, "voice_binding": "metadata/voices.csv"},
        "stage4_noise": {
            "ambient_scenes": AMBIENTS,
            "ambient_profiles_lowpass_hz_and_hum": {k: list(v) for k, v in NZ.AMBIENT_PROFILES.items()},
            "ambient_snr_db_uniform": [20.0, 38.0], "ambient_snr_floor_db": NZ.AMBIENT_SNR_FLOOR_DB,
            "discrete_event_labels": DISCRETE_EVENTS, "discrete_event_snr_db": NZ.DISCRETE_EVENT_SNR_DB,
            "discrete_event_slots": "k drawn uniformly from [0, 1, 1, 2]",
            "discrete_event_source": "MUSAN noise/sound-bible clips (<= 4 s)",
            "background_speech_probability": 0.2, "background_speech_snr_db": NZ.BG_SPEECH_SNR_DB,
            "background_speech_source": "MUSAN speech, 2-4 faded snippets of 1.5-3 s",
            "noise_shared_across_channels": True, "channel_peak_dbfs": -1.0},
        "stage5_qc": {"duration_tolerance_s": 0.6, "speech_ratio_range": [0.03, 0.99], "min_clip_coverage": 0.5,
                      "max_leakage": 0.3, "min_overlap_realism": 0.6},
    }


def conversation_row(cid: str, split: str, gold: Dict[str, Any], script: Dict[str, Any],
                     qc_passed: bool) -> Dict[str, Any]:
    """One row of ``conversations.csv``."""
    m = gold["metadata"]
    cnt: collections.Counter = collections.Counter()
    for a in derive_anchors(gold).values():
        cnt["eot_pos"] += len(a.eot.positives)
        cnt["eot_neg"] += len(a.eot.negative_spans)
        cnt["int_pos"] += len(a.interruption.positives)
        cnt["int_neg"] += len(a.interruption.negative_spans)
        cnt["int_excl"] += len(a.interruption.excluded_spans)
    lab = collections.Counter((e["label"], e.get("floor_outcome") or e.get("resolved_as") or e.get("sublabel"))
                              for e in gold["events"])
    np_ = m["noise_profile"]
    return {
        "conversation_id": cid, "split": split, "num_speakers": gold["num_speakers"],
        "conversation_type": gold["conversation_type"], "domain": m["domain"], "tone": m.get("tone", ""),
        "duration_s": round(gold["duration_s"], 3),
        "voices": ";".join(f"{s}={v}" for s, v in m["voices"].items()),
        "genders": ";".join(f"{s}={m['voice_styles'][s]['gender']}" for s in gold["speakers"]),
        "ambient": np_["ambient"], "snr_db": np_["snr_db"], "discrete_events": ";".join(np_["discrete_events"]),
        "background_speech": np_["background_speech"],
        "n_utterances": len(script["turns"]),
        "n_turn_yield": lab[("TURN", "yield")], "n_turn_hold": lab[("TURN", "hold")],
        "n_turn_interrupted": lab[("TURN", "interrupted")], "n_interruption": lab[("CLAIM_FLOOR", "INTERRUPTION")],
        "n_failed_claim": lab[("CLAIM_FLOOR", "NON_FLOOR_TAKING_INTERRUPTION")],
        "n_backchannel": lab[("VOCALIZE_NO_CLAIM", "BACKCHANNEL")], "n_laughter": lab[("VOCALIZE_NO_CLAIM", "LAUGHTER")],
        "eot_positives": cnt["eot_pos"], "eot_negative_spans": cnt["eot_neg"], "int_positives": cnt["int_pos"],
        "int_negative_spans": cnt["int_neg"], "int_excluded_spans": cnt["int_excl"], "qc_passed": qc_passed}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows with a header (csv module defaults, i.e. CRLF line endings)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def viewer_record(cid: str, split: str, gold: Dict[str, Any], script: Dict[str, Any],
                  render: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-conversation record for the offline viewer (``viewer/data.js``)."""
    kinds = {"holder": "turn", "claim": "failed_claim", "backchannel": "backchannel", "laughter": "laughter"}
    m, spk, ev = gold["metadata"], gold["speakers"], gold["events"]
    int_on = {(e["speaker"], round(e["onset"], 2)) for e in ev
              if e["label"] == "CLAIM_FLOOR" and e.get("resolved_as") == "INTERRUPTION"}
    turn_ev = {(e["speaker"], round(e["offset"], 2)): e for e in ev if e["label"] == "TURN"}
    utts = []
    for cl in sorted(render["clips"], key=lambda x: (x["start_s"], x["src_index"])):
        t = script["turns"][cl["src_index"]]
        cls = kinds[cl["kind"]]
        if cls == "turn" and (cl["speaker"], round(cl["start_s"], 2)) in int_on:
            cls = "interruption"
        note = ""
        e = turn_ev.get((cl["speaker"], round(cl["end_s"], 2)))
        if cl.get("truncated"):
            note = "cut off by interruption"
        elif e is not None and cls in ("turn", "interruption"):
            if e["floor_outcome"] == "yield":
                nh = e.get("next_holder")
                note = f"yields to S{spk.index(nh) + 1}" if nh else "yields (end)"
            elif e["floor_outcome"] == "hold":
                note = "holds the floor, pauses"
        if cls == "interruption" and t.get("target") in spk:
            note = f"takes floor from S{spk.index(t['target']) + 1}" + ("; " + note if note else "")
        utts.append([spk.index(cl["speaker"]), round(cl["start_s"], 2), round(cl["end_s"], 2), cls,
                     t["text"], note])
    anc: Dict[str, list] = {k: [] for k in ("eot_pos", "eot_neg", "eot_excl", "int_pos", "int_neg", "int_excl")}
    anchors = derive_anchors(gold)
    for k, s in enumerate(spk):
        a = anchors[s]
        anc["eot_pos"] += [[k, round(t, 2)] for t in a.eot.positives]
        anc["eot_neg"] += [[k, round(x, 2), round(y, 2)] for x, y in a.eot.negative_spans]
        anc["eot_excl"] += [[k, round(x, 2), round(y, 2)] for x, y in a.eot.excluded_spans]
        anc["int_pos"] += [[k, round(t, 2)] for t in a.interruption.positives]
        anc["int_neg"] += [[k, round(x, 2), round(y, 2)] for x, y in a.interruption.negative_spans]
        anc["int_excl"] += [[k, round(x, 2), round(y, 2)] for x, y in a.interruption.excluded_spans]
    floor = [[spk.index(e["speaker"]), round(e["onset"], 2), round(e["offset"], 2)] for e in ev if e["label"] == "TURN"]
    np_ = m["noise_profile"]
    noise = (f"{np_['ambient'].replace('_', ' ')}, {np_['snr_db']} dB SNR"
             + ("; events: " + ", ".join(x.replace("_", " ") for x in np_["discrete_events"])
                if np_["discrete_events"] else "")
             + ("; distant speech" if np_["background_speech"] else ""))
    return {"id": cid, "split": split, "N": gold["num_speakers"], "type": gold["conversation_type"],
            "domain": m["domain"], "tone": m.get("tone", ""), "dur": round(gold["duration_s"], 2),
            "speakers": spk,
            "voices": [[f"S{k + 1}", m["voices"][s], m["voice_styles"][s]["gender"], m["voice_styles"][s]["style"]]
                       for k, s in enumerate(spk)],
            "noise": noise, "utts": utts, "anchors": anc, "floor": floor}


def mixdown_mp3(noisy_files: List[Path], out_mp3: Path, ffmpeg: str) -> None:
    """Sum the noisy channels, normalise the peak to 0.89 and encode a 64 kbit/s mono MP3."""
    import numpy as np
    import soundfile as sf
    arrs, sr = [], 24000
    for p in noisy_files:
        x, sr = sf.read(str(p), dtype="float32")
        arrs.append(x)
    mix = np.zeros(max(len(a) for a in arrs), np.float32)
    for a in arrs:
        mix[:len(a)] += a
    mix = mix / (np.max(np.abs(mix)) or 1) * 0.89
    wav = out_mp3.with_suffix(".tmp.wav")
    sf.write(str(wav), mix, sr, subtype="PCM_16")
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame",
                    "-b:a", "64k", "-ac", "1", str(out_mp3)], check=True)
    wav.unlink()


def main() -> int:
    """Build the public layout (see module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="working corpus directory")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--viewer", action="store_true", help="also build viewer/ (index.html, data.js, mix/*.mp3)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable for the viewer mixdowns")
    args = ap.parse_args()

    src, out = Path(args.corpus), Path(args.out)
    split = json.loads((src / "split.json").read_text())["split"]
    cids = sorted(split, key=lambda c: (0 if split[c] == "dev" else 1, c))
    golds = {c: json.loads((src / "gold" / f"{c}.json").read_text()) for c in cids}
    scripts = {c: json.loads((src / "scripts" / f"{c}.json").read_text()) for c in cids}
    renders = {c: json.loads((src / "render" / f"{c}.json").read_text()) for c in cids}
    seeds = json.loads((src / "noise_seeds.json").read_text())
    qc = json.loads((src / "qc_report.json").read_text())
    qc_passed = {r["conversation_id"]: r["passed"] for r in qc["results"]}

    rows: Dict[str, List[dict]] = {"dev": [], "test": []}
    manifest: Dict[str, List[dict]] = {"dev": [], "test": []}
    for c in cids:
        s, g = split[c], golds[c]
        ann = out / s / "annotations"
        write_json(ann / "events" / f"{c}.json", g)
        write_json(ann / "anchors" / f"{c}.json", {"conversation_id": c, "split": s,
                                                    "num_speakers": g["num_speakers"],
                                                    "anchors": anchors_record(g)})
        write_json(ann / "scripts" / f"{c}.json", scripts[c])
        r = dict(renders[c])
        r["duration_s"] = round(r["duration_s"], 3)
        write_json(ann / "render" / f"{c}.json", r)
        audio = {}
        for kind in ("noisy", "clean"):
            audio[kind] = {}
            for spk in g["speakers"]:
                dst = out / s / "audio" / kind / f"{c}_{spk}.flac"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / "audio" / kind / f"{c}_{spk}.flac", dst)
                audio[kind][spk] = f"{s}/audio/{kind}/{c}_{spk}.flac"
        manifest[s].append({
            "conversation_id": c, "split": s, "num_speakers": g["num_speakers"],
            "conversation_type": g["conversation_type"], "duration_s": round(g["duration_s"], 3),
            "audio_noisy": audio["noisy"], "audio_clean": audio["clean"],
            "events": f"{s}/annotations/events/{c}.json", "anchors": f"{s}/annotations/anchors/{c}.json",
            "script": f"{s}/annotations/scripts/{c}.json", "render": f"{s}/annotations/render/{c}.json"})
        rows[s].append(conversation_row(c, s, g, scripts[c], qc_passed[c]))
    for s in ("dev", "test"):
        with open(out / s / "manifest.jsonl", "w") as f:
            for m in manifest[s]:
                f.write(json.dumps(m) + "\n")
        write_csv(out / s / "conversations.csv", rows[s])

    md = out / "metadata"
    write_json(md / "split.json", {s: [c for c in cids if split[c] == s] for s in ("dev", "test")})
    write_json(md / "noise_seeds.json", {"description": NOISE_SEEDS_DESCRIPTION,
                                         "seeds": {c: seeds[c] for c in cids}})
    write_json(md / "dataset_stats.json", dataset_stats(golds))
    write_json(md / "qc_report.json", qc)
    write_csv(md / "conversations.csv", rows["dev"] + rows["test"])
    binding = json.loads(BINDING.read_text())["binding"]
    dev_voices = {v for c in cids if split[c] == "dev" for v in golds[c]["metadata"]["voices"].values()}
    with open(md / "voices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["voice_id", "kokoro_voice", "gender", "style", "split_pool"])
        for v in load_voice_pool():
            w.writerow([v["voice_id"], binding[v["voice_id"]]["kokoro_voice"], v["gender"], v["style"],
                        "dev" if v["voice_id"] in dev_voices else "test"])

    gen = out / "generation"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "stage1_system_prompt.txt").write_text(SYSTEM_PROMPT + "\n")
    (gen / "stage1_user_prompt_template.txt").write_text(USER_PROMPT_TEMPLATE)
    write_json(gen / "generation_config.json", generation_config(), indent=2)

    if args.viewer:
        v = out / "viewer"
        (v / "mix").mkdir(parents=True, exist_ok=True)
        recs = [viewer_record(c, split[c], golds[c], scripts[c], renders[c]) for c in cids]
        (v / "data.js").write_text("window.MPTB=" + json.dumps(recs, ensure_ascii=False, separators=(",", ":")) + ";\n")
        shutil.copyfile(VIEWER_HTML, v / "index.html")
        for c in cids:
            mixdown_mp3([src / "audio" / "noisy" / f"{c}_{spk}.flac" for spk in golds[c]["speakers"]],
                        v / "mix" / f"{c}.mp3", args.ffmpeg)

    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}"
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"packaged {len(cids)} conversations -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
