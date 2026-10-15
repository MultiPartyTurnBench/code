"""End-to-end, checkpointed corpus generation (paper Section 3.1, Figure 2).

For each :class:`~mptb.generation.strata.ConvSpec`:
Stage 1 (LLM script) -> Stage 2 (orchestration, gold events) -> Stage 3 (Kokoro TTS) and
channel assembly -> Stage 4 (shared noise scene). Artifacts are written to a working corpus
directory and re-used on restart::

    <out>/generation_manifest.jsonl   conversation specs (input; scripts/generation/sample_manifest.py)
    <out>/split.json                  dev/test assignment and voice pools
    <out>/noise_seeds.json            {conversation_id: Stage-4 seed} actually used
    <out>/scripts/<id>.json           Stage-1 tagged script
    <out>/render/<id>.json            Stage-2 placement plan + Stage-4 noise manifest
    <out>/gold/<id>.json              gold event log
    <out>/audio/clean/<id>_<spk>.flac clean per-speaker channels (QC reference)
    <out>/audio/noisy/<id>_<spk>.flac noisy per-speaker channels (benchmark input)

``scripts/generation/package_release.py`` converts this directory into the public dataset layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..eval import schema
from .assemble import assemble_channels, write_channels
from .noise import add_noise, noise_seed
from .stage1_script import generate_script, normalize_script, save_script
from .stage2_orchestrate import TimingConfig, build_gold, orchestrate, render_record
from .strata import ConvSpec

SAMPLE_RATE = 24000


def voice_map(spec: ConvSpec) -> Dict[str, str]:
    """``{speaker_key: voice_id}`` of a spec."""
    return {sk: v["voice_id"] for sk, v in spec.voices.items()}


def gold_metadata(spec: ConvSpec, script: Dict[str, Any],
                  sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
    """The ``metadata`` block of a gold event log."""
    return {
        "domain": spec.domain,
        "voices": voice_map(spec),
        "voice_styles": {sk: {"gender": v["gender"], "style": v["style"]}
                         for sk, v in spec.voices.items()},
        "noise_profile": spec.noise_profile,
        "sample_rate": sample_rate,
        "tone": script.get("tone"),
        "generator_model": script.get("generator_model"),
        "engine": "kokoro",
    }


def _is_complete(root: Path, cid: str) -> bool:
    gold_path = root / "gold" / f"{cid}.json"
    render_path = root / "render" / f"{cid}.json"
    if not (gold_path.exists() and render_path.exists()):
        return False
    gold = json.loads(gold_path.read_text())
    files = list(gold["audio"]["files"].values())
    return (all((root / "audio" / "clean" / f).exists() for f in files)
            and all((root / "audio" / "noisy" / f).exists() for f in files))


def generate_conversation(spec: ConvSpec, tts: Any, llm_client: Any, out_root: str,
                          musan: Dict[str, List[str]], seed_for_noise: int,
                          gen_model: Optional[str] = None, overwrite: bool = False,
                          verbose: bool = True) -> Dict[str, Any]:
    """Generate one conversation into ``out_root`` and return its gold event log.

    An existing script is re-used (``llm_client`` may then be None); a complete conversation
    is skipped unless ``overwrite``. ``seed_for_noise`` seeds Stage 4.
    """
    root = Path(out_root)
    cid = spec.conversation_id
    if not overwrite and _is_complete(root, cid):
        if verbose:
            print(f"  [{cid}] already complete -- skipping")
        return json.loads((root / "gold" / f"{cid}.json").read_text())

    script_path = root / "scripts" / f"{cid}.json"
    if script_path.exists():
        script = json.loads(script_path.read_text())
    else:
        if llm_client is None:
            raise RuntimeError(f"[{cid}] no script on disk and no LLM client given")
        script = generate_script(llm_client, spec, model=gen_model, verbose=verbose)
        save_script(script, str(root / "scripts"))
    if normalize_script(script):
        save_script(script, str(root / "scripts"))

    vmap = voice_map(spec)
    result = orchestrate(script, tts.duration_fn(vmap), seed=spec.seed, cfg=TimingConfig())
    render = render_record(cid, result, SAMPLE_RATE)

    clean = assemble_channels(result.clips, vmap, tts, spec.speakers, result.duration_s,
                              sample_rate=SAMPLE_RATE)
    write_channels(clean, str(root / "audio" / "clean"), cid, sample_rate=SAMPLE_RATE)
    noisy, noise_manifest = add_noise(clean, spec.noise_profile, musan, SAMPLE_RATE, seed_for_noise)
    render["noise"] = noise_manifest
    render["noisy_files"] = write_channels(noisy, str(root / "audio" / "noisy"), cid,
                                           sample_rate=SAMPLE_RATE)
    schema.save_json(render, root / "render" / f"{cid}.json")

    gold = build_gold(script, result, spec_meta=gold_metadata(spec, script))
    errs = schema.validate_gold(gold)
    if errs:
        raise RuntimeError(f"[{cid}] gold failed validation: {errs[:5]}")
    schema.save_json(gold, root / "gold" / f"{cid}.json")
    if verbose:
        print(f"  [{cid}] N={spec.num_speakers} {spec.conversation_type} "
              f"dur={gold['duration_s']:.1f}s clips={len(result.clips)} "
              f"events={len(gold['events'])} -> OK")
    return gold


def generate_corpus(specs: List[ConvSpec], tts: Any, llm_client: Any, out_root: str,
                    musan: Dict[str, List[str]], gen_model: Optional[str] = None,
                    overwrite: bool = False, verbose: bool = True) -> List[str]:
    """Generate every spec into ``out_root``; returns the ids that completed.

    Stage-4 seeds are read from ``<out_root>/noise_seeds.json`` when present (e.g. the
    released ``metadata/noise_seeds.json``) and otherwise derived with
    :func:`mptb.generation.noise.noise_seed`; the seeds used are written back. A failure
    on one conversation is reported and does not stop the run.
    """
    root = Path(out_root)
    seeds_path = root / "noise_seeds.json"
    seeds: Dict[str, int] = json.loads(seeds_path.read_text()) if seeds_path.exists() else {}
    done: List[str] = []
    for spec in specs:
        cid = spec.conversation_id
        seeds.setdefault(cid, noise_seed(cid, spec.seed))
        seeds_path.write_text(json.dumps(seeds, indent=1, sort_keys=True))
        try:
            generate_conversation(spec, tts, llm_client, out_root, musan, seeds[cid],
                                  gen_model=gen_model, overwrite=overwrite, verbose=verbose)
            done.append(cid)
        except Exception as e:  # keep going; one bad conversation should not stop the corpus
            print(f"  [{cid}] FAILED: {e!r}")
    return done
