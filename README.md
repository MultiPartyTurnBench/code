# MultiPartyTurnBench

Code for **MultiPartyTurnBench (MPTB): Benchmarking Turn-taking Dynamics in Multi-Party Spoken Dialogues**.

MPTB evaluates *multi-party floor control* on 240 synthetic, channel-separated conversations with 2–5
speakers and six interaction types, each with an exact, role-relative event log. Two causal tasks are scored
per speaker channel:

- **End-of-Turn (EOT):** detect when the floor holder *yields* (positives) without firing during mid-turn
  *hold* pauses (hard negatives).
- **Interruption (INT):** detect when a listener *successfully takes* the floor (positives) without firing on
  *backchannels or laughter* (hard negatives). Failed floor claims are excluded from scoring.

This repository contains:

1. the **official scorer** (standard library only) to evaluate any system on the benchmark;
2. the **12 offline baselines** of the paper and the harness that runs them;
3. the **full-duplex behavioural study** of five voice agents;
4. the **real-audio validation slices** (TurnBench dev, AMI);
5. the **data-generation pipeline** that produced the corpus.

The data are not in this repository: they are on the Hugging Face Hub as the dataset
**`MultiPartyTurnBench`**.

---

## Contents

- [Installation](#installation)
- [Data](#data)
- [Evaluating a system](#evaluating-a-system)
- [Reproducing the paper](#reproducing-the-paper)
  - [Offline baselines (Table 3)](#1-offline-baselines)
  - [Real-audio slices (Table 14)](#2-real-audio-slices)
  - [Full-duplex behavioural study (Table 4, Appendices F–G)](#3-full-duplex-behavioural-study)
  - [Data generation (Section 3, Appendices C–D)](#4-data-generation)
- [Repository layout](#repository-layout)
- [Code reference](#code-reference)
- [Tests](#tests)
- [License](#license)

---

## Installation

```bash
git clone <this repository> && cd MultiPartyTurnBench
pip install -e .            # the scorer has no dependencies (Python >= 3.9)
```

Everything else is optional and split by purpose; each file in `requirements/` pins the versions used
for the paper:

| File | Needed for |
|---|---|
| `requirements/systems.txt` | baselines `energy`, `webrtc`, `silero`, `smart_turn`, `smart_turn_int`, `kyutai`, `kyutai_int`, `vap`, `mimi_ep`, `dualturn`, `turn1mini` |
| `requirements/muvap.txt` | baseline `muvap` |
| `requirements/espnet.txt` | baselines `espnet_tt_mix`, `espnet_tt_perchannel` |
| `requirements/anchor.txt` | baseline `wavlm_anchor` |
| `requirements/realdata.txt` | real-audio slices (on top of `systems.txt`) |
| `requirements/fd.txt`, `requirements/fd_local.txt` | full-duplex study (API agents / local agents) |
| `requirements/generation.txt` | data-generation pipeline (Kokoro TTS, OpenAI client) |

The baselines need separate environments because their dependencies conflict (e.g. MuVAP and the ESPnet
models need different `torch`/`transformers` versions).

## Data

```bash
huggingface-cli download <owner>/MultiPartyTurnBench --repo-type dataset --local-dir MultiPartyTurnBench
export MPTB_ROOT=$PWD/MultiPartyTurnBench      # every script also accepts --root
```

The dataset has a `dev/` split (48 conversations; tune thresholds here only) and a `test/` split (192). Each
split has per-speaker noisy (evaluation input) and clean 24 kHz FLAC channels and the annotations
(`events` = gold event log, `anchors` = derived scoring anchors, `scripts`, `render`). `metadata/` holds the
split, voices, statistics, QC report and noise seeds. See the dataset card for the full format.
`mptb.data.Dataset` reads this layout.

## Evaluating a system

1. **Run your system causally on every speaker channel** of `<split>/audio/noisy/` and write the times at
   which it *commits* to an EOT or INT decision:

   ```json
   {"schema_version": 1,
    "predictions": [
      {"conversation_id": "test_00001",
       "speaker_1": {"eot": [3.21, 17.04], "interruption": [9.87]},
       "speaker_2": {"eot": [], "interruption": []}}]}
   ```

   Times are seconds on the conversation timeline. A decision may use only audio up to its time. If your
   system outputs scores, convert them to decisions with the benchmark's commit rule (rising edge above a
   threshold θ, 2 s refractory period; `mptb.systems.common.rising_edges` and `refractory`).

2. **Choose θ on dev.** Score one dev submission per candidate θ and keep, per track, the highest recall
   with dev FPR ≤ 0.10:

   ```bash
   python scripts/evaluate.py select --split dev --submission 0.5=dev_0.5.json 0.7=dev_0.7.json 0.9=dev_0.9.json
   ```

3. **Score test once** at the frozen θ:

   ```bash
   python scripts/evaluate.py score --split test --submission test.json --out report.json
   ```

   The scorer reports recall, FPR and signed latency (p10/p50/p90), pooled and by number of speakers and
   interaction type. A system is *within budget* if dev FPR ≤ 0.10 and test FPR ≤ 0.15.

**Scoring rule** (`mptb/eval/scorer.py`, following TurnBench). Predictions inside excluded spans are dropped.
Each gold positive at time *t* is matched one-to-one to the earliest unclaimed prediction in
[*t* − 0.25 s, *t* + 3.0 s] (TP, else FN). Each hard-negative span counts once: FP if any remaining
prediction falls in it, else TN. Recall = TP/(TP+FN), FPR = FP/(FP+TN), pooled over all channels and
conversations; latency = *t*_pred − *t*_gold over matched positives.

From Python:

```python
from mptb.data import Dataset
from mptb.eval import score_dataset
golds = Dataset("MultiPartyTurnBench").golds("test")
report = score_dataset(golds, submission)        # submission: dict in the format above
print(report["overall"]["eot"]["recall"], report["overall"]["interruption"]["fpr"])
```

---

## Reproducing the paper

All commands run from the repository root with `MPTB_ROOT` set. Model weights are downloaded from the
Hugging Face Hub at first use.

### 1. Offline baselines

`scripts/run_system.py` runs one system in two steps: inference over all 240 conversations (`raw/` or
`scores/`), then `--assemble`, which selects the operating point on dev per track and scores test once
(`report.json`, `predictions/<track>.json`).

```bash
for s in energy webrtc silero smart_turn smart_turn_int kyutai kyutai_int vap mimi_ep dualturn turn1mini; do
  python scripts/run_system.py --system $s --out results                # add --shard k --nshards n to split over GPUs
  python scripts/run_system.py --system $s --out results --assemble
done
# separate environments (see Installation): muvap, espnet_tt_mix, espnet_tt_perchannel, wavlm_anchor
```

| System id | Paper row | Inputs / external code | INT score |
|---|---|---|---|
| `energy`, `webrtc`, `silero` | RMS Energy, WebRTC VAD, Silero VAD | per channel | speech onset while another channel is voiced |
| `smart_turn` (EOT), `smart_turn_int` (INT) | Smart Turn v2 | `pipecat-ai/smart-turn-v2` | 1 − P(complete) (inversion) |
| `kyutai` (EOT), `kyutai_int` (INT) | Kyutai SVAD | `kyutai/stt-1b-en_fr-candle` (moshi) | 1 − P(end of turn) (inversion) |
| `vap` | VAP | `$VAP_REPO` = VoiceActivityProjection @ f39a78b; `viks66/VAP_checkpoints` oto.ckpt | P(now) of s vs rest, rest voice-active |
| `muvap` | MuVAP | `$MUVAP_REPO` = MuVAP @ 0fe3c66; `Haotian-Qi/MuVAP` vap-speaker-mimi | as VAP, rest gated by WebRTC |
| `mimi_ep` | Mimi Endpointer | `$TURNBENCH_REPO` = turnbench @ 38a6f87; `viks66/mimi-endpointer` | P(user), s vs rest |
| `dualturn` | DualTurn | `anyreach-ai/dualturn-qwen2.5-mimi-0.5B` | head chosen on dev (`hold_probs`) |
| `turn1mini` | Turn-1-Mini | `p99turn` 1.2.3 | — (EOT only) |
| `wavlm_anchor` | WavLM Anchor | `ZhuoyanTao/causal-wavlm-turn-taking`, ESPnet Universa | 1 − P(continue) |
| `espnet_tt_mix`, `espnet_tt_perchannel` | ESPnet Turn-Taking (mix / per-channel) | `espnet/Turn_taking_prediction_SWBD`, ESPnet @ 750e3749 | P(interruption) |

Every system uses the same commit rule and the same dev-selection rule. Grid thresholds are given in
`mptb/systems/*.py`. Score-stream systems use a fixed grid plus quantiles of their own dev scores. INT of
score-stream systems is gated by "another speaker voice-active" (WebRTC VAD). DualTurn's head mapping is
chosen on dev automatically (`--dualturn-map` fixes it). Results can differ slightly across GPUs and
library versions.

### 2. Real-audio slices

The real data are not redistributed. TurnBench dev (`mundo-ai/turn-benchmark-dev`, gated, non-commercial)
and AMI (CC BY 4.0) have to be downloaded separately. Only aggregate metrics are written.

```bash
pip install -r requirements/realdata.txt
# TurnBench dev slice (first 12 dev conversations, N = 2)
python scripts/realdata/prepare_turnbench.py --parquet-dir TB_DEV --out work/turnbench_slice
python scripts/realdata/run_slice.py --slice turnbench --data work/turnbench_slice \
    --backends smart_turn vap kyutai --vap-repo $VAP_REPO --report results/realdata/turnbench_report.json
# AMI slice (15 four-party meetings, headset audio)
python scripts/realdata/build_ami_slice.py --annotations ami_manual --audio-dir ami_audio --out work/ami_slice
python scripts/realdata/run_slice.py --slice ami --data work/ami_slice --audio-dir ami_audio \
    --backends energy smart_turn vap kyutai --vap-repo $VAP_REPO --report results/realdata/ami_report.json
```

Thresholds are tuned in-slice (there is no dev/test split), and the real-slice VAP runs use the generic
checkpoint shipped with the VAP repository.

### 3. Full-duplex behavioural study

Five full-duplex agents join 120 MPTB floor events as an extra party; each scenario runs three times
(1,800 trials):
- **START:** does the agent start after a genuine end of turn, but not during a hold pause?
- **STOP:** does it stop when interrupted, but keep talking through a backchannel?

```bash
pip install -r requirements/fd.txt                          # + requirements/generation.txt (Kokoro) for step 2
python scripts/fd/build_scenarios.py --out outputs/fd/scenarios.json           # deterministic (seed 20260929)
python scripts/fd/build_audio_packs.py --scenarios outputs/fd/scenarios.json --out outputs/fd/audio
export OPENAI_API_KEY=... GEMINI_API_KEY=... STEPFUN_API_KEY=...             # API agents are billed
python scripts/fd/run_experiment.py --models gpt-live-1 gemini-3.8-live step-audio-3-realtime --runs 3 --concurrency 6
pip install -r requirements/fd_local.txt                                       # local agents, one CUDA GPU
python scripts/fd/run_experiment.py --models minicpm-o-4.5 realtime-venus --runs 3
python scripts/fd/run_offline_detector.py --models gemini-3.8-live step-audio-3-realtime minicpm-o-4.5 realtime-venus
python scripts/fd/analyze.py --runs outputs/fd/runs --offline outputs/fd/offline.json --out outputs/fd/analysis
```

Local checkpoints default to `openbmb/MiniCPM-o-4_5@503e754` and `inclusionAI/Realtime-Venus@ad42af5`; you
can override them with `$MPTB_MINICPM_O_DIR` / `$MPTB_VENUS_DIR`. The local agents decode by sampling, so
repeated runs differ.

### 4. Data generation

```bash
pip install -r requirements/generation.txt                 # Python 3.10, CUDA GPU
export OPENAI_API_KEY=...                                   # Stage 1 (gpt-4.1); optional OPENAI_BASE_URL
python scripts/generation/sample_manifest.py --out corpus/v1                       # 48 dev + 192 test specs (seed 23)
python scripts/generation/generate_corpus.py --out corpus/v1 --musan /data/musan \
    --noise-seeds $MPTB_ROOT/metadata/noise_seeds.json                            # Stages 1-4, checkpointed
python scripts/generation/regenerate_conversations.py --out corpus/v1 --musan /data/musan --cids test_00176 --seed-bump 777
python scripts/generation/check_corpus.py --out corpus/v1                          # QC + oracle/balance/split checks
python scripts/generation/package_release.py --corpus corpus/v1 --out release/MultiPartyTurnBench --viewer
```

The pipeline has five stages:
1. LLM script generation;
2. floor-state orchestration, which produces the gold log;
3. Kokoro-82M synthesis (v0.9.4, HF revision `f3ff357`);
4. channel assembly and the noise scene (MUSAN, OpenSLR 17);
5. QC and packaging.

Stages 2–4 are deterministic given the specification seed and the noise seed. Stage 1 depends on the LLM.
One conversation (`test_00176`) was regenerated with a bumped seed after QC.

---

## Repository layout

```
mptb/
  data.py                 dataset access (released layout)
  eval/                   scorer: schema, anchors, matching, operating-point rule (stdlib only)
  systems/                offline baselines + harness (inference, dev selection, test scoring)
  realdata/               TurnBench dev and AMI slices
  fd/                     full-duplex behavioural study (scenarios, audio packs, agents, protocols, analysis)
  generation/             data-generation pipeline (Stages 1-5, QC, statistics)
scripts/
  evaluate.py             score / select for any submission
  run_system.py           run one offline baseline
  realdata/ fd/ generation/   command-line entry points per part
configs/                  voice pool and Kokoro voice binding
requirements/             pinned environments per part
tests/                    unit tests (synthetic fixtures; no data needed)
```

## Code reference

### Benchmark access and scoring

| File | Public API |
|---|---|
| `mptb/data.py` | `Dataset(root)`: `ids(split)`, `split_of(cid)`, `split_map`, `gold(cid)` (validated event log), `golds(split)`, `render(cid)`, `script(cid)`, `audio_path(cid, speaker, kind)`, `audio(cid, kind)` (per-speaker float32 channels, 24 kHz); `default_root()` (`$MPTB_ROOT`). |
| `mptb/eval/schema.py` | Vocabularies (`CONVERSATION_TYPES`, `EVENT_LABELS`, `RESOLVED_AS`, `FLOOR_OUTCOMES`, `VOCALIZE_SUBLABELS`); `GoldEvent`, `GoldConversation` (build gold logs); `validate_gold`, `validate_submission` (lists of errors); `assert_valid_gold`, `assert_valid_submission`; `load_gold`, `load_submission`, `load_json`, `save_json`; `speaker_key`. |
| `mptb/eval/anchors.py` | `derive_anchors(gold)` → per speaker `SpeakerAnchors` with EOT/INT `TrackAnchors` (positives, negative spans, excluded spans). |
| `mptb/eval/scorer.py` | `TAU_PRE`, `TAU_MAX`, `DEV_FPR_BUDGET`, `TEST_FPR_CEILING`; `TrackResult` (counts, recall, FPR, latencies); `percentile`; `score_track` (matching rule); `score_conversation`; `score_dataset` (pooled + by N + by type); `select_operating_point` (dev rule over several submissions). |
| `scripts/evaluate.py` | CLI: `score` and `select` (see [Evaluating a system](#evaluating-a-system)). |

### Offline baselines (`mptb/systems`)

| File | Public API |
|---|---|
| `__init__.py` | `SYSTEMS`: registry of the 15 system ids (family, tracks, tie-break rule, INT kind, model factory). |
| `common.py` | `refractory` (2 s commit filter), `rising_edges`, `resample_16k`, `resample`, `equal_length_channels`, `webrtc_masks` (20 ms WebRTC VAD), `rest_mix` (mean of the other channels), `rest_active` (other-speaker-active gate). |
| `harness.py` | `GridSystem`, `ScoreSystem` (system descriptions); `compute` (inference to `raw/` or `scores/`); `assemble` (dev selection, test scoring, `report.json`, `predictions/`); `decode` (commit rule on scores); `candidates` (grid + dev quantiles); `score_sweep`; `select_grid_point`, `select_score_point` (operating-point rules); `select_dualturn_mapping` (dev-only head choice); `save_scores`, `load_scores`, `submission`. |
| `vad.py` | `THETAS`; `masks_to_predictions` (voiced masks → EOT/INT commits); `rms_db`, `energy_masks`, `silero_masks`; `VADBaseline(backend).predict`. |
| `smart_turn.py` | `load_model`, `predict_probs`, `eot_candidates`; `SmartTurnEOT` (energy-VAD candidates scored by Smart Turn); `SmartTurnINT` (TurnBench streaming pipeline, 1 − P(complete)). |
| `kyutai.py` | `load_model`, `vad_series` (streamed P(end of turn)), `eot_commits`; `KyutaiEOT`; `KyutaiINT` (batched streaming, inverted score). |
| `vap.py` | `load_vap`, `step_probs` (causal 20 s / 5 s sliding window); `VAP.predict` (s vs rest decoding). |
| `muvap.py` | `rest_gate` (WebRTC gate per model frame); `MuVAP` (`p_now`, `predict`). |
| `mimi_ep.py` | `frame_end_times`; `MimiEndpointer.scores`. |
| `dualturn.py` | `DualTurn.scores` (all heads); `EOT_CANDIDATES`, `INT_CANDIDATES`; `mapping_view` (heads → EOT/INT). |
| `turn1mini.py` | `Turn1Mini.scores` (released single-channel head, 160 ms commit steps). |
| `wavlm_anchor.py` | `frame_times`; `WavLMAnchor` (`probs_for_signal`, `scores`). |
| `espnet_tt.py` | `ESPnetTurnTakingMix` (mono mix + energy attribution), `ESPnetTurnTakingPerChannel`. |
| `scripts/run_system.py` | CLI: inference (`--shard/--nshards`) and `--assemble`. |

### Real-audio slices (`mptb/realdata`, `scripts/realdata`)

| File | Public API |
|---|---|
| `mptb/realdata/turnbench.py` | Constants (`HF_DATASET`, `SLICE_SIZE`, `AUDIO_SR`, `CANONICAL_MAP`, `CONSENSUS_TOL`); `to_canonical`; `RealEvent`, `ConsensusEvent`; `iter_dev_conversations`, `iter_dev_audio`; `build_consensus` (2-of-3 annotators, ±200 ms); `derive_real_anchors`; `anchors_for_conversation`; `anchors_to_json`, `anchors_from_json`. |
| `mptb/realdata/ami.py` | Floor-state thresholds (`PAUSE_MIN`, `BC_MAX_DUR`, `BC_MAX_WORDS`, `FLOOR_MIN`, `FLOOR_TAKE_WIN`, `OVERLAP_EPS`); `AMI_SPK`, `SLICE_MEETINGS`, `ANNOTATIONS_URL`, `AUDIO_URL`; `headset_files`; `build_ami_gold` (gold log from AMI segments and words). |
| `mptb/realdata/slices.py` | `Backend`, `BACKENDS` (slice grids); `VAP_SLICE_CHECKPOINT`; `make_predictor`; `load_turnbench_slice`, `load_ami_slice`; `select_point`; `run_backend`. |
| `scripts/realdata/prepare_turnbench.py` | TurnBench parquet → consensus anchors + 16 kHz audio of the slice. |
| `scripts/realdata/build_ami_slice.py` | AMI annotations + headset audio → gold logs and manifest (schema and oracle checks). |
| `scripts/realdata/run_slice.py` | Runs baselines on a slice and writes its report. |

### Full-duplex behavioural study (`mptb/fd`, `scripts/fd`)

| File | Public API |
|---|---|
| `mptb/fd/scenarios.py` | `iter_conversations`, `clip_for_event`, `candidates`, `pick_balanced`, `build_scenarios` (120 scenarios: 4 families × 30, N ∈ {2,3,4}); selection constants and `EXCLUDED_CONVERSATIONS`. |
| `mptb/fd/audio_packs.py` | `PROMPTS`; `pick_prompt_voice`; `peak_norm`, `mix`, `cut`, `silence`; `build_start_pack`, `build_stop_pack`, `build_pack_audio`; `write_packs`, `load_packs`. |
| `mptb/fd/protocols.py` | Probe constants (`SYSTEM_PROMPT`, windows, delays); `TrialResult`; `load_pack_audio`; `Streamer` (20 ms pacing); `run_start_trial`, `score_start`, `run_stop_trial`, `score_stop`, `run_trial`. |
| `mptb/fd/runner.py` | `run_model` (resumable concurrent trials → JSONL); `select_scenarios`, `load_done`, `failed_record`, `scenario_fields`. |
| `mptb/fd/offline.py` | Offline-detector pass of the agents: `EOT_PROMPT`, `INT_PROMPT`, `build_stimulus`, `detect_once`, `hit`, `collect`, `load_raw`, `score_detector`, `negative_span_rel`, `cut_padded`. |
| `mptb/fd/analysis.py` | `load_runs`; `valid`, `fired`, `bargein_stop`; `rate`, `mean`, `percentile`, `spearman`; `model_metrics` (coverage, accuracy, latency, barge-in, bACC, bias); `bacc_bootstrap` (B = 10,000, seed 20261012); `contingency`, `association`; `write_trials_csv`, `analyze`. |
| `mptb/fd/adapters/base.py` | `FDAdapter` (common agent interface), `VirtualPlayout`, `ExperimentClock`, `AudioChunk`, `ProviderEvent`, `SessionConfig`; `api_key`, `float_to_pcm16`, `pcm16_to_float`, `resample`, `ws_connect`. |
| `mptb/fd/adapters/local_base.py` | `StreamClock`, `LocalChunkResult`, `cached_model`, `resolve_checkpoint`, `read_audio_16k`, `load_duplex_model`, `LocalCausalChunkAdapter`. |
| `mptb/fd/adapters/{openai_live,gemini_live,step_audio,minicpm_o,venus}.py` | `GptLive1Adapter`, `GeminiLiveAdapter`, `StepAudio3RealtimeAdapter`, `MiniCPMOAdapter`, `RealtimeVenusAdapter`. |
| `mptb/fd/adapters/__init__.py` | `ADAPTERS`, `available`, `get_adapter`. |
| `scripts/fd/*.py` | `build_scenarios.py`, `build_audio_packs.py`, `run_experiment.py`, `run_offline_detector.py` (`--score-only`), `analyze.py`. |

### Data generation (`mptb/generation`, `scripts/generation`)

| File | Public API |
|---|---|
| `strata.py` | `DOMAINS`, `AMBIENTS`, `DISCRETE_EVENTS`, `TARGET_DURATION_RANGE_S`, `TABLE_III` (TurnBench soft targets); `ConvSpec`; `compute_targets`; `load_voice_pool`; `sample_stratified`; `save_manifest`, `load_manifest`. |
| `split.py` | `partition_voices` (gender-balanced, disjoint dev/test voice pools). |
| `llm.py` | `ChatClient` (OpenAI-compatible; `chat`, `chat_json`; retries, JSON mode); `extract_json`. |
| `stage1_script.py` | `SYSTEM_PROMPT`, `TYPE_STYLE`, `TONES`, `MAX_BACKCHANNEL_WORDS`; `build_user_prompt`; `validate_script`; `normalize_script`; `generate_script`; `save_script`. |
| `stage2_orchestrate.py` | `TimingConfig`; `Clip`; `OrchestrationResult`; `orchestrate` (floor-state machine); `render_record`; `build_gold`. |
| `tts.py` | `KOKORO_REVISION`; `TurnAudio`; `KokoroTTS` (`synth`, `duration_fn`, `kokoro_voice`). |
| `assemble.py` | `assemble_channels` (per-speaker channels, truncation fade, peak normalisation); `write_channels`. |
| `noise.py` | `noise_seed`; `synth_ambient`; `index_musan`; `load_clip`; `rms`, `active_speech_rms`, `scale_to_snr`, `peak_limit`; `build_ambient_bed`, `build_discrete_events`, `build_background_speech`; `mix_noise_profile`; `add_noise`. |
| `qc.py` | `qc_conversation` (acoustic QC); `summarize`. |
| `pipeline.py` | `voice_map`; `gold_metadata`; `generate_conversation`; `generate_corpus` (checkpointed Stages 1–4). |
| `checks.py` | `perfect_submission`; `check_oracle_round_trip`, `check_qc`, `check_balance`, `check_split`; `check_corpus`. |
| `stats.py` | `overlap_seconds`, `conv_stats`, `aggregate`, `dataset_stats` (`metadata/dataset_stats.json`). |
| `viewer.html` | Offline dataset viewer (copied by the packager). |
| `scripts/generation/*.py` | `sample_manifest.py`, `generate_corpus.py`, `regenerate_conversations.py`, `check_corpus.py`, `package_release.py`. |
| `configs/*.json` | 24 abstract voices (`voice_pool.json`) and their Kokoro binding (`voice_binding_kokoro.json`). |

## Tests

```bash
pip install pytest numpy
pytest            # 77 tests on synthetic fixtures (scorer, baseline harness, FD, real-data, generation)
```

## License

Code: MIT (see `LICENSE`). The dataset has its own license (CC BY 4.0, see the dataset card). Third-party
models and corpora (Kokoro-82M, MUSAN, the baselines' checkpoints, TurnBench, AMI) keep their licenses.
