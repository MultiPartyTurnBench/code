"""Tests for assembly, Stage-4 noise, QC and the end-to-end pipeline with a stub TTS and fake MUSAN."""
from __future__ import annotations

import json
import random

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from mptb.generation.assemble import assemble_channels, write_channels  # noqa: E402
from mptb.generation.noise import (active_speech_rms, add_noise, index_musan, mix_noise_profile,  # noqa: E402
                                   noise_seed, peak_limit, synth_ambient)
from mptb.generation.pipeline import generate_corpus  # noqa: E402
from mptb.generation.qc import qc_conversation, summarize  # noqa: E402
from mptb.generation.strata import ConvSpec  # noqa: E402
from mptb.generation.tts import _trim_bounds  # noqa: E402

SR = 24000


class StubTTS:
    """Deterministic tone per (text, voice): 0.05 s per character, 220-440 Hz."""

    def synth(self, text, voice_id):
        n = int(0.05 * len(text) * SR)
        f = 220 + 10 * (sum(map(ord, text + voice_id)) % 22)
        audio = (0.5 * np.sin(2 * np.pi * f * np.arange(n) / SR)).astype(np.float32)
        return type("TA", (), {"audio": audio, "sample_rate": SR, "duration_s": n / SR})()

    def duration_fn(self, voice_map):
        return lambda text, spk: self.synth(text, voice_map[spk]).duration_s


@pytest.fixture()
def musan(tmp_path):
    rng = np.random.default_rng(0)
    for sub, n in (("noise/sound-bible", 3), ("speech/librivox", 2)):
        d = tmp_path / "musan" / sub
        d.mkdir(parents=True)
        for i in range(n):
            sf.write(str(d / f"clip-{i}.wav"), (0.3 * rng.standard_normal(16000 * 5)).astype(np.float32), 16000)
    return index_musan(str(tmp_path / "musan"))


def test_trim_bounds():
    x = np.zeros(SR, np.float32)
    x[10000:12000] = 0.5
    s, e = _trim_bounds(x, SR)
    assert s == 41 * 240 - 720 and e == 50 * 240 + 720  # 10 ms frames, 30 ms margin


def test_ambient_is_deterministic_and_unit_peak():
    a = synth_ambient("cafe", 4800, SR, random.Random(1))
    assert np.array_equal(a, synth_ambient("cafe", 4800, SR, random.Random(1)))
    assert abs(float(np.max(np.abs(a))) - 1.0) < 1e-6


def test_noise_mix_shared_seeded_and_limited(musan):
    clean = {"speaker_1": np.zeros(SR * 6, np.float32), "speaker_2": np.zeros(SR * 6, np.float32)}
    clean["speaker_1"][SR:2 * SR] = 0.8
    clean["speaker_2"][3 * SR:4 * SR] = -0.8
    prof = {"ambient": "office_hum", "snr_db": 10.0, "discrete_events": ["door_knock", "phone_ring"],
            "background_speech": True}
    noisy, man = add_noise(clean, prof, musan, SR, seed=5)
    noisy2, man2 = add_noise(clean, prof, musan, SR, seed=5)
    assert man == man2 and all(np.array_equal(noisy[s], noisy2[s]) for s in clean)
    assert list(man)[:2] == ["shared", "signal_rms"] and man["ambient_snr_db"] == 20.0
    assert [e["label"] for e in man["discrete_events"]] == ["door_knock", "phone_ring"]
    assert 2 <= len(man["background_speech_starts_s"]) <= 4
    raw, _ = mix_noise_profile(clean, prof, musan, SR, random.Random(5))
    assert np.allclose(raw["speaker_1"] - clean["speaker_1"], raw["speaker_2"] - clean["speaker_2"], atol=1e-6)
    assert max(float(np.max(np.abs(b))) for b in noisy.values()) <= 10 ** (-1 / 20) + 1e-6
    assert abs(active_speech_rms(clean) - 0.8) < 1e-6
    assert peak_limit({"a": np.full(4, 0.5, np.float32)})["a"][0] == 0.5
    assert noise_seed("test_00001", 7) == noise_seed("test_00001", 7) != noise_seed("test_00002", 7)


def _spec():
    voices = {"speaker_1": {"voice_id": "v01", "gender": "F", "style": "x"},
              "speaker_2": {"voice_id": "v03", "gender": "M", "style": "y"}}
    return ConvSpec("dev_00001", 2, ["speaker_1", "speaker_2"], "book club", "casual", 150.0, voices,
                    {"ambient": "cafe", "snr_db": 30.0, "discrete_events": ["door_knock"],
                     "background_speech": False}, 11)


SCRIPT = {"conversation_id": "dev_00001", "num_speakers": 2, "speakers": ["speaker_1", "speaker_2"],
          "conversation_type": "casual", "tone": "casual and friendly", "generator_model": "stub",
          "turns": [{"speaker": "speaker_1", "text": "hello there " * 8, "event_type": "TURN", "timing_hint": "after_pause"},
                    {"speaker": "speaker_2", "text": "mm-hm", "event_type": "VOCALIZE_NO_CLAIM",
                     "sublabel": "BACKCHANNEL", "target": "speaker_1", "timing_hint": "during_holder"},
                    {"speaker": "speaker_2", "text": "my turn now " * 6, "event_type": "TURN", "timing_hint": "latched"},
                    {"speaker": "speaker_1", "text": "wait a second", "event_type": "CLAIM_FLOOR",
                     "resolved_hint": "INTERRUPTION", "target": "speaker_2", "timing_hint": "overlap_onset"},
                    {"speaker": "speaker_1", "text": "and so on " * 6, "event_type": "TURN", "timing_hint": "latched"}]}


def test_pipeline_end_to_end_with_stub_tts(tmp_path, musan):
    out = tmp_path / "corpus"
    (out / "scripts").mkdir(parents=True)
    (out / "scripts" / "dev_00001.json").write_text(json.dumps(SCRIPT))
    assert generate_corpus([_spec()], StubTTS(), None, str(out), musan, verbose=False) == ["dev_00001"]
    gold = json.loads((out / "gold" / "dev_00001.json").read_text())
    render = json.loads((out / "render" / "dev_00001.json").read_text())
    assert render["noisy_files"] == gold["audio"]["files"] and "noise" in render
    assert json.loads((out / "noise_seeds.json").read_text()) == {"dev_00001": noise_seed("dev_00001", 11)}
    for kind in ("clean", "noisy"):
        for f in gold["audio"]["files"].values():
            x, sr = sf.read(str(out / "audio" / kind / f))
            assert sr == SR and abs(len(x) - gold["duration_s"] * SR) < 0.001 * SR
    r = qc_conversation(gold, render, str(out / "audio" / "clean"))
    assert r["passed"], r["fails"]
    assert summarize([r])["n_passed"] == 1
    before = (out / "audio" / "noisy" / "dev_00001_speaker_1.flac").read_bytes()
    generate_corpus([_spec()], StubTTS(), None, str(out), musan, overwrite=True, verbose=False)
    assert (out / "audio" / "noisy" / "dev_00001_speaker_1.flac").read_bytes() == before


def test_assemble_truncates_with_fade(tmp_path):
    from mptb.generation.stage2_orchestrate import Clip
    tts = StubTTS()
    clips = [Clip("c000", "speaker_1", "x" * 40, 0.3, 1.0, "holder", truncated=True)]
    buf = assemble_channels(clips, {"speaker_1": "v01"}, tts, ["speaker_1"], 3.0)
    x = buf["speaker_1"]
    assert len(x) == 3 * SR and abs(float(np.max(np.abs(x))) - 10 ** (-1 / 20)) < 1e-6
    assert np.all(x[int(1.0 * SR) + 1:] == 0) and abs(x[int(1.0 * SR) - 1]) < 1e-3
    assert write_channels(buf, str(tmp_path), "c") == {"speaker_1": "c_speaker_1.flac"}
