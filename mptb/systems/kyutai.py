"""Kyutai Semantic-VAD (``kyutai/stt-1b-en_fr-candle``): semantic end-of-turn head (paper Table 3).

Kyutai's delayed-streams STT model emits auxiliary "extra head" distributions alongside its text
tokens; head 2, bucket 0 is P(the current speaker's turn has ended). Each per-speaker channel is
streamed through Mimi + the LM (as in Kyutai's ``stt_from_file_pytorch.py``) and ``pr_vad`` is read per
80 ms frame. A commit is stamped at the audio time the model has consumed when it fires (causal; the
model's internal delay is not subtracted). A trailing silence of one audio delay is fed so the head can
fire for the final turn; those frames are clamped to the true audio end.

* **EOT** (:class:`KyutaiEOT`): rising edge of ``pr_vad >= theta`` over :data:`EOT_THETAS`, 2 s refractory.
* **INT by inversion** (:class:`KyutaiINT`): all N channels streamed as one batch; INT = 1 - P(end of
  turn) on the interrupter's channel (TurnBench protocol), decoded by the score-stream harness. This is
  an inverted end-of-turn score, not an interruption detector.

Requires the ``moshi`` package with extra-heads support (the paper used 0.2.13 for EOT, 0.2.11 for INT).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .common import refractory

HF_REPO = "kyutai/stt-1b-en_fr-candle"
VAD_HEAD_INDEX = 2
VAD_BUCKET = 0
EOT_THETAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def load_model(device: str = "cuda", hf_repo: str = HF_REPO) -> dict:
    """Load Mimi + the STT LM (bf16) and a greedy ``LMGen``; returns a handle dict."""
    import moshi.models
    import torch

    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(hf_repo)
    mimi = info.get_mimi(device=device)
    lm = info.get_moshi(device=device, dtype=torch.bfloat16)
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    return {
        "mimi": mimi,
        "lm_gen": lm_gen,
        "device": device,
        "frame_rate": float(mimi.frame_rate),
        "frame_size": int(mimi.frame_size),
        "sample_rate": int(mimi.sample_rate),
        "audio_delay_s": float(info.stt_config.get("audio_delay_seconds", 0.5)),
        "silence_prefix_s": float(info.stt_config.get("audio_silence_prefix_seconds", 0.0)),
    }


def vad_series(handle: dict, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stream one mono channel; return ``(times_s, pr_vad)`` (consumed audio time, P(end of turn))."""
    import julius
    import torch

    mimi, lm_gen = handle["mimi"], handle["lm_gen"]
    device = handle["device"]
    fr, fsz, msr = handle["frame_rate"], handle["frame_size"], handle["sample_rate"]

    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(device)
    if sr != msr:
        x = julius.resample_frac(x, sr, msr)
    if x.shape[-1] % fsz != 0:
        x = torch.nn.functional.pad(x, (0, fsz - x.shape[-1] % fsz))
    audio_len_s = len(audio) / float(sr)

    n_prefix = int(round(handle["silence_prefix_s"] * fr))
    n_suffix = max(1, int(np.ceil(handle["audio_delay_s"] * fr)) + 1)
    silence = torch.zeros((1, 1, fsz), dtype=torch.float32, device=device)
    real_chunks = list(torch.split(x[None, None], fsz, dim=-1))
    chunks = [silence] * n_prefix + real_chunks + [silence] * n_suffix

    times: List[float] = []
    prv: List[float] = []
    consumed_real = 0
    with mimi.streaming(1), lm_gen.streaming(1):
        for k, chunk in enumerate(chunks):
            atok = mimi.encode(chunk)
            out = lm_gen.step_with_extra_heads(atok)
            if out is None:
                if n_prefix <= k < n_prefix + len(real_chunks):
                    consumed_real += 1
                continue
            _text, heads = out
            if n_prefix <= k < n_prefix + len(real_chunks):
                consumed_real += 1
            t = min(audio_len_s, consumed_real / fr)
            p = float(heads[VAD_HEAD_INDEX][0, 0, VAD_BUCKET].float().cpu().item()) if heads else 0.0
            times.append(t)
            prv.append(p)
    return np.asarray(times, dtype=np.float64), np.asarray(prv, dtype=np.float64)


def eot_commits(times: np.ndarray, prv: np.ndarray, theta: float) -> List[float]:
    """Rising edges of ``pr_vad >= theta`` (before the refractory filter)."""
    if len(prv) == 0:
        return []
    above = prv >= theta
    rises = np.where(above[1:] & ~above[:-1])[0] + 1
    if above[0]:
        rises = np.insert(rises, 0, 0)
    return [float(times[i]) for i in rises]


class KyutaiEOT:
    """EOT over the theta grid, one channel at a time."""

    thetas = EOT_THETAS

    def __init__(self, device: str = "cuda"):
        self.handle = load_model(device=device)

    def predict(self, audio: Dict[str, np.ndarray], sr: int) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
        per_theta = {float(t): {} for t in self.thetas}
        for spk, a in audio.items():
            times, prv = vad_series(self.handle, a, sr)
            for t in self.thetas:
                per_theta[float(t)][spk] = {"eot": refractory(eot_commits(times, prv, t)), "interruption": []}
        return per_theta


class KyutaiINT:
    """All channels of a conversation streamed as one batch; per-frame P(end of turn) and its inversion."""

    def __init__(self, device: str = "cuda"):
        import torch
        self.torch = torch
        self.h = load_model(device)
        self.sr = self.h["sample_rate"]

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        torch = self.torch
        h = self.h
        mimi, lm_gen = h["mimi"], h["lm_gen"]
        fr, fsz = h["frame_rate"], h["frame_size"]
        spk = list(ch)
        x = torch.from_numpy(np.stack([ch[s] for s in spk])).cuda()
        if x.shape[-1] % fsz:
            x = torch.nn.functional.pad(x, (0, fsz - x.shape[-1] % fsz))
        audio_len_s = len(next(iter(ch.values()))) / sr
        n_pre = int(round(h["silence_prefix_s"] * fr))
        n_suf = max(1, int(np.ceil(h["audio_delay_s"] * fr)) + 1)
        N = len(spk)
        sil = torch.zeros((N, 1, fsz), device="cuda")
        real = list(torch.split(x[:, None], fsz, dim=-1))
        chunks = [sil] * n_pre + real + [sil] * n_suf
        ts, pv, consumed = [], [], 0
        with torch.no_grad(), mimi.streaming(N), lm_gen.streaming(N):
            for k, c in enumerate(chunks):
                o = lm_gen.step_with_extra_heads(mimi.encode(c))
                if n_pre <= k < n_pre + len(real):
                    consumed += 1
                if o is None:
                    continue
                _, heads = o
                ts.append(min(audio_len_s, consumed / fr))
                pv.append(heads[VAD_HEAD_INDEX][:, 0, VAD_BUCKET].float().cpu().numpy())
        t = np.asarray(ts)
        P = np.stack(pv)
        return {s: {"t": t, "eot": P[:, i], "int": 1.0 - P[:, i]} for i, s in enumerate(spk)}
