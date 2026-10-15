"""Smart Turn v2 (``pipecat-ai/smart-turn-v2``): semantic end-of-turn classifier (paper Table 3).

Smart Turn is a wav2vec2 model that predicts, from the audio of an utterance, the probability that the
utterance is *complete*. Two pipelines are used:

* **EOT** (:class:`SmartTurnEOT`): an energy VAD proposes candidate boundaries (speech-to-silence
  offsets after a 0.3 s hangover); Smart Turn scores the last <= 8 s of audio ending at each candidate,
  and a candidate commits at threshold theta if P(complete) >= theta.
* **INT by inversion** (:class:`SmartTurnINT`): TurnBench's Smart Turn streaming pipeline (Silero VAD on
  512-sample chunks, accumulate the utterance, re-score while it settles); the per-chunk value is
  P(complete), and INT = 1 - P(complete) on the interrupter's channel (TurnBench protocol). This is an
  inverted end-of-turn score, not an interruption detector.

The v2 checkpoint is wav2vec2-base + attention pooling + an MLP head (``Wav2Vec2ForEndpointing``); the
architecture is reproduced here to load the released weights (pipecat-ai/smart-turn, BSD-2-Clause).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .common import refractory, resample_16k

HF_REPO = "pipecat-ai/smart-turn-v2"
SR = 16000
MAX_WIN_S = 8.0
EOT_THETAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def load_model(model_dir: str, device: str = "cuda"):
    """Load Smart Turn v2 from a local snapshot directory. Returns ``(model, processor, device)``."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

    class Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            h = config.hidden_size
            self.pool_attention = nn.Sequential(nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, 1))
            self.classifier = nn.Sequential(
                nn.Linear(h, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
            self.post_init()

        def forward(self, input_values, attention_mask=None):
            hidden = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
            w = torch.softmax(self.pool_attention(hidden), dim=1)
            pooled = (hidden * w).sum(dim=1)
            return self.classifier(pooled)

    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = Wav2Vec2ForEndpointing.from_pretrained(model_dir).to(dev).eval()
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    return model, processor, dev


def predict_probs(model, processor, dev, windows: List[np.ndarray], batch_size: int = 16) -> List[float]:
    """P(complete) for each 16 kHz mono window (each truncated to its last <= 8 s)."""
    import torch
    probs: List[float] = []
    trunc = [w[-int(MAX_WIN_S * SR):] if len(w) > MAX_WIN_S * SR else w for w in windows]
    for i in range(0, len(trunc), batch_size):
        chunk = trunc[i:i + batch_size]
        inputs = processor([np.asarray(w, dtype=np.float32) for w in chunk],
                           sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(dev),
                           attention_mask=inputs.get("attention_mask").to(dev)
                           if inputs.get("attention_mask") is not None else None)
            probs.extend(torch.sigmoid(logits).squeeze(-1).cpu().tolist())
    return probs


def eot_candidates(audio16k: np.ndarray, vad_db: float = -40.0, frame_ms: float = 20.0,
                   hang_s: float = 0.3) -> List[float]:
    """Energy-VAD speech-to-silence offsets (after a hangover): candidate turn ends in seconds."""
    hop = max(1, int(SR * frame_ms / 1000.0))
    n = len(audio16k) // hop
    if n == 0:
        return []
    frames = audio16k[:n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1) + 1e-12) + 1e-12)
    voiced = db > vad_db
    hang = max(1, int(hang_s * 1000.0 / frame_ms))
    cands = []
    for i in range(1, n):
        if voiced[i - 1] and not voiced[i]:
            j = min(n, i + hang)
            if not voiced[i:j].any():
                cands.append(i * hop / SR)
    return cands


def _snapshot(model_dir: str = None) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPO)


class SmartTurnEOT:
    """EOT-only pipeline: energy-VAD candidates scored by Smart Turn, thresholded per theta."""

    thetas = EOT_THETAS

    def __init__(self, model_dir: str = None, device: str = "cuda", vad_db: float = -40.0):
        self.model, self.processor, self.dev = load_model(_snapshot(model_dir), device)
        self.vad_db = vad_db

    def scored_candidates(self, audio: Dict[str, np.ndarray], sr: int) -> Dict[str, List[Tuple[float, float]]]:
        """``{speaker: [(candidate_time_s, p_complete), ...]}`` for one conversation."""
        conv = {}
        for s, a in audio.items():
            a16 = resample_16k(a, sr)
            cands = eot_candidates(a16, vad_db=self.vad_db)
            w = int(MAX_WIN_S * SR)
            windows = [a16[max(0, int(t * SR) - w): max(1, int(t * SR))] for t in cands]
            probs = predict_probs(self.model, self.processor, self.dev, windows) if windows else []
            conv[s] = list(zip(cands, probs))
        return conv

    def predict(self, audio: Dict[str, np.ndarray], sr: int) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
        """``{theta: {speaker: {"eot": commits, "interruption": []}}}`` over :data:`EOT_THETAS`."""
        conv = self.scored_candidates(audio, sr)
        return {float(th): {s: {"eot": refractory([t for (t, p) in tp if p >= th]), "interruption": []}
                            for s, tp in conv.items()}
                for th in self.thetas}


class SmartTurnINT:
    """TurnBench's Smart Turn streaming pipeline with the v2 model; INT = 1 - P(complete)."""

    sr = SR
    CHUNK = 512

    def __init__(self, model_dir: str = None, device: str = "cuda"):
        import torch
        from silero_vad import load_silero_vad
        self.torch = torch
        self.m, self.proc, self.dev = load_model(_snapshot(model_dir), device)
        self.vad = load_silero_vad()
        ms = self.CHUNK / SR * 1000
        self.PRE, self.STOP = math.ceil(200 / ms), math.ceil(1000 / ms)
        self.SETTLE, self.MAXC, self.STRIDE = math.ceil(2000 / ms), math.ceil(8000 / ms), 8

    def _p(self, seg) -> float:
        return float(predict_probs(self.m, self.proc, self.dev, [np.concatenate(seg)])[0])

    def channel(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-chunk P(complete) of one 16 kHz channel, stamped at the end of each 512-sample chunk."""
        torch = self.torch
        self.vad.reset_states()
        C = self.CHUNK
        pre, seg = deque(maxlen=self.PRE), []
        active, sil, extra, cur = False, 0, 0, 0.0
        t, v = [], []
        for i in range(0, len(wav) - C + 1, C):
            c = wav[i:i + C].astype(np.float32)
            sp = float(self.vad(torch.from_numpy(c), SR).item()) > 0.5
            if not active:
                pre.append(c)
                if sp:
                    cur, seg, active, sil, extra = 0.0, list(pre) + [c], True, 0, 0
            elif extra == 0:
                seg.append(c)
                if len(seg) > self.MAXC:
                    seg.pop(0)
                sil = 0 if sp else sil + 1
                if sil >= self.STOP:
                    extra = 1
                    cur = self._p(seg)
            else:
                if sp:
                    cur, seg, active, sil, extra = 0.0, [c], True, 0, 0
                else:
                    seg.append(c)
                    extra += 1
                    if extra % self.STRIDE == 1:
                        cur = self._p(seg)
                    if extra >= self.SETTLE:
                        active, sil, extra = False, 0, 0
                        pre.clear()
            t.append((i + C) / SR)
            v.append(cur)
        return np.asarray(t), np.asarray(v, np.float32)

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for s, x in ch.items():
            t, v = self.channel(x)
            out[s] = {"t": t, "eot": v, "int": 1.0 - v}
        return out
