"""DualTurn (``anyreach-ai/dualturn-qwen2.5-mimi-0.5B``) (paper Table 3).

A dual-channel model (frozen Mimi codec + Qwen2.5-0.5B with LoRA and per-channel heads), run as
"s vs rest" (channel 0 = s, channel 1 = mean of the other speakers) at 12.5 Hz. All per-channel heads are
saved; the EOT and INT scores are chosen on the *dev* split only (:data:`MAPPING_CANDIDATES`,
:func:`mptb.systems.harness.select_dualturn_mapping`). The paper's mapping is EOT = ``eot_probs`` (raw)
and INT = ``hold_probs`` (a floor-holding signal; INT is derived, gated by other-speaker activity).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .common import rest_mix
from .mimi_ep import frame_end_times

HF_REPO = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"
FRAME_HZ = 12.5
HEADS = ("vad_probs", "eot_probs", "bot_probs", "hold_probs", "bc_probs")
# (head, inverted) candidates for EOT and head candidates for INT, compared on dev only.
EOT_CANDIDATES = (("eot_probs", False), ("eot_probs", True))
INT_CANDIDATES = ("bot_probs", "hold_probs")


class DualTurn:
    sr = 24000

    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True).eval().cuda()

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        torch = self.torch
        out = {}
        with torch.no_grad():
            for s in ch:
                wav = torch.from_numpy(np.stack([ch[s], rest_mix(ch, s)])).cuda()
                o = self.model(wav, sr=sr)
                d = {}
                for h in HEADS:
                    v = getattr(o, h)[0].float().cpu().numpy()
                    d[h] = v[:, 0]
                    d[h + "_rest"] = v[:, 1]
                out[s] = {"t": frame_end_times(len(d["eot_probs"]), FRAME_HZ), **d}
        return out


def mapping_view(eot_head: str, eot_inverted: bool, int_head: str) -> Callable[[dict], dict]:
    """Return a transform that sets ``eot``/``int`` scores of every speaker from the chosen heads."""
    def view(per_spk: dict) -> dict:
        for d in per_spk.values():
            d["eot"] = 1.0 - d[eot_head] if eot_inverted else d[eot_head]
            d["int"] = d[int_head]
        return per_spk
    return view

