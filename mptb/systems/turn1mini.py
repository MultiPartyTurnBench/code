"""turn-1-mini (``p99lab/turn-1-mini``, package ``p99turn`` 1.2.3) (paper Table 3, EOT only).

Only the single-channel end-of-turn head is released (the leaderboard system's interruption and
cross-channel heads are not public), so this system has no INT output. Each channel is streamed causally
with the package's benchmarked settings; values are produced every 20 ms but become available at the end
of each 160 ms streaming step, which is their commit time.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


class Turn1Mini:
    sr = 16000

    def __init__(self):
        from p99turn import Turn1Mini as _T1
        self.model = _T1(device="cuda")

    def scores(self, ch: Dict[str, np.ndarray], sr: int) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for s, x in ch.items():
            p = self.model.stream().push(x)
            k = np.arange(len(p))
            t = (k // 8 + 1) * 0.16
            out[s] = {"t": t, "eot": p.astype(np.float32)}
        return out
