"""Offline baselines of the paper (Table 3) and the registry used by ``scripts/run_system.py``.

Every system reads the noisy per-speaker channels of the released dataset and is evaluated with the same
commit rule, operating-point rule and scorer. See :mod:`mptb.systems.harness` for the two system
families. ``int_kind`` records how a system's INT score is obtained:

* ``native``   - a dedicated interruption output (ESPnet's Interruption class);
* ``derived``  - a floor-taking/holding projection gated by other-speaker activity (VAP, MuVAP, Mimi-EP,
  DualTurn) or a class-based proxy (WavLM anchor: 1 - P(continue));
* ``inverted`` - 1 - P(end of turn) of an end-of-turn model (Kyutai, Smart Turn; TurnBench protocol);
* ``acoustic`` - a voice-activity onset while another channel is voiced (VADs);
* ``none``     - EOT only.

The paper reports Smart Turn v2 and Kyutai with their EOT row from the grid pipeline (``smart_turn``,
``kyutai``) and their INT row from the score-stream inversion (``smart_turn_int``, ``kyutai_int``).
"""
from __future__ import annotations

from typing import Dict, Union

from .harness import GridSystem, ScoreSystem


def _vad(backend):
    def make():
        from .vad import VADBaseline
        return VADBaseline(backend)
    return make


def _smart_turn_eot():
    from .smart_turn import SmartTurnEOT
    return SmartTurnEOT()


def _smart_turn_int():
    from .smart_turn import SmartTurnINT
    return SmartTurnINT()


def _kyutai_eot():
    from .kyutai import KyutaiEOT
    return KyutaiEOT()


def _kyutai_int():
    from .kyutai import KyutaiINT
    return KyutaiINT()


def _vap():
    from .vap import VAP
    return VAP()


def _muvap():
    from .muvap import MuVAP
    return MuVAP()


def _mimi_ep():
    from .mimi_ep import MimiEndpointer
    return MimiEndpointer()


def _dualturn():
    from .dualturn import DualTurn
    return DualTurn()


def _turn1mini():
    from .turn1mini import Turn1Mini
    return Turn1Mini()


def _wavlm():
    from .wavlm_anchor import WavLMAnchor
    return WavLMAnchor()


def _espnet_mix():
    from .espnet_tt import ESPnetTurnTakingMix
    return ESPnetTurnTakingMix()


def _espnet_perchannel():
    from .espnet_tt import ESPnetTurnTakingPerChannel
    return ESPnetTurnTakingPerChannel()


SYSTEMS: Dict[str, Union[GridSystem, ScoreSystem]] = {s.name: s for s in (
    GridSystem("energy", "RMS energy VAD (theta = dBFS threshold)", _vad("energy"), int_kind="acoustic"),
    GridSystem("webrtc", "WebRTC VAD (theta = aggressiveness)", _vad("webrtc"), int_kind="acoustic"),
    GridSystem("silero", "Silero VAD (theta = speech probability)", _vad("silero"), int_kind="acoustic"),
    GridSystem("smart_turn", "Smart Turn v2 on energy-VAD candidates (EOT)", _smart_turn_eot, tracks=("eot",)),
    ScoreSystem("smart_turn_int", "Smart Turn v2, TurnBench streaming pipeline, INT = 1 - P(complete)",
                _smart_turn_int, tracks=("interruption",), int_kind="inverted"),
    GridSystem("kyutai", "Kyutai Semantic-VAD stt-1b-en_fr (EOT)", _kyutai_eot, tracks=("eot",)),
    ScoreSystem("kyutai_int", "Kyutai Semantic-VAD stt-1b-en_fr, INT = 1 - P(end of turn)", _kyutai_int,
                tracks=("interruption",), int_kind="inverted"),
    GridSystem("vap", "VAP, TurnBench oto checkpoint, s vs rest", _vap, tie_break="theta", int_kind="derived"),
    GridSystem("muvap", "MuVAP vap-speaker-mimi, s vs rest", _muvap, tie_break="theta", int_kind="derived"),
    ScoreSystem("mimi_ep", "Mimi Endpointer oto_d1f, s vs rest", _mimi_ep, int_kind="derived"),
    ScoreSystem("dualturn", "DualTurn Qwen2.5-Mimi 0.5B, s vs rest", _dualturn, int_kind="derived"),
    ScoreSystem("turn1mini", "turn-1-mini released single-channel head (EOT)", _turn1mini, tracks=("eot",)),
    ScoreSystem("wavlm_anchor", "WavLM-Large ANCHOR, per channel", _wavlm, int_kind="derived"),
    ScoreSystem("espnet_tt_mix", "ESPnet turn-taking (Talking Turns SWBD), mono mix with energy attribution",
                _espnet_mix, int_kind="native"),
    ScoreSystem("espnet_tt_perchannel", "ESPnet turn-taking (Talking Turns SWBD), each isolated channel",
                _espnet_perchannel, int_kind="native"),
)}
