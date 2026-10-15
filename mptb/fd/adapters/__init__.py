"""Provider adapters for the five full-duplex agents of the paper (Table 4).

=========================  ==========================================  =======================================
name                       agent                                       access
=========================  ==========================================  =======================================
``gpt-live-1``             GPT-Live-1                                  OpenAI Live session API (``OPENAI_API_KEY``)
``gemini-3.8-live``        Gemini-3.8-Live                             Gemini Live WebSocket (``GEMINI_API_KEY``)
``step-audio-3-realtime``  Step-Audio3-Realtime                        StepFun realtime API (``STEPFUN_API_KEY``)
``minicpm-o-4.5``          MiniCPM-o-4.5                               local GPU, Hugging Face checkpoint
``realtime-venus``         Realtime-Venus                              local GPU, Hugging Face checkpoint
=========================  ==========================================  =======================================

Adapters are imported lazily (local ones pull in torch/transformers) through :func:`get_adapter`.
"""
from __future__ import annotations

import importlib
from typing import List, Type

from .base import FDAdapter

ADAPTERS = {
    "gpt-live-1": "openai_live:GptLive1Adapter",
    "gemini-3.8-live": "gemini_live:GeminiLiveAdapter",
    "step-audio-3-realtime": "step_audio:StepAudio3RealtimeAdapter",
    "minicpm-o-4.5": "minicpm_o:MiniCPMOAdapter",
    "realtime-venus": "venus:RealtimeVenusAdapter",
}


def available() -> List[str]:
    """Names of all adapters."""
    return sorted(ADAPTERS)


def get_adapter(name: str) -> Type[FDAdapter]:
    """Adapter class registered under ``name`` (imports its module on first use)."""
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; available: {available()}")
    module, cls = ADAPTERS[name].split(":")
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)
