"""OpenAI-compatible chat client used by Stage 1 (script generation).

Credentials come only from the environment: ``OPENAI_API_KEY`` (required) and, optionally,
``OPENAI_BASE_URL`` for any OpenAI-compatible endpoint. The paper's scripts were generated
with ``gpt-4.1``. Features:

* retries with exponential backoff (1, 2, 4, 8, 16 s; capped at 30 s);
* optional strict JSON mode (``response_format={"type": "json_object"}``) and robust parsing;
* optional content-hash disk cache so identical requests are answered locally.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "gpt-4.1"


class ChatClient:
    """Minimal chat-completions client with retry, JSON mode and an optional disk cache.

    ``client`` may be any object exposing ``chat.completions.create(**kwargs)`` (an
    ``openai.OpenAI`` instance is created from the environment when omitted).
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, cache_dir: Optional[str] = None,
                 max_retries: int = 5, timeout: float = 180.0, client: Any = None):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        if client is None:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("No API key: set the OPENAI_API_KEY environment variable")
            from openai import OpenAI
            base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        f = self.cache_dir / f"{key}.txt"
        return f.read_text() if f.exists() else None

    def _cache_put(self, key: str, value: str) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.txt").write_text(value)

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.8,
             max_tokens: int = 4096, json_mode: bool = False,
             model: Optional[str] = None, use_cache: bool = True,
             seed: Optional[int] = None) -> str:
        """Send one chat-completions request and return the message content.

        ``seed`` is forwarded to the API when given; ``json_mode`` requests a JSON object.
        Retries up to ``max_retries`` times on any error, then raises ``RuntimeError``.
        """
        mdl = model or self.model
        payload = {"model": mdl, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "json_mode": json_mode, "seed": seed}
        ck = self._cache_key(payload)
        if use_cache:
            hit = self._cache_get(ck)
            if hit is not None:
                return hit

        kwargs: Dict[str, Any] = {"model": mdl, "messages": messages,
                                  "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                if use_cache:
                    self._cache_put(ck, content)
                return content
            except Exception as e:  # network, rate-limit and server errors are all retried
                last_err = e
                time.sleep(min(2.0 ** attempt, 30.0))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def chat_json(self, messages: List[Dict[str, str]], **kw: Any) -> Any:
        """:meth:`chat` with JSON mode on by default, parsed with :func:`extract_json`."""
        kw.setdefault("json_mode", True)
        return extract_json(self.chat(messages, **kw))


def extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating Markdown code fences or surrounding prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            return json.loads(text[i:j + 1])
    raise ValueError(f"Could not parse JSON from response:\n{text[:500]}")
