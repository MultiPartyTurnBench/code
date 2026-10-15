"""Tests for the OpenAI-compatible Stage-1 client (mptb.generation.llm); no network access."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mptb.generation import llm as llm_mod
from mptb.generation.llm import ChatClient, extract_json


class StubCompletions:
    """Records calls to ``chat.completions.create``; fails the first ``n_fail`` calls."""

    def __init__(self, content: str, n_fail: int = 0):
        self.content, self.n_fail, self.calls = content, n_fail, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.n_fail:
            raise ConnectionError("transient")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def make_client(content: str, n_fail: int = 0, **kw):
    comp = StubCompletions(content, n_fail)
    stub = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return ChatClient(client=stub, **kw), comp


def test_request_parameters_and_json_mode():
    client, comp = make_client('{"turns": []}')
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = client.chat_json(msgs, temperature=0.9, max_tokens=8000, seed=7, use_cache=False)
    assert out == {"turns": []}
    assert comp.calls == [{"model": "gpt-4.1", "messages": msgs, "temperature": 0.9, "max_tokens": 8000,
                           "seed": 7, "response_format": {"type": "json_object"}}]


def test_plain_chat_omits_seed_and_response_format():
    client, comp = make_client("pong", model="other-model")
    assert client.chat([{"role": "user", "content": "ping"}], use_cache=False) == "pong"
    assert comp.calls[0] == {"model": "other-model", "messages": [{"role": "user", "content": "ping"}],
                             "temperature": 0.8, "max_tokens": 4096}


def test_retry_with_exponential_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", sleeps.append)
    client, comp = make_client("ok", n_fail=3)
    assert client.chat([{"role": "user", "content": "x"}], use_cache=False) == "ok"
    assert len(comp.calls) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    client, comp = make_client("ok", n_fail=99, max_retries=5)
    with pytest.raises(RuntimeError, match="after 5 retries"):
        client.chat([{"role": "user", "content": "x"}], use_cache=False)
    assert len(comp.calls) == 5


def test_disk_cache(tmp_path):
    client, comp = make_client('{"a": 1}', cache_dir=str(tmp_path))
    msgs = [{"role": "user", "content": "x"}]
    assert client.chat_json(msgs, seed=1) == {"a": 1}
    assert client.chat_json(msgs, seed=1) == {"a": 1}
    assert len(comp.calls) == 1
    client.chat_json(msgs, seed=2)
    assert len(comp.calls) == 2


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ChatClient()


@pytest.mark.parametrize("text", ['{"x": 1}', '```json\n{"x": 1}\n```', 'Here you go: {"x": 1} done'])
def test_extract_json(text):
    assert extract_json(text) == {"x": 1}


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here")
    assert json.loads("[1]") == extract_json("[1]")
