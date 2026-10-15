"""Run the behavioural matrix: every scenario x agent x repeat (Appendix H.3; 3 repeats -> 1,800 trials).

Every agent gets the same audio, session instruction and protocol. Results are appended to
``<out>/<model>.jsonl`` (one :class:`~mptb.fd.protocols.TrialResult` per line, plus scenario metadata) and
the runner is resumable: trials already recorded (valid or with an invalid reason) are skipped.
Live API sessions may overlap (``concurrency``); local models always run serially on one GPU.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .adapters import get_adapter
from .protocols import run_trial


def load_done(path: Path) -> Set[Tuple[str, str, int]]:
    """``(model, scenario_id, run)`` keys already present in a results JSONL file."""
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok") or r.get("invalid_reason"):
                    done.add((r["model"], r["scenario_id"], r["run"]))
    return done


def failed_record(sc: Dict[str, Any], model: str, run: int, exc: BaseException) -> Dict[str, Any]:
    """Invalid-trial record for a session that raised (e.g. the provider refused to open it)."""
    return {"scenario_id": sc["scenario_id"], "model": model, "run": run,
            "protocol": sc["protocol"], "event_type": sc["event_type"], "ok": False,
            "invalid_reason": f"exception: {exc!r}", "correct": None, "latency_ms": None,
            "model_speech_start": [], "model_speech_stop": [], "provider_events": [],
            "detail": {"traceback": traceback.format_exc()[-1500:]}}


def scenario_fields(sc: Dict[str, Any], pack: Dict[str, Any]) -> Dict[str, Any]:
    """Scenario metadata appended to every trial record (joins the trial to its gold event)."""
    return {"N": sc["N"], "conversation_type": sc["conversation_type"],
            "conversation_id": sc["conversation_id"], "target_speaker": sc["target_speaker"],
            "source_event_id": sc["source_event_id"], "gold_event_time": sc["event_time"],
            "gold_agent_behavior": sc["gold_agent_behavior"], "polarity": sc["polarity"],
            "audio_condition": pack.get("audio_condition"), "ts": time.time()}


async def run_model(model: str, scenarios: Dict[str, Dict[str, Any]], packs: Dict[str, Dict[str, Any]],
                    audio_root: Path, out: Path, sids: List[str], runs: int = 3,
                    concurrency: int = 1, timeout: float = 240.0, retries: int = 2,
                    backoff: float = 2.0, sleep: float = 0.0, log_every: int = 5,
                    adapter_kwargs: Optional[Dict[str, Any]] = None, resume: bool = True) -> Path:
    """Run all missing trials of one agent and append them to ``<out>/<model>.jsonl``."""
    adapter_kwargs = adapter_kwargs or {}
    jpath = out / f"{model}.jsonl"
    done = load_done(jpath) if resume else set()
    cls = get_adapter(model)
    todo = [(sid, r) for r in range(runs) for sid in sids if (model, sid, r) not in done]
    conc = 1 if not cls.realtime else max(1, concurrency)
    print(f"\n=== {model}: {len(todo)} trials ({len(sids)} scenarios x {runs} runs, "
          f"{len(done)} already done, concurrency={conc})")
    t_start = time.time()
    counters = {"ok": 0, "err": 0, "i": 0}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(conc)

    with open(jpath, "a") as fh:
        async def one(sid: str, run: int) -> None:
            sc, pack = scenarios[sid], packs[sid]
            rec = None
            async with sem:
                for attempt in range(retries + 1):
                    adapter = cls(**adapter_kwargs)
                    try:
                        res = await asyncio.wait_for(run_trial(adapter, sc, pack, audio_root / sid, run),
                                                     timeout=timeout)
                        rec = asdict(res)
                        break
                    except Exception as e:
                        try:
                            await adapter.close_session()
                        except Exception:
                            pass
                        if attempt >= retries:
                            rec = failed_record(sc, model, run, e)
                        else:
                            await asyncio.sleep(backoff * (2 ** attempt) + random.random())
            rec.update(scenario_fields(sc, pack))
            async with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                counters["i"] += 1
                counters["ok" if rec.get("ok") else "err"] += 1
                i = counters["i"]
                if i % log_every == 0 or i == len(todo):
                    el = time.time() - t_start
                    print(f"  [{i}/{len(todo)}] {sid} run={run} correct={rec.get('correct')} "
                          f"({counters['ok']} valid / {counters['err']} invalid, "
                          f"{el / max(i, 1):.1f}s per trial)", flush=True)
            if sleep:
                await asyncio.sleep(sleep)

        await asyncio.gather(*(one(sid, run) for sid, run in todo))
    return jpath


def select_scenarios(scenarios: Dict[str, Dict[str, Any]], packs: Dict[str, Any],
                     only: Optional[Iterable[str]] = None, event_types: Optional[Iterable[str]] = None,
                     limit: Optional[int] = None) -> List[str]:
    """Sorted scenario ids that have a pack, optionally filtered by id / event type and truncated."""
    sids = [s for s in scenarios if s in packs]
    if only:
        sids = [s for s in sids if s in set(only)]
    if event_types:
        sids = [s for s in sids if scenarios[s]["event_type"] in set(event_types)]
    sids.sort()
    return sids[:limit] if limit else sids
