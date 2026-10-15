"""Analysis of the full-duplex behavioural study (Table 4; Appendix F Tables 16-19 and companions; Appendix G).

Input: the per-agent trial logs written by :mod:`mptb.fd.runner` (``<runs>/<model>.jsonl``) and, for the
offline-online association, the raw detector records of :mod:`mptb.fd.offline`.

Metric definitions (rates over *valid* trials):

* S  = P(agent starts in ``[t-0.25, t+3.0]`` | true_eot)            START recall
* FS = P(agent starts inside the hold span | hold_pause)            START false-alarm rate
* Y  = P(agent's utterance ends within 1.5 s of injection | true_int) STOP recall
* FY = P(same | backchannel)                                         STOP false-alarm rate
* bACC = (recall + 1 - FPR) / 2, action bias A = recall + FPR - 1.

Bootstrap CIs of bACC resample *scenarios* (the 3 repeats of a scenario are repeated measures),
independently for the positive and negative family (:func:`bacc_bootstrap`).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

FAMILIES = ["true_eot", "hold_pause", "true_int", "backchannel"]
POSITIVE = {"true_eot": True, "hold_pause": False, "true_int": True, "backchannel": False}
PROTOCOL = {"true_eot": "START", "hold_pause": "START", "true_int": "STOP", "backchannel": "STOP"}
PROBES = {"START": ("true_eot", "hold_pause"), "STOP": ("true_int", "backchannel")}
BOOT_B = 10000
BOOT_SEED = 20261012
BOOT_METHOD = ("scenario-clustered stratified bootstrap of balanced accuracy over valid trials, "
               f"B={BOOT_B}, seed {BOOT_SEED}")
CORRELATION_NOTE = "n is small (one point per model); reported descriptively, not as a test."


# ----------------------------------------------------------------------------------------- loading
def load_runs(run_dir: Path) -> List[Dict[str, Any]]:
    """All trial records under ``run_dir/*.jsonl``, de-duplicated on (model, scenario, run) keeping the last."""
    rows: List[Dict[str, Any]] = []
    for p in sorted(Path(run_dir).glob("*.jsonl")):
        for line in p.open():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    seen: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        seen[(r["model"], r["scenario_id"], r["run"])] = r
    return list(seen.values())


def valid(r: Dict[str, Any]) -> bool:
    """A trial is valid if the session was scorable (see :mod:`mptb.fd.protocols`)."""
    return bool(r.get("ok")) and r.get("correct") is not None


def fired(r: Dict[str, Any]) -> Optional[bool]:
    """Did the agent take the probed action (START: started in the scored window; STOP: stopped in it)?"""
    d = r.get("detail") or {}
    if r["protocol"] == "START":
        if r["event_type"] == "true_eot":
            return d.get("delta_start_s") is not None
        return d.get("false_start")
    return d.get("stopped_in_window")


def bargein_stop(r: Dict[str, Any]) -> Optional[bool]:
    """STOP trial whose in-window stop coincided with a provider cancel/interrupt event (Table 19)."""
    if r.get("protocol") != "STOP":
        return None
    d = r.get("detail") or {}
    if not d:
        return None
    return bool(d.get("stopped_in_window")) and bool(d.get("provider_cancel_events"))


# -------------------------------------------------------------------------------------- statistics
def mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values (None if there are none)."""
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def percentile(xs: Sequence[Optional[float]], q: float) -> Optional[float]:
    """Linear-interpolation percentile of the non-None values."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo))


def rate(rows: Sequence[Dict[str, Any]], key: Callable[[Dict[str, Any]], Optional[bool]]) -> Dict[str, Any]:
    """Pooled rate of ``key`` over ``rows`` (None values skipped), with trial and scenario counts."""
    by_sc: Dict[str, List[bool]] = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is not None:
            by_sc[r["scenario_id"]].append(bool(v))
    flat = [v for k in sorted(by_sc) for v in by_sc[k]]
    return {"mean": sum(flat) / len(flat) if flat else None, "n_trials": len(flat), "n_scenarios": len(by_sc)}


def spearman(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    """Spearman rank correlation (average ranks for ties) over pairs where both values exist."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None

    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        rk = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


# ------------------------------------------------------------------------------- per-agent metrics
def model_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Coverage, per-family accuracy/latency/barge-in/by-N metrics and the headline rates of one agent."""
    all_rows = list(rows)
    ok_rows = [r for r in all_rows if valid(r)]
    out: Dict[str, Any] = {
        "n_trials_attempted": len(all_rows),
        "n_trials_valid": len(ok_rows),
        "validity_rate": len(ok_rows) / len(all_rows) if all_rows else None,
        "n_runs_attempted": len({r.get("run") for r in all_rows if r.get("run") is not None}),
    }
    reasons: Dict[str, int] = defaultdict(int)
    for r in all_rows:
        if not valid(r):
            reasons[str(r.get("invalid_reason"))[:80]] += 1
    out["invalid_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))

    fam: Dict[str, Any] = {}
    for f in FAMILIES:
        fr = [r for r in ok_rows if r["event_type"] == f]
        if not fr:
            continue
        block: Dict[str, Any] = {
            "n_trials": len(fr),
            "n_scenarios": len({r["scenario_id"] for r in fr}),
            "accuracy": rate(fr, lambda r: r.get("correct")),
            "fire_rate": rate(fr, fired),
        }
        per_sc: Dict[str, List[bool]] = defaultdict(list)
        for r in fr:
            per_sc[r["scenario_id"]].append(bool(r["correct"]))
        # run-to-run consistency: fraction of scenarios (>1 valid repeat) on which all repeats agree
        block["run_consistency"] = mean([1.0 if len(set(v)) == 1 else 0.0 for v in per_sc.values() if len(v) > 1])
        lat = [r["latency_ms"] for r in fr if r.get("latency_ms") is not None]
        block["latency_ms"] = {"p10": percentile(lat, 10), "p50": percentile(lat, 50),
                               "p90": percentile(lat, 90), "n": len(lat)}
        if PROTOCOL[f] == "STOP":
            block["resumed_within_2s"] = rate(fr, lambda r: (r.get("detail") or {}).get("resumed_within_2s"))
            block["provider_cancel_rate"] = rate(
                fr, lambda r: bool((r.get("detail") or {}).get("provider_cancel_events")))
            block["bargein_stop_rate"] = rate(fr, bargein_stop)
        block["by_N"] = {str(v): rate([r for r in fr if r["N"] == v], lambda r: r.get("correct"))
                         for v in sorted({r["N"] for r in fr}, key=str)}
        fam[f] = block
    out["families"] = fam

    def get(f, k):
        return (fam.get(f, {}).get(k) or {}).get("mean")

    h = {"S_correct_start": get("true_eot", "accuracy"), "false_start_rate": get("hold_pause", "fire_rate"),
         "Y_correct_yield": get("true_int", "accuracy"), "false_yield_rate": get("backchannel", "fire_rate")}
    if h["S_correct_start"] is not None and h["false_start_rate"] is not None:
        h["start_balanced"] = 0.5 * (h["S_correct_start"] + (1 - h["false_start_rate"]))
    if h["Y_correct_yield"] is not None and h["false_yield_rate"] is not None:
        h["stop_balanced"] = 0.5 * (h["Y_correct_yield"] + (1 - h["false_yield_rate"]))
    if h["S_correct_start"] is not None and h["false_start_rate"] is not None:
        h["start_bias"] = h["S_correct_start"] + h["false_start_rate"] - 1.0
    if h["Y_correct_yield"] is not None and h["false_yield_rate"] is not None:
        h["stop_bias"] = h["Y_correct_yield"] + h["false_yield_rate"] - 1.0
    h["bargein_rate_true_int"] = get("true_int", "bargein_stop_rate")
    h["bargein_rate_backchannel"] = get("backchannel", "bargein_stop_rate")
    out["headline"] = h
    return out


# ------------------------------------------------------------------ bACC bootstrap (Table 4 CIs)
def bacc_bootstrap(rows: Sequence[Dict[str, Any]], b: int = BOOT_B, seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Balanced accuracy and 95% CI per agent and probe (Table 4 brackets).

    Valid trials are summed per scenario; positive and negative scenarios are resampled independently
    (scenario-clustered, stratified), ``b`` replicates, RNG ``numpy.random.default_rng(seed)``
    re-created for each agent x probe. Returns ``{model: {"START"|"STOP": {bacc, lo, hi, p_le_half,
    n_pos, n_neg}}}``.
    """
    import numpy as np

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("ok"):
            by_model[r["model"]].append(r)
    out: Dict[str, Any] = {}
    for m in sorted(by_model):
        g = by_model[m]

        def counts(ev):
            s: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
            for r in g:
                if r["event_type"] != ev:
                    continue
                v = fired(r)
                cell = s[r["scenario_id"]]   # every scenario with a valid trial is a cluster
                if v is not None:
                    cell[0] += int(v)
                    cell[1] += 1
            keys = sorted(s)
            return (np.array([s[k][0] for k in keys], dtype=float), np.array([s[k][1] for k in keys], dtype=float))

        res = {}
        for proto, (pos, neg) in PROBES.items():
            rng = np.random.default_rng(seed)
            ps, pc = counts(pos)
            ns, nc = counts(neg)
            tpr = ps.sum() / pc.sum()
            fpr = ns.sum() / nc.sum()
            bacc = (tpr + 1 - fpr) / 2
            ip = rng.integers(0, len(ps), (b, len(ps)))
            i_n = rng.integers(0, len(ns), (b, len(ns)))
            bb = (ps[ip].sum(1) / pc[ip].sum(1) + 1 - ns[i_n].sum(1) / nc[i_n].sum(1)) / 2
            res[proto] = dict(bacc=bacc, lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)),
                              p_le_half=float((bb <= 0.5).mean()), n_pos=int(pc.sum()), n_neg=int(nc.sum()))
        out[m] = res
    return out


# ------------------------------------------------------------- offline-online association (App. G)
def contingency(rows: Sequence[Dict[str, Any]], offline: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    """Event-level 2x2: offline detector correct (x) vs behaviour correct (y) on the same scenario.

    ``offline[model][scenario_id]`` is the detector outcome. For each scope (all / START / STOP):
    P(y | x), P(y | not x), odds ratio with Haldane-Anscombe correction (+0.5 per cell) and Wald 95% CI.
    """
    out: Dict[str, Any] = {}
    for scope, sel in (("all", lambda r: True),
                       ("START", lambda r: r["protocol"] == "START"),
                       ("STOP", lambda r: r["protocol"] == "STOP")):
        n11 = n10 = n01 = n00 = 0
        for r in rows:
            if not valid(r) or not sel(r):
                continue
            x = offline.get(r["model"], {}).get(r["scenario_id"])
            if x is None:
                continue
            y = bool(r["correct"])
            if x and y:
                n11 += 1
            elif x and not y:
                n10 += 1
            elif not x and y:
                n01 += 1
            else:
                n00 += 1
        p1 = n11 / (n11 + n10) if (n11 + n10) else None
        p0 = n01 / (n01 + n00) if (n01 + n00) else None
        a, b, c, d = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
        orv = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        out[scope] = {
            "n_detector_correct": n11 + n10, "n_detector_wrong": n01 + n00,
            "P_behaviour_ok_given_detector_ok": p1,
            "P_behaviour_ok_given_detector_wrong": p0,
            "absolute_difference": (p1 - p0) if (p1 is not None and p0 is not None) else None,
            "odds_ratio": orv,
            "odds_ratio_ci95": [math.exp(math.log(orv) - 1.96 * se), math.exp(math.log(orv) + 1.96 * se)],
            "counts": {"x1y1": n11, "x1y0": n10, "x0y1": n01, "x0y0": n00},
        }
    return out


def association(rows: Sequence[Dict[str, Any]], by_model: Dict[str, Any],
                offline: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Appendix G: model-level Spearman correlations, event-level contingency and offline scores.

    ``offline`` maps model -> :func:`mptb.fd.offline.score_detector` output (agents without an offline
    pass are absent). Spearman pairs the offline balanced score (E_eot / I_int) with the positive-only
    behavioural rate (S / Y).
    """
    models = sorted(by_model)
    e = [(offline.get(m) or {}).get("E_eot") for m in models]
    s = [by_model[m]["headline"]["S_correct_start"] for m in models]
    i = [(offline.get(m) or {}).get("I_int") for m in models]
    y = [by_model[m]["headline"]["Y_correct_yield"] for m in models]
    per_event = {m: ((offline.get(m) or {}).get("per_event") or {}) for m in models}
    return {
        "correlations": {"spearman_EOT_vs_START": spearman(e, s), "spearman_INT_vs_STOP": spearman(i, y),
                         "note": CORRELATION_NOTE},
        "contingency": contingency(rows, per_event) if any(per_event.values()) else {
            "status": "skipped: no offline per-event results available"},
        "mptb_offline_fd_models": {m: {k: v for k, v in (offline.get(m) or {}).items() if k != "per_event"}
                                   for m in models},
    }


# ------------------------------------------------------------------------------------- exports
def _r(v, nd: int = 4):
    return "" if v is None else round(float(v), nd)


TRIAL_COLUMNS = ["model", "scenario_id", "run", "protocol", "event_type", "polarity",
                 "N", "conversation_type", "conversation_id", "source_event_id",
                 "ok", "invalid_reason", "correct", "latency_ms", "fired",
                 "bargein_stop", "provider_cancel_events", "gold_event_time"]


def write_trials_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """One row per trial (incl. invalid ones) with the derived ``fired`` and ``bargein_stop`` flags."""
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRIAL_COLUMNS)
        for r in sorted(rows, key=lambda x: (x["model"], x["scenario_id"], x["run"])):
            d = r.get("detail") or {}
            w.writerow([r.get("model"), r.get("scenario_id"), r.get("run"),
                        r.get("protocol"), r.get("event_type"), r.get("polarity"),
                        r.get("N"), r.get("conversation_type"),
                        r.get("conversation_id"), r.get("source_event_id"),
                        r.get("ok"), r.get("invalid_reason"), r.get("correct"),
                        _r(r.get("latency_ms"), 1), fired(r), bargein_stop(r),
                        d.get("provider_cancel_events"), r.get("gold_event_time")])


def analyze(run_dir: Path, offline: Dict[str, Dict[str, Any]], out: Path) -> Dict[str, Any]:
    """Run the whole analysis and write its outputs to ``out``.

    Writes ``fd_trials.csv``, ``fd_by_model.json`` (per-agent metrics), ``fd_bacc_ci.json`` (Table 4
    CIs) and ``fd_association.json`` (Appendix G). Returns ``{"by_model", "bacc_ci", "association"}``.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_runs(run_dir)
    if not rows:
        raise SystemExit(f"no trial records under {run_dir}")
    models = sorted({r["model"] for r in rows})
    by_model = {m: model_metrics([r for r in rows if r["model"] == m]) for m in models}
    ci = bacc_bootstrap(rows)
    assoc = association(rows, by_model, offline)

    write_trials_csv(out / "fd_trials.csv", rows)
    (out / "fd_by_model.json").write_text(json.dumps({"n_trials": len(rows), "models": by_model}, indent=2))
    (out / "fd_bacc_ci.json").write_text(json.dumps({"method": BOOT_METHOD, "results": ci}, indent=1))
    (out / "fd_association.json").write_text(json.dumps(assoc, indent=1))
    return {"by_model": by_model, "bacc_ci": ci, "association": assoc}
