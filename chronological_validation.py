"""Private, resumable outer walk-forward comparison with delayed close execution.

The fitter receives only a training prefix. Its internal selection/search is part
of the experiment, not a set of finalists nominated from the evaluation period.
"""
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from risk_aware_allocator import funded_target, scenario_cvar

METHOD = "whole-selection-walk-forward-v1"
STRATEGIES = ("incumbent", "risk-aware")


class ResearchPause(Exception):
    """Scheduling budget reached; committed trials remain resumable."""


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                # Windows readers/indexers can briefly hold a completed file.
                # Keep the prior atomic checkpoint while retrying the rename.
                time.sleep(.02 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def experiment_lock(path):
    """An OS lock releases on abrupt shutdown; two tabs cannot lose each other's work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.with_suffix(".lock").open("a+b")
    locked = False
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise ValueError("This research experiment is already running in another window") from exc
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def experiment_id(prices, volumes, metadata, settings):
    digest = hashlib.sha256()
    for frame in (prices, volumes):
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        digest.update(json.dumps(list(frame.columns)).encode())
    digest.update(json.dumps(metadata, sort_keys=True, default=str).encode())
    digest.update(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
    digest.update(METHOD.encode())
    return digest.hexdigest()


def fold_windows(index, minimum_train=252, test_sessions=30, folds=3):
    if minimum_train < 60 or test_sessions < 2 or not 2 <= folds <= 12:
        raise ValueError("Invalid chronological window settings")
    if len(index) < minimum_train + folds * test_sessions:
        raise ValueError(f"Need {minimum_train + folds * test_sessions} observed sessions; have {len(index)}")
    first = len(index) - folds * test_sessions
    return [(first + i * test_sessions, first + (i + 1) * test_sessions) for i in range(folds)]


def normalized_weights(value):
    weights = pd.Series(value, dtype=float)
    if (weights.empty or weights.index.duplicated().any() or not np.isfinite(weights).all()
            or weights.lt(0).any() or not math.isclose(float(weights.sum()), 1., abs_tol=1e-7)):
        raise ValueError("Complete normalized research weights required")
    return weights[weights > 1e-12]


def training_seed(train, volume, metadata, minimum_train):
    """A common, train-only starting basket, balanced across listing clusters."""
    eligible = {c for c in train if train[c].notna().sum() >= minimum_train
                and pd.notna(train[c].iloc[-1]) and train[c].iloc[-1] > 0}
    traded_value = (train.tail(63) * volume.reindex_like(train).tail(63)).median().fillna(0.)
    clusters = {}
    for row in metadata:
        ticker = row["Yahoo Ticker"]
        if ticker not in eligible or row.get("research_role") == "benchmark":
            continue
        cluster = str(row.get("Exchange", "Unknown")) + " · " + str(row.get("Currency", "Unknown"))
        clusters.setdefault(cluster, []).append(ticker)
    for names in clusters.values():
        names.sort(key=lambda t: (-float(traded_value[t]), t))
    chosen = []
    while len(chosen) < 50 and any(clusters.values()):
        for cluster in sorted(clusters):
            if clusters[cluster] and len(chosen) < 50:
                chosen.append(clusters[cluster].pop(0))
    if not chosen:
        raise ValueError("No training-only initial portfolio available")
    return {ticker: 1 / len(chosen) for ticker in chosen}


def simulate_fold(prices, initial, target, cost_bps):
    """Signal at training close; rebalance at NEXT close, then hold drifting units.

The first forward return belongs to old holdings. Selected prices are never used
to fit or replace missing target weights. Costs cover gross buys plus sells;
funding is deducted from capital rather than added as unexplained leverage.
"""
    old = normalized_weights(initial)
    target = normalized_weights(target)
    names = old.index.union(target.index, sort=False)
    frame = prices.reindex(columns=names).apply(pd.to_numeric, errors="coerce")
    before_fill = frame.iloc[:2].reindex(columns=old.index)
    after_fill = frame.iloc[1:].reindex(columns=target.index)
    if (len(frame) < 3 or not np.isfinite(before_fill.to_numpy()).all()
            or not np.isfinite(after_fill.to_numpy()).all()
            or before_fill.le(0).any().any() or after_fill.le(0).any().any()):
        raise ValueError("Observed forward marks missing for held/selected securities; fold withheld")
    if not math.isfinite(float(cost_bps)) or not 0 <= float(cost_bps) < 2500:
        raise ValueError("Invalid modeled trading cost")
    first_move = before_fill.iloc[1] / before_fill.iloc[0]
    wealth = float(old @ first_move)
    drift = (old * first_move / wealth).reindex(names, fill_value=0.).to_numpy()
    proposed = target.reindex(names, fill_value=0.)
    amounts, cost = funded_target(proposed.to_numpy(), drift, float(cost_bps) / 10000)
    turnover = float(np.abs(amounts - drift).sum())
    cost_amount = wealth * cost
    wealth *= 1 - cost
    weights = target.to_numpy()
    path = [wealth]
    gross = after_fill.to_numpy()[1:] / after_fill.to_numpy()[:-1]
    for daily in gross:
        change = float(weights @ daily)
        wealth *= change
        weights = weights * daily / change
        path.append(wealth)
    return {"path": path, "ending_weights": {name: float(w) for name, w in zip(target.index, weights) if w > 1e-12},
            "gross_turnover": turnover, "cost_fraction": cost,
            "cost_amount_initial_base": cost_amount,
            "net_return": wealth - 1., "execution_date": str(frame.index[1].date())}


def path_metrics(path, turnover=0., cost=0.):
    curve = np.r_[1., np.asarray(path, dtype=float)]
    if len(curve) < 2 or not np.isfinite(curve).all() or (curve <= 0).any():
        raise ValueError("Invalid observed equity path")
    daily = curve[1:] / curve[:-1] - 1
    return {"net_return": float(curve[-1] - 1),
            "max_drawdown": float(np.max(1 - curve / np.maximum.accumulate(curve))),
            "observed_one_session_es95": scenario_cvar(daily, .95),
            "gross_turnover": float(turnover), "cost_fraction": float(cost),
            "sessions": len(daily)}


def comparison_summary(state):
    if state.get("status") != "complete":
        return {"status": "incomplete", "decision": "Resume to finish the comparison"}
    completed = [row for row in state["folds"] if row.get("status") == "complete"]
    if len(completed) != state["settings"]["folds"] * len(STRATEGIES):
        return {"status": "unavailable", "decision": "Missing forward evidence; no comparison established"}
    totals = {}
    for strategy in STRATEGIES:
        rows = [r for r in completed if r["strategy"] == strategy]
        wealth, path, total_cost = 1., [], 0.
        for row in rows:
            total_cost += wealth * row["cost_amount_initial_base"]
            path.extend((wealth * np.asarray(row["path"])).tolist())
            wealth = path[-1]
        totals[strategy] = path_metrics(path, sum(r["gross_turnover"] for r in rows),
                                        total_cost)
        totals[strategy]["fit_seconds"] = sum(r["fit_seconds"] for r in rows)
        totals[strategy]["held_folds"] = sum(r["fit_status"] != "selected" for r in rows)
    base, new = totals["incumbent"], totals["risk-aware"]
    supported = (new["held_folds"] == 0 and base["held_folds"] == 0
                 and new["net_return"] > base["net_return"]
                 and new["max_drawdown"] <= base["max_drawdown"]
                 and new["observed_one_session_es95"] <= base["observed_one_session_es95"])
    return {"status": "available", "strategies": totals,
            "decision": ("Risk-aware candidate improved return and observed downside in this sample"
                         if supported else "No consistent improvement established in this sample"),
            "auto_promote": False}


def run_validation(prices, volumes, metadata, settings, fit, checkpoint, *, on_progress=None,
                   runtime_seconds=120):
    with experiment_lock(checkpoint):
        return _run_validation(prices, volumes, metadata, settings, fit, checkpoint,
                               on_progress=on_progress, runtime_seconds=runtime_seconds)


def _run_validation(prices, volumes, metadata, settings, fit, checkpoint, *, on_progress=None,
                    runtime_seconds=120):
    """Advance a frozen experiment. Checkpoint after each fit trial and outer fold."""
    if (prices.empty or prices.index.duplicated().any() or prices.columns.duplicated().any()
            or not prices.index.is_monotonic_increasing):
        raise ValueError("Unique chronological price dates and security identifiers required")
    if not math.isfinite(float(settings["cost_bps"])) or not 0 <= float(settings["cost_bps"]) < 2500:
        raise ValueError("Invalid modeled trading cost")
    windows = fold_windows(prices.index, settings["minimum_train"], settings["test_sessions"], settings["folds"])
    key = experiment_id(prices, volumes, metadata, settings)
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("experiment_id") != key:
            raise ValueError("Evidence/configuration changed; start a new experiment, old checkpoint retained")
    else:
        state = {"method": METHOD, "experiment_id": key, "settings": settings, "folds": [],
                 "trials": {}, "work_seconds": {}, "status": "paused", "as_of": str(prices.index[-1].date()),
                 "limitations": ["Current supplied universe: historical membership and delisted names not reconstructed",
                                 "Adjusted-price cache can contain later vendor revisions",
                                 "Fractional, close-based simulation with per-side costs; no broker fills, taxes or market impact",
                                 "Full-target research rebalances; whole-share rounding and partial-sell suppression are not replayed",
                                 "Small-sample research comparison; no automatic strategy promotion"]}
    deadline = time.monotonic() + runtime_seconds

    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_save(checkpoint, state)

    if state["status"] == "complete":
        return state
    save()
    try:
        for number, (start, end) in enumerate(windows, 1):
            train = prices.iloc[:start].copy()
            cutoff = train.index[-1]
            for strategy in STRATEGIES:
                if any(r["fold"] == number and r["strategy"] == strategy for r in state["folds"]):
                    continue
                if time.monotonic() >= deadline:
                    raise ResearchPause()
                prior = [r for r in state["folds"] if r["strategy"] == strategy]
                if prior:
                    initial = prior[-1]["ending_weights"]
                else:
                    # A common synthetic starting portfolio, chosen from TRAINING
                    # coverage only. Today's personal holdings never seed a past fold.
                    initial = training_seed(train, volumes.reindex(train.index), metadata, settings["minimum_train"])
                trial_key = f"{number}:{strategy}"
                trials = state["trials"].setdefault(trial_key, {})
                if on_progress:
                    on_progress(f"Fold {number}/{len(windows)} · {strategy} · training through {cutoff.date()}",
                                len(state["folds"]) / (len(windows) * 2))
                started = time.monotonic()
                try:
                    choice = fit(train, volumes.reindex(train.index).copy(), metadata, initial, strategy,
                                 trials, save, deadline, on_progress)
                finally:
                    state["work_seconds"][trial_key] = state["work_seconds"].get(trial_key, 0.) + time.monotonic() - started
                    save()
                fit_seconds = state["work_seconds"][trial_key]
                target = choice.get("weights") or initial
                # No filtering/replacement of selected assets based on test outcomes.
                result = simulate_fold(prices.iloc[start - 1:end], initial, target, settings["cost_bps"])
                result.update(fold=number, strategy=strategy, status="complete",
                              fit_status=choice.get("status", "held"), reason=choice.get("reason"),
                              pair=choice.get("pair"), fit_seconds=fit_seconds,
                              train_end=str(cutoff.date()), test_end=str(prices.index[end - 1].date()),
                              trial_count=len(trials))
                state["folds"].append(result)
                save()
        state["status"] = "complete"
    except ResearchPause:
        state["status"] = "paused"
    except (ValueError, KeyError, TypeError) as exc:
        state["status"] = "blocked"
        state["error"] = str(exc)
    state["summary"] = comparison_summary(state)
    save()
    return state
