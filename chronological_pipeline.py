"""Offline bridge from the production selection/search code to outer validation."""
import hashlib
import ast
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from chronological_validation import ResearchPause
from market_history_cache import load_cached_market_history
from scalable_universe_preselection import calculate_risk_appetite_regime, filter_candidates_by_market_cap, rank_scalable_candidates


def source_fingerprint():
    root = Path(__file__).parent
    names = ("chronological_validation.py", "chronological_pipeline.py", "portfolio_rebalancer_database.py",
             "risk_aware_allocator.py", "allocation_integrity.py", "scalable_universe_preselection.py",
             "robust_momentum_filter.py", "portfolio_risk_metrics.py", "price_history_integrity.py",
             "portfolio_optimizer_config.json", "two_dimensional_search.py", "adaptive_allocation.py", "allocation_policy_config.py")
    digest = hashlib.sha256()
    logic = {"get_daily_log_returns", "select_redundant_tickers", "weight_bounds", "feasible_initial_weights",
             "apply_allocation_policy", "validate_optimizer_allocation", "shrunk_covariance", "portfolio_stats",
             "portfolio_stats_comparison", "optimize_prepared_return_matrix", "search_universal_shortlist_caps",
             "optimize_portfolio_max_return_given_daily_risk", "optimize_portfolio_target_volatility",
             "optimize_max_sharpe_ratio", "enforce_min_weight_postprocess", "_percent_drop_count",
             "_select_drop_bottom_tickers_fallback", "screen_log_returns_for_exact_optimizer", "_optimizer_deadline_callback"}
    logic.update({"_normalize_currency_code", "_currency_major_and_unit_factor"})
    for name in names:
        content = (root / name).read_bytes()
        if name == "portfolio_rebalancer_database.py":
            tree = ast.parse(content.decode("utf-8-sig"))
            content = ast.dump(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                               and n.name in logic], type_ignores=[]), include_attributes=False).encode()
        digest.update(content)
    return digest.hexdigest()


def research_universe(core, owner):
    """Freeze the CURRENT universe/pauses; do not claim historical membership."""
    shared_owner = core.chronological_research_config()["universal_owner"]
    rows = pd.concat([core.load_master_holdings(shared_owner), core.load_master_holdings(owner)], ignore_index=True)
    with core.get_db_connection() as db:
        rows = core.filter_optimizer_candidates(rows, core.optimizer_exclusions_frame(db))
        rows, _ = core.exclude_deferred_candidates(rows, core.active_trade_deferrals(db, owner))
    rows["Yahoo Ticker"] = rows["Yahoo Ticker"].astype(str).str.strip().str.upper()
    # Quantity/cost-basis edits do not change the synthetic experiment and must
    # not invalidate its work. Only actual selection inputs enter the fingerprint.
    columns = [c for c in ("Yahoo Ticker", "Exchange", "Currency", "Market Cap Millions",
                           "Market Cap As Of", "Market Cap Imported At") if c in rows]
    return rows.drop_duplicates("Yahoo Ticker").sort_values("Yahoo Ticker")[columns].reset_index(drop=True)


def _listing_calendar(row):
    ticker, exchange = str(row["Yahoo Ticker"]), str(row.get("Exchange", "")).upper()
    if ticker.endswith(".NS"): return "NSE"
    if ticker.endswith(".BO"): return "BSE"
    if exchange == "CCC" or ticker.endswith("-USD"): return "24/7"
    for suffix, calendar in ((".SA", "BMF"), (".L", "LSE"), (".TO", "TSX"),
                             (".HK", "HKEX"), (".T", "JPX"), (".AX", "ASX")):
        if ticker.endswith(suffix): return calendar
    if exchange in {"NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS", "NYSE", "NASDAQ", "NYSEARCA"}:
        return "NYSE"
    raise ValueError("Listing calendar not mapped")


def closed_session_marks(raw, days, opened):
    """Forward carry only verified closed sessions, never a missing open quote."""
    native = raw.where(raw > 0).reindex(days)
    gap = native.isna().astype("boolean").where(days.isin(opened)).ffill().fillna(True).astype(bool)
    marked = native.copy()
    closed = ~days.isin(opened)
    marked.loc[closed & ~gap] = native.ffill().loc[closed & ~gap]
    return marked


def cached_research_panel(cache_path, universe, as_of, *, currency_unit, history_years=3):
    """No downloads, no future FX/backfill, and no full-history outcome screens."""
    end = pd.Timestamp(as_of).normalize()
    start = end - pd.DateOffset(years=history_years)
    rows = universe.to_dict("records")
    currencies = {currency_unit(r.get("Currency", ""))[0] for r in rows}
    routes = {currency: [f"{currency}INR=X", f"{currency}USD=X"] for currency in currencies - {"INR"}}
    routes.setdefault("USD", ["USDINR=X", "INR=X"])
    symbols = [r["Yahoo Ticker"] for r in rows]
    regime_names = ("IWM", "SPY", "XLY", "XLP", "HYG", "LQD")
    rows.extend({"Yahoo Ticker": t, "Currency": "USD", "Exchange": "NYSE", "research_role": "benchmark"}
                for t in regime_names if t not in symbols)
    requested = tuple(dict.fromkeys(symbols + list(regime_names) + [t for names in routes.values() for t in names]))
    prices, volumes = load_cached_market_history(cache_path, requested, start, end + pd.Timedelta(days=1))
    if prices.empty:
        raise ValueError("No cached history available. Run the normal data refresh first; this test never downloads.")
    end = min(end, prices.index.max())
    days = pd.date_range(start, end)
    # A common valuation/session clock, declared rather than guessed business days.
    joint = mcal.get_calendar("NSE").valid_days(start, end).tz_localize(None).intersection(
        mcal.get_calendar("NYSE").valid_days(start, end).tz_localize(None))
    fx = {"INR": pd.Series(1., index=days)}

    def route(name):
        # Joint NSE/NYSE dates are open FX weekdays. A missing same-day FX quote
        # must not be replaced with yesterday's rate and reported as current.
        return prices[name].where(prices[name] > 0).reindex(days) if name in prices else pd.Series(np.nan, index=days)

    fx["USD"] = route("USDINR=X").combine_first(route("INR=X"))
    for currency in currencies - {"INR", "USD"}:
        fx[currency] = route(f"{currency}INR=X").combine_first(route(f"{currency}USD=X") * fx["USD"])
    marked, included, omitted, calendars = {}, [], [], {}
    for row in rows:
        ticker = row["Yahoo Ticker"]
        currency, unit_factor = currency_unit(row.get("Currency", ""))
        try:
            calendar = _listing_calendar(row)
            if calendar not in calendars:
                calendars[calendar] = days if calendar == "24/7" else mcal.get_calendar(calendar).valid_days(start, end).tz_localize(None)
            opened = calendars[calendar]
            if ticker not in prices or currency not in fx:
                raise ValueError("Cached prices/FX unavailable")
            series = closed_session_marks(prices[ticker], days, opened) * fx[currency] * unit_factor
            if not series.notna().any():
                raise ValueError("No usable cached marks")
            marked[ticker] = series.reindex(joint)
            included.append(row)
        except (ValueError, KeyError, RuntimeError):
            omitted.append(ticker)
    if not marked:
        raise ValueError("No cached securities have supported price, FX and calendar evidence")
    frame = pd.DataFrame(marked, index=joint)
    # Trim only the unavailable prefix/tail of the entire supplied cache, not an
    # individual security's poor outcomes or test-period coverage.
    first = frame.notna().any(axis=1)
    if not first.any():
        raise ValueError("No observed research session marks")
    frame = frame.loc[first[first].index[0]:]
    return frame, volumes.reindex(index=frame.index, columns=frame.columns), included, {
        "requested": len(symbols), "included": sum(r.get("research_role") != "benchmark" for r in included),
        "benchmark_series": sum(r.get("research_role") == "benchmark" for r in included),
        "omitted": sum(t in symbols for t in omitted),
        "omitted_tickers": [t for t in omitted if t in symbols], "clock": "joint NSE/NYSE sessions",
        "basis": "INR", "network_requests": 0, "history_years": history_years,
        "data_through": str(frame.dropna(how="all").index[-1].date()),
    }


def make_pipeline_fitter(core, settings):
    """Use the actual production funnel, adaptive X/Y search and allocation guard."""
    config = dict(settings["search"])
    active = core.chronological_research_config()
    active["allocation_policy"] = config.get("allocation_policy", active["allocation_policy"])
    prepared_by_training = {}
    cached_cutoff = None

    def fit(train, volume, metadata, initial, strategy, trials, save, deadline, progress):
        nonlocal cached_cutoff
        cutoff = train.index[-1]
        if cached_cutoff != cutoff:
            prepared_by_training.clear()  # Bound memory to the current outer fold.
            cached_cutoff = cutoff
        rows = pd.DataFrame(metadata).copy()
        owned = set(initial)
        for key in ("Market Cap As Of", "Market Cap Imported At"):
            observed = pd.to_datetime(rows.get(key, pd.Series(pd.NaT, index=rows.index)), utc=True, errors="coerce")
            rows.loc[observed.isna() | observed.gt(pd.Timestamp(cutoff, tz="UTC")), "Market Cap Millions"] = np.nan
        candidates, cap_report = filter_candidates_by_market_cap(
            rows.loc[rows.get("research_role", pd.Series("candidate", index=rows.index)).ne("benchmark")
                     & ~rows["Yahoo Ticker"].isin(owned)],
            exclusion_fraction=active["market_cap_exclusion_fraction"], reference_date=cutoff.date())
        metadata_map = rows.set_index("Yahoo Ticker").to_dict("index")
        allocation = pd.DataFrame([
            {"Yahoo Ticker": name, "Symbol": name, "Currency": "INR",
             "Weight": weight, "Quantity": 1., "Exchange": metadata_map.get(name, {}).get("Exchange", "Unknown")}
            for name, weight in initial.items()])
        prepared = prepared_by_training.setdefault(tuple(sorted(initial.items())), {})
        policy = dict(active["allocation_policy"], enabled=(strategy == "risk-aware"))

        def prepare(cap):
            if time.monotonic() >= deadline:
                raise ResearchPause()
            if cap not in prepared:
                if progress:
                    progress(f"{strategy} · training {cutoff.date()} · rebuilding shortlist X {cap:,} from cached training history", None)
                names = [t for t in candidates["Yahoo Ticker"] if t in train and t not in owned]
                selected, _ = rank_scalable_candidates(
                    train.reindex(columns=names).tail(504), volume.reindex(columns=names).tail(504),
                    {r["Yahoo Ticker"]: f"{r.get('Exchange', 'Unknown')} · {r.get('Currency', 'Unknown')}" for r in metadata},
                    maximum_candidates=cap, minimum_per_cluster=5,
                    risk_appetite_regime=calculate_risk_appetite_regime(train))
                names = list(dict.fromkeys([*initial, *selected]))
                extended = pd.concat([allocation, pd.DataFrame([
                    {"Yahoo Ticker": t, "Symbol": t, "Currency": "INR", "Weight": 0., "Quantity": 0.}
                    for t in selected if t not in owned])], ignore_index=True)
                returns, info = core.get_daily_log_returns(
                    names, end_date=cutoff, drop_bottom_pct=config["drop_bottom_pct"],
                    redundancy_corr_threshold=config["redundancy_corr_threshold"], owned_tickers=tuple(owned),
                    progress_callback=(lambda stage, fraction, details: progress(f"{strategy} · X {cap:,} · {stage}", None)) if progress else None,
                    _history_snapshot=(train, volume, {}))
                returns.attrs["research_cap"] = cap
                info["universal_preselection"] = {"cap": cap, "eligible": len(candidates), "shortlisted": len(selected),
                                                   "market_cap_filter": cap_report}
                prepared[cap] = (extended, returns, info)
            return prepared[cap]

        def solve(matrix, current, **kwargs):
            cap, assets = matrix.attrs["research_cap"], kwargs["maximum_assets"]
            trial_id = f"{cap}:{assets}"
            if trial_id not in trials:
                if len(trials) >= settings["trials_per_fold"]:
                    raise ResearchTrialsComplete()
                if time.monotonic() >= deadline:
                    raise ResearchPause()
                if progress:
                    progress(f"{strategy} · training {cutoff.date()} · X {cap:,} × Y {assets:,} · trial {len(trials)+1}/{settings['trials_per_fold']}", None)
                started = time.monotonic()
                weights, fitted, _, stats, report = core.optimize_prepared_return_matrix(
                    matrix, current, allocation_policy=policy, **kwargs)
                reason = (report.get("allocation_policy") or {}).get("reason")
                if not reason:
                    reason = "; ".join((report.get("allocation_validation") or {}).get("violations") or []) or None
                trial = {"pair": [int(cap), int(assets)], "columns": list(fitted.columns),
                         "weights": None if weights is None else dict(zip(fitted.columns, np.asarray(weights, dtype=float).tolist())),
                         "annual_return": None if not stats else float(stats["Annual Return"]),
                         "train_sessions": len(fitted),
                         "seconds": time.monotonic() - started, "reason": reason,
                         "allocator_selected": (report.get("allocation_policy") or {}).get("selected", "incumbent"),
                         "policy_constraints_applied": bool((report.get("allocation_policy") or {}).get("constraints_applied")),
                         "timed_out": bool(report.get("timed_out"))}
                if trial["annual_return"] is not None and not math.isfinite(trial["annual_return"]):
                    trial.update(weights=None, annual_return=None, reason="Nonfinite training score")
                trials[trial_id] = trial
                save()
            trial = trials[trial_id]
            fitted = matrix.reindex(columns=trial["columns"])
            weights = None if not trial["weights"] else np.array([trial["weights"][c] for c in fitted])
            return weights, fitted, None, ({"Annual Return": trial["annual_return"]} if weights is not None else None), {
                "selected_assets": len(fitted.columns), "timed_out": trial["timed_out"]}

        context = {"allocation": allocation, "prepare": prepare, "solve": solve}
        # No full-history checkpoint, current winning pair, or current holdings.
        # Training-only deterministic lower-bound start makes resume reproducible.
        search = {key: config[key] for key in (
            "minimum_cap", "maximum_cap", "step", "adaptive", "max_dd", "target_volatility",
            "drop_bottom_pct", "redundancy_corr_threshold", "exact_optimizer_asset_caps", "minimum_trading_days")}
        search.update(starting_cap=search["minimum_cap"], starting_asset_limit=min(search["exact_optimizer_asset_caps"]),
                      history_buffer_days=0, transaction_cost_bps=settings["cost_bps"],
                      solver_timeout_seconds=settings["solver_seconds"], prepared_matrix_store={}, _research=context)
        try:
            core.search_universal_shortlist_caps("research", **search)
        except ResearchTrialsComplete:
            pass
        valid = [r for r in trials.values() if r.get("weights") and r.get("annual_return") is not None
                 and r.get("train_sessions", 0) >= config["minimum_trading_days"]]
        if not valid:
            reasons = list(dict.fromkeys(str(r.get("reason")) for r in trials.values() if r.get("reason")))
            return {"status": "held", "reason": "No feasible training allocation; previous research mix held. " + "; ".join(reasons)[:250], "weights": dict(initial)}
        winner = max(valid, key=lambda r: (r["annual_return"], -r["pair"][0], -r["pair"][1]))
        return {"status": "selected", "weights": winner["weights"], "pair": winner["pair"]}
    return fit


class ResearchTrialsComplete(Exception):
    """The declared training search budget was consumed."""
