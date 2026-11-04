"""Rebuild private execution quantities at saved weights, without any optimizer."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
import pandas as pd

from rebalance_review import assess_executable_rebalance, estimate_review_window
from portfolio_optimizer_config import RISK_FREE_RATE_ANNUAL


def saved_targets(payload):
    weights = {}
    for row in payload.get("optimal_allocation") or []:
        ticker = str(row.get("ticker", "")).strip().upper()
        weight = float(row.get("target_weight", 0))
        if not ticker or ticker in weights or not math.isfinite(weight) or weight < 0:
            raise ValueError("Saved optimal allocation is invalid")
        weights[ticker] = weight
    if not weights or abs(sum(weights.values()) - 1) > 1e-6:
        raise ValueError("Complete saved optimal weights are required; run optimization once")
    return {t: w for t, w in weights.items() if w > 0}


def target_fingerprint(payload):
    return hashlib.sha256(json.dumps(saved_targets(payload), sort_keys=True).encode()).hexdigest()


def holdings_fingerprint(frame):
    owned = {}
    for row in pd.DataFrame(frame).to_dict(orient="records"):
        quantity = float(row.get("Quantity", 0) or 0)
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("Invalid holding quantity")
        if quantity == 0:
            continue
        ticker = str(row.get("Yahoo Ticker", "")).strip().upper()
        currency = str(row.get("Currency", "")).strip().upper()
        average = float(row.get("Average Price", 0) or 0)
        key = ticker + "|" + currency
        previous = owned.setdefault(key, [0., 0.])
        previous[0] += quantity
        previous[1] += quantity * average
    return hashlib.sha256(json.dumps(owned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_fixed_universe(payload, current_holdings):
    """Fresh holdings are authoritative; old quantities are never inherited."""
    targets = saved_targets(payload)
    rows = pd.DataFrame(current_holdings).to_dict(orient="records")
    metadata = {str(r.get("Yahoo Ticker", "")).upper(): r
                for r in payload.get("current_allocation") or []}
    owned = {}
    for row in rows:
        qty = float(row.get("Quantity", 0) or 0)
        if not math.isfinite(qty) or qty < 0:
            raise ValueError("Invalid current holding quantity")
        if qty == 0:
            continue
        ticker = str(row.get("Yahoo Ticker", "")).strip().upper()
        if not ticker or not row.get("Currency"):
            raise ValueError("Every owned security needs a verified ticker and currency")
        row = dict(row, **{"Yahoo Ticker": ticker})
        if ticker in owned:
            prior = owned[ticker]
            if str(prior["Currency"]).upper() != str(row["Currency"]).upper():
                raise ValueError("Conflicting currencies for one security")
            total = float(prior["Quantity"]) + qty
            prior["Average Price"] = (float(prior.get("Average Price", 0)) * prior["Quantity"]
                                       + float(row.get("Average Price", 0)) * qty) / total
            prior["Quantity"] = total
        else:
            owned[ticker] = row
    for ticker in targets:
        if ticker not in owned:
            source = metadata.get(ticker)
            if not source or not source.get("Currency"):
                raise ValueError("Saved listing metadata unavailable for " + ticker)
            owned[ticker] = dict(source, **{"Yahoo Ticker": ticker, "Quantity": 0., "Average Price": 0.})
    return pd.DataFrame(owned.values()), targets


def rebuild_saved_plan(payload, current_holdings, log_returns, native_prices, fx_rates,
                       *, plan_builder, stats_builder, paused=(), now=None):
    """Return a complete candidate; failures leave the input saved payload intact."""
    allocation, targets = prepare_fixed_universe(payload, current_holdings)
    tickers = allocation["Yahoo Ticker"].tolist()
    for column, values in (("Latest Price", native_prices), ("FX to INR", fx_rates)):
        keys = allocation["Currency"] if column == "FX to INR" else allocation["Yahoo Ticker"]
        allocation[column] = pd.to_numeric(keys.map(values), errors="coerce")
        if not np.isfinite(allocation[column]).all() or (allocation[column] <= 0).any():
            raise ValueError("Missing or invalid current " + column + "; previous plan retained")
    allocation["Latest Price INR"] = allocation["Latest Price"] * allocation["FX to INR"]
    allocation["Current Value INR"] = allocation["Quantity"] * allocation["Latest Price INR"]
    allocation["Invested INR"] = allocation["Quantity"] * allocation["Average Price"] * allocation["FX to INR"]
    base = float(allocation["Current Value INR"].sum())
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Positive current holdings value is required")
    allocation["Weight"] = allocation["Current Value INR"] / base
    returns = pd.DataFrame(log_returns).reindex(columns=tickers)
    weights = np.array([targets.get(t, 0.) for t in tickers])
    cost = float(((payload.get("settings") or {}).get("security_scenario_forecast") or {}).get("one_way_cost_bps", 22.))
    days = int((payload.get("settings") or {}).get("days_to_flip", 1))
    plan, missing_prices, missing_alloc = plan_builder(
        allocation, weights, returns, native_prices, days,
        transaction_cost_bps=cost, excluded_trade_tickers=paused,
    )
    if missing_prices or missing_alloc:
        raise ValueError("Incomplete rebalance coverage; previous plan retained")
    candidate = deepcopy(payload)
    owned_values = dict(zip(allocation["Yahoo Ticker"], allocation["Current Value INR"]))
    orders = plan.to_dict(orient="records")
    common = returns.replace([np.inf, -np.inf], np.nan).dropna()
    candidate.update(
        current_allocation=allocation.to_dict(orient="records"),
        rebalancing_plan=orders, rebalance_review_orders=orders,
        executable_trade_count=len(plan), holdings_analyzed=int(allocation["Quantity"].gt(0).sum()),
        total_invested=float(allocation["Invested INR"].sum()), current_portfolio_value=base,
        partial_sell_sharpe_check=plan.attrs.get("partial_sell_sharpe_check", {}),
        executable_rebalance_review=assess_executable_rebalance(common, owned_values, orders, cost_bps=cost),
        next_review_estimate=estimate_review_window(common, owned_values, targets, cost_bps=cost,
                                                  risk_free_rate=RISK_FREE_RATE_ANNUAL),
    )
    if len(common) >= 126:
        candidate.update(
            current_stats=stats_builder(allocation["Weight"].to_numpy(), common),
            optimal_stats=stats_builder(weights, common),
            portfolio_metrics_as_of=str(common.index[-1])[:10], portfolio_metrics_refresh_error=None,
            leverage_return_history={"dates": [str(d) for d in common.index], "portfolio_log_returns": (common @ weights).tolist()},
            leverage_assessment=None, leverage_assessment_error="Plan and current holdings refreshed; older detailed debt scenarios invalidated.",
        )
    else:
        candidate["portfolio_metrics_refresh_error"] = "Plan resized; insufficient common history to refresh all portfolio metrics. Earlier dated metrics retained."
    now = now or datetime.now(timezone.utc)
    candidate["rebalance_plan_context"] = {
        "updated_at": now.isoformat(), "weights_saved_at": payload.get("saved_at"),
        "weights_fingerprint": target_fingerprint(payload),
        "holdings_fingerprint": holdings_fingerprint(current_holdings),
        "holdings_source": "current master holdings", "optimizer_run": False,
        "history_through": str(common.index[-1])[:10] if len(common) else None,
    }
    return candidate, allocation, plan
