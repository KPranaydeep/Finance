"""Conditional joint stationary-block simulations, not market-timing promises."""
import numpy as np
from .costs import tax_rate
from .core import digest

METHOD = "joint-security-first-passage-v2"


def paths(returns, days, count, block, seed):
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or len(values) < 126 or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Insufficient/invalid common return history")
    rng = np.random.default_rng(seed)
    index = rng.integers(len(values), size=count)
    out = np.empty((count, days, values.shape[1]))
    for d in range(days):
        if d:
            restart = rng.random(count) < 1 / block
            index = np.where(restart, rng.integers(len(values), size=count), (index + 1) % len(values))
        out[:, d] = values[index]
    return out


def estimate(baseline, prices, returns, future_dates, policy, peak, validation=None, count=None, dividends=None):
    if not future_dates:
        raise ValueError("Future exits must follow entry")
    lots = baseline["lots"]
    tickers = [r["ticker"] for r in lots]
    matrix = returns[tickers].to_numpy(dtype=float)
    count = count or policy["simulation_paths"]
    shocks = paths(matrix, len(future_dates), count, policy["block_length"], policy["seed"])
    start = np.array([prices[t] for t in tickers])
    future = start * np.cumprod(1 + shocks, axis=1)
    q = np.array([r["quantity"] for r in lots])
    cost = np.array([r["price"] for r in lots])
    gross = future * q
    # Conservative analytic envelope for simulations; exact rupee/paise-rounded
    # liquidation engine is always used for today's actionable trigger.
    brokerage = np.minimum(20., np.minimum(np.maximum(5., gross * .001), gross * .025))
    dp = np.where(gross >= 100, 20., 13.)
    stt = np.array([.001 if r["kind"] == "equity" else .00001 if r["kind"] == "equity_etf" else 0. for r in lots])
    regulated = gross * (.0000297 + .000001 + .000001)
    fees = ((brokerage + dp + regulated) * 1.18 + gross * (stt + policy["slippage_bps"] / 10000) + 1.25)
    rates = np.array([[tax_rate(r["kind"], r["entry_date"], d, policy) for r in lots] for d in future_dates])
    tax = np.maximum(0., (future - cost) * q) * rates
    net = baseline["cash"] + (gross - fees - tax).sum(axis=2)
    total = baseline["cash"] + gross.sum(axis=2)
    weights = gross / total[:, :, None]
    from datetime import date
    days = np.array([(date.fromisoformat(d) - date.fromisoformat(baseline["entry_date"])).days for d in future_dates])
    if not len(days) or np.any(days <= 0):
        raise ValueError("Future exits must follow entry")
    # NPV at the target rate >= 0 is equivalent to crossing target XIRR
    # for conventional cash flows. Include entry charges and dated net dividends.
    outlays = q * cost + np.array([r["entry_charges"]["total"] for r in lots])
    dividend_pv = np.array([sum(
        x["net"] / (1 + policy["target_xirr"]) ** (
            (date.fromisoformat(x["date"]) - date.fromisoformat(r["entry_date"])).days / 365)
        for x in (dividends or []) if x["ticker"] == r["ticker"]) for r in lots])
    security_target = (outlays - dividend_pv)[None, :] * np.power(
        1 + policy["target_xirr"], days[:, None] / 365)
    security_profit = gross - fees - tax >= security_target[None, :, :]
    security_probability = np.maximum.accumulate(security_profit, axis=1).mean(axis=0)
    security_crossings = []
    for j, ticker in enumerate(tickers):
        hits = np.flatnonzero(security_probability[:, j] >= policy["crossing_probability"])
        index = int(hits[0]) if len(hits) else None
        security_crossings.append({
            "ticker": ticker, "crossing_date": future_dates[index] if index is not None else None,
            "probability": float(security_probability[index, j]) if index is not None else None,
            "horizon_probability": float(security_probability[-1, j])})
    earliest_security = min((r["crossing_date"] for r in security_crossings if r["crossing_date"]), default=None)
    target = baseline["capital"] * np.power(1 + policy["target_xirr"], days / 365)
    profit = net >= target
    running_peak = np.maximum.accumulate(np.maximum(net, peak), axis=1)
    risk = ((net / running_peak - 1 <= -policy["drawdown_limit"]) |
            (weights.max(axis=2) > policy["concentration_limit"]))
    target_weights = np.array([baseline["weights"].get(t, 0.) for t in tickers])
    drift = np.max(np.abs(weights - target_weights), axis=2) >= policy["drift_limit"]
    crossing = np.maximum.accumulate(profit | security_profit.any(axis=2) | risk | drift, axis=1)
    probability = crossing.mean(axis=0)
    # Select an individually qualifying security, not a union of weak chances
    # across many securities. Risk/basket triggers may require earlier review.
    other_probability = np.maximum.accumulate(profit | risk | drift, axis=1).mean(axis=0)
    hit = np.flatnonzero(other_probability >= policy["crossing_probability"])
    # Review one session before the first probability-limit breach, never before
    # tomorrow. No crossing -> bounded monitoring horizon, not "never".
    offset = max(0, int(hit[0]) - 1) if len(hit) else len(future_dates) - 1
    candidate = future_dates[offset]
    if earliest_security:
        candidate = min(candidate, earliest_security)
    approved = bool(validation and validation.get("passed") and
                    validation.get("policy_hash") == digest(policy) and
                    validation.get("tickers") == tickers and validation.get("method") == METHOD)
    return {"method": METHOD,
            "status": "WALK_FORWARD_CHECKS_PASSED_EXPERIMENTAL" if approved else "RESEARCH_ONLY",
            "next_review": candidate if approved else None, "research_candidate": candidate,
            "earliest_security_crossing": earliest_security,
            "trigger_securities": [r["ticker"] for r in security_crossings if earliest_security and r["crossing_date"] == earliest_security],
            "security_crossings": security_crossings, "target_xirr": policy["target_xirr"],
            "crossing_probability_threshold": policy["crossing_probability"],
            "never_crossed_fraction": float(1 - probability[-1]),
            "paths": count, "common_returns": len(matrix), "seed": policy["seed"],
            "curve": [{"date": d, "any_review_probability": float(probability[i]),
                       "profit_crossing_probability": float(np.maximum.accumulate(profit, axis=1)[:, i].mean()),
                       "downside_or_concentration_probability": float(np.maximum.accumulate(risk, axis=1)[:, i].mean()),
                       "median_net_value": float(np.median(net[:, i]))} for i, d in enumerate(future_dates)],
            "limitations": "Conditional on today's selected basket; no selection-alpha validation. Price-return resampling, no forecast dividends, jumps/regime changes can be missed. Future charges/taxes held to configured rules."}


def validate(returns, baseline, prices, dates, policy):
    """Non-overlapping walk-forward score of boundary forecasts vs weekly review.

    Uses synthetic equal-scale historical entry lots, NOT a backtest of published
    investment performance. Every fold trains only on earlier rows. Dates/target
    holding age are shifted together. Passing is an empirical gate, not a proof.
    """
    from copy import deepcopy
    from datetime import date, timedelta
    from .core import evaluate
    horizon, train = policy["validation_horizon"], policy["validation_train"]
    values = returns[[r["ticker"] for r in baseline["lots"]]]
    folds = []
    age = max(1, (date.fromisoformat(dates[-1]) - date.fromisoformat(baseline["entry_date"])).days)
    for cut in range(train, len(values) - horizon + 1, horizon):
        b = deepcopy(baseline)
        # Translate historical scenario calendar to supported contemporary tax
        # dates. No future returns enter the resampling distribution.
        anchor = date.fromisoformat(dates[-1])
        # Keep today's holding age. Use a future synthetic anchor only if needed
        # to stay inside the explicitly supported tax regime.
        anchor = max(anchor, date(2026, 4, 1) + timedelta(days=age))
        b["entry_date"] = str(anchor - timedelta(days=age))
        for lot in b["lots"]:
            lot["entry_date"] = b["entry_date"]
        origin = date.fromisoformat(dates[cut - 1])
        future_dates = [str(anchor + (date.fromisoformat(dates[cut + i]) - origin)) for i in range(horizon)]
        f = estimate(b, prices, values.iloc[:cut], future_dates, policy, b["capital"], count=200)
        actual_prices = dict(prices)
        hit_day, peak = None, b["capital"]
        for i in range(horizon):
            for t in actual_prices:
                actual_prices[t] *= 1 + float(values.iloc[cut + i][t])
            m = evaluate(b, actual_prices, future_dates[i], policy)
            peak = max(peak, m["net_proceeds"])
            w = {r["ticker"]: r["gross"] / m["gross_value"] for r in m["rows"]}
            if ((m["xirr"] is not None and m["xirr"] >= policy["target_xirr"]) or
                any(r["xirr"] is not None and r["xirr"] >= policy["target_xirr"] for r in m["rows"]) or
                m["net_proceeds"] / peak - 1 <= -policy["drawdown_limit"] or
                max(w.values()) > policy["concentration_limit"] or
                max(abs(w.get(t, 0) - b["weights"].get(t, 0)) for t in w) >= policy["drift_limit"]):
                hit_day = i + 1
                break
        review = future_dates.index(f["research_candidate"]) + 1
        occurred = hit_day is not None
        p = f["curve"][-1]["any_review_probability"]
        folds.append({"train_end_row": cut - 1, "test_start_row": cut, "test_end_row": cut + horizon - 1,
                      "event": occurred, "probability": p, "brier": (p - occurred) ** 2,
                      "adaptive_late": bool(occurred and review > hit_day),
                      "weekly_late": bool(occurred and 5 > hit_day),
                      "monthly_late": bool(occurred and 20 > hit_day),
                      "adaptive_unnecessary": bool(not occurred and review < horizon)})
    n = len(folds)
    brier = sum(f["brier"] for f in folds) / n if n else None
    events = sum(f["event"] for f in folds)
    late = sum(f["adaptive_late"] for f in folds)
    weekly = sum(f["weekly_late"] for f in folds)
    # Require both event and non-event coverage and a nontrivial Brier threshold.
    passed = (n >= policy["validation_min_folds"] and events >= 5 and n - events >= 5 and
              brier <= policy["validation_max_brier"] and late <= weekly)
    return {"passed": bool(passed), "policy_hash": digest(policy), "method": METHOD,
            "tickers": list(values.columns), "folds": n, "events": events, "brier": brier,
            "adaptive_late": late, "weekly_late": weekly,
            "monthly_late": sum(f["monthly_late"] for f in folds),
            "adaptive_unnecessary": sum(f["adaptive_unnecessary"] for f in folds),
            "detail": folds, "scope": "Conditional review timing only; not tax, alpha, or net-strategy-performance validation"}
