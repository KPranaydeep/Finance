"""Transparent, cost-aware comparison of saved portfolio alternatives."""
import math
import numpy as np
import pandas as pd
from datetime import timedelta
from nse_market_calendar import get_nse_equity_session


METRICS = {
    "Annual Return": 1,
    "Annual Volatility": -1,
    "Historical ES 95% (1 Session)": -1,
    "Block-Bootstrap ES 95% (20 Sessions)": -1,
    "Block-Bootstrap Median Return (20 Sessions)": 1,
    "Block-Bootstrap Tail Loss SD 95% (20 Sessions)": -1,
    "Sharpe Ratio": 1,
}


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def assess_rebalance(current, optimal, trades, base, cost_bps=22):
    """Compare all target metrics, with an explicit 20-session cost hurdle.

    The fee assumption includes spread/slippage only to the extent covered by
    cost_bps. No independent statistical confidence is inferred from point metrics.
    """
    rows = []
    for metric in dict.fromkeys([*METRICS, *current, *optimal]):
        before, after = _finite(current.get(metric)), _finite(optimal.get(metric))
        direction = METRICS.get(metric)
        change = None if before is None or after is None else after - before
        outcome = "Unavailable"
        if change is not None:
            outcome = "Unclassified" if direction is None else (
                "Unchanged" if abs(change) <= 1e-10 else
                "Improved" if change * direction > 0 else "Worsened"
            )
        rows.append({"Metric": metric, "Current": before, "Optimal": after,
                     "Change": change, "Assessment": outcome})
    base, cost_bps = _finite(base), _finite(cost_bps)
    valid_cost = base is not None and base > 0 and cost_bps is not None and cost_bps >= 0
    gross = buys = sells = 0.0
    for trade in trades:
        value = _finite(trade.get("Executable Value"))
        action = str(trade.get("Action", "")).casefold()
        if action not in {"buy", "sell"}:
            continue
        if value is None:
            valid_cost = False
            continue
        gross += abs(value)
        buys += abs(value) if action == "buy" else 0
        sells += abs(value) if action == "sell" else 0
    drag = gross / base * cost_bps / 10000 if valid_cost else None
    median_key = "Block-Bootstrap Median Return (20 Sessions)"
    before, after = _finite(current.get(median_key)), _finite(optimal.get(median_key))
    gain = after - before if before is not None and after is not None else None
    net_gain = gain - drag if gain is not None and drag is not None else None
    improved = sum(r["Assessment"] == "Improved" for r in rows)
    worsened = sum(r["Assessment"] == "Worsened" for r in rows)
    complete = all(r["Assessment"] not in {"Unavailable", "Unclassified"} for r in rows)
    risk_improves = any(r["Assessment"] == "Improved" and METRICS.get(r["Metric"]) == -1 for r in rows)
    if not trades:
        decision = "Hold — no executable trades"
    elif not complete or net_gain is None:
        decision = "Review — incomplete comparison or cost data"
    elif improved and not worsened and net_gain > 0:
        decision = "Rebalance candidate — target improves all-or-equal metrics and clears estimated costs"
    elif risk_improves and not worsened:
        decision = "Risk-reduction candidate — return benefit does not clear estimated costs"
    elif not improved or (not risk_improves and net_gain <= 0):
        decision = "Hold candidate — insufficient benefit after estimated costs"
    else:
        decision = "Review — return/risk trade-offs remain"
    return {"decision": decision, "metrics": rows, "gross_traded": gross,
            "gross_traded_fraction": gross / base if base and base > 0 else None,
            "two_way_turnover": min(buys, sells) / base if base and base > 0 else None,
            "cost_amount": gross * cost_bps / 10000 if valid_cost else None,
            "cost_fraction": drag, "median_gain": gain, "net_median_gain": net_gain}


def assess_executable_rebalance(log_returns, current_values, trades, cost_bps=22, cash=0.0, seed=430):
    """Paired historical review of feasible whole-order subsets against holding.

    Evaluate identical buy-and-hold 20-session windows for both alternatives.
    Block-bootstrap confidence bounds test average log-wealth improvement in
    each chronological third, with a family-wise candidate/segment correction.
    Orders and asset selection were chosen on full history: this is a stability
    check, not an independent out-of-sample backtest.
    """
    result = {"decision": "Hold / review — insufficient executable-history evidence", "candidates": [], "selected_orders": []}
    lr = pd.DataFrame(log_returns).replace([np.inf, -np.inf], np.nan)
    values = {str(k): _finite(v) for k, v in current_values.items()}
    if not values or any(v is None or v < 0 for v in values.values()):
        return result
    orders = [dict(t) for t in trades if str(t.get("Action", "")).lower() in {"buy", "sell"}]
    base = sum(values.values()) + cash
    if base <= 0 or _finite(cost_bps) is None or cost_bps < 0 or cash < 0:
        return result
    symbols = sorted({t for t, v in values.items() if v > 0} | {str(t.get("Yahoo Ticker", "")) for t in orders})
    if not symbols or any(t not in lr for t in symbols):
        result["decision"] = "Hold / review — missing history for owned or traded securities"
        return result
    lr = lr[symbols].dropna()
    if len(lr) < 252:
        return result
    for order in orders:
        if _finite(order.get("Executable Value")) is None or float(order["Executable Value"]) <= 0:
            return result
    if not orders:
        result["decision"] = "Hold — no executable trades"
        return result
    # Bounded candidates: full plan, priority prefixes, and individual sell/buy
    # pairs. Whole orders preserve existing lot/rounding decisions.
    ordered = sorted(orders, key=lambda t: _finite(t.get("Expected Annual Return Lift (%)")) or 0, reverse=True)
    subsets = [orders] + [ordered[:k] for k in (1, 2, 4, 8) if k < len(ordered)]
    sells = [t for t in ordered if str(t["Action"]).lower() == "sell"]
    buys = [t for t in ordered if str(t["Action"]).lower() == "buy"]
    subsets += [[s, b] for s in sells[:3] for b in buys[:3]]
    unique, seen = [], set()
    for subset in subsets:
        key = tuple(sorted((str(t["Yahoo Ticker"]), str(t["Action"]), float(t["Executable Value"])) for t in subset))
        if key not in seen:
            unique.append(subset)
            seen.add(key)
    horizon, repeats = 20, 1000
    alpha = .05 / max(1, len(unique) * 3)
    initial = np.array([values.get(t, 0) for t in symbols]) / base
    # No forward filling or synthetic flat prices; all alternatives share dates.
    terminal = np.expm1(lr.rolling(horizon).sum().dropna().to_numpy())
    hold = terminal @ initial
    segments = np.array_split(np.arange(len(terminal)), 3)
    rng = np.random.default_rng(seed)
    bootstrap_indices = []
    for segment in segments:
        n = len(segment)
        block = max(20, int(math.ceil(n ** (1 / 3))))
        starts = rng.integers(0, n, size=(repeats, math.ceil(n / block)))
        bootstrap_indices.append(((starts[..., None] + np.arange(block)) % n).reshape(repeats, -1)[:, :n])
    for subset in unique:
        weights = initial.copy()
        remaining_cash = cash / base
        gross = 0.0
        for order in subset:
            amount = float(order["Executable Value"]) / base
            sign = 1 if str(order["Action"]).lower() == "buy" else -1
            weights[symbols.index(str(order["Yahoo Ticker"]))] += sign * amount
            remaining_cash -= sign * amount
            gross += amount
        cost = gross * cost_bps / 10000
        record = {"Orders": len(subset), "Gross traded / base": gross, "Cost / base": cost}
        if weights.min() < -1e-9 or remaining_cash < 2 * cost - 1e-9:
            record.update({"Status": "Not funded at 2× cost stress", "Extra cash needed": max(0, (2 * cost - remaining_cash) * base)})
            result["candidates"].append(record)
            continue
        proposed = terminal @ weights - cost
        stressed = terminal @ weights - 2 * cost
        if np.any(1 + stressed <= 0) or np.any(1 + hold <= 0):
            record["Status"] = "Invalid wealth path"
            result["candidates"].append(record)
            continue
        difference = np.log1p(stressed) - np.log1p(hold)
        bounds = []
        for segment, indices in zip(segments, bootstrap_indices):
            means = difference[segment][indices].mean(axis=1)
            bounds.append(float(np.quantile(means, alpha)))
        tail_size = max(1, int(math.ceil(len(hold) * .05)))
        hold_es = -float(np.sort(hold)[:tail_size].mean())
        candidate_es = -float(np.sort(stressed)[:tail_size].mean())
        passed = min(bounds) > 0 and candidate_es <= hold_es + 1e-10
        record.update({"Net mean gain": float(np.mean(proposed - hold)), "Worst segment lower bound (log wealth)": min(bounds),
                       "Hold ES95": hold_es, "Post-trade ES95 (2× costs)": candidate_es,
                       "Status": "Passed stability check" if passed else "Hold — uncertain benefit or worse tail loss"})
        result["candidates"].append(record)
        if passed:
            prior = result.get("selected_gross", float("inf"))
            if gross < prior:
                result.update({"selected_orders": subset, "selected_gross": gross, "decision": "Lower-turnover candidate passes paired stability and cost-stress checks"})
    if not result["selected_orders"]:
        result["decision"] = "Hold / review — no funded tested subset shows supported improvement after cost stress"
    result["observations"] = len(lr)
    result["as_of"] = str(lr.index.max())[:10]
    result["confidence_note"] = "95% family-wise historical bootstrap check across tested candidates and three chronological segments; full-history selection bias remains."
    return result


def estimate_review_window(log_returns, current_values, target_weights, cost_bps=22,
                           risk_free_rate=.08225, horizon=20, seed=431):
    """Estimate a bounded review window from all seven portfolio metrics.

    Joint stationary bootstrap paths preserve cross-asset dependence. Compare
    drifted holdings against their initial 95% simulation bands and the target.
    A breach calls for review, not an order. The existing scenario horizon is
    the evidence expiry, not a claim of an optimal or safe holding period.
    """
    lr = pd.DataFrame(log_returns).replace([np.inf, -np.inf], np.nan)
    values = {str(t): _finite(v) for t, v in current_values.items()}
    targets = {str(t): _finite(w) for t, w in target_weights.items()}
    symbols = sorted({t for t,v in values.items() if v is not None and v > 0} | {t for t,w in targets.items() if w is not None and w > 0})
    if not symbols or any(t not in lr for t in symbols) or any(v is None or v < 0 for v in values.values()):
        return {"sessions": 0, "reason": "missing portfolio history or valuation", "metrics": []}
    matrix = lr[symbols].dropna()
    if len(matrix) < 252:
        return {"sessions": 0, "reason": "insufficient common history", "metrics": []}
    base = sum(values.values())
    target = np.array([targets.get(t, 0) or 0 for t in symbols], dtype=float)
    if base <= 0 or target.sum() <= 0:
        return {"sessions": 0, "reason": "incomplete current or target allocation", "metrics": []}
    current = np.array([values.get(t, 0) or 0 for t in symbols]) / base
    target /= target.sum()
    groups, group_size = 32, 128
    rng = np.random.default_rng(seed)
    n = len(matrix)
    block = max(2, int(math.ceil(n ** (1/3))))
    indices = rng.integers(n, size=groups * group_size)
    data = matrix.to_numpy()
    cumulative = [np.zeros((groups * group_size, len(symbols)))]
    for _ in range(2 * horizon):
        cumulative.append(cumulative[-1] + data[indices])
        indices = np.where(rng.random(len(indices)) < 1 / block, rng.integers(n, size=len(indices)), (indices + 1) % n)
    cumulative = np.asarray(cumulative)

    def metrics(terminal, one_day):
        returns = terminal.reshape(groups, group_size)
        daily = one_day.reshape(groups, group_size)
        logs = np.log1p(np.maximum(returns, -.999999))
        annual = np.expm1(logs.mean(axis=1) * 252 / horizon)
        vol = logs.std(axis=1) * np.sqrt(252 / horizon)
        tail = np.sort(returns, axis=1)[:, :max(1, math.ceil(group_size * .05))]
        day_tail = np.sort(daily, axis=1)[:, :max(1, math.ceil(group_size * .05))]
        return np.array([annual, vol, -day_tail.mean(axis=1), -tail.mean(axis=1),
                         np.median(returns, axis=1), tail.std(axis=1),
                         np.divide(annual - risk_free_rate, vol, out=np.zeros_like(vol), where=vol>0)])

    initial_end = np.expm1(cumulative[horizon]) @ current
    initial_day = np.expm1(cumulative[1]) @ current
    reference = metrics(initial_end, initial_day)
    names = list(METRICS)
    # Bonferroni simulation bands cover the seven metrics and tested horizons.
    alpha = .05 / (len(names) * horizon)
    bands = [np.quantile(row, [alpha/2, 1-alpha/2]) for row in reference]
    timeline = []
    chosen = None
    for wait in range(1, horizon + 1):
        drift = current * np.exp(cumulative[wait])
        drift /= drift.sum(axis=1, keepdims=True)
        future = np.expm1(cumulative[wait+horizon] - cumulative[wait])
        day = np.expm1(cumulative[wait+1] - cumulative[wait])
        hold_metrics = metrics((future * drift).sum(axis=1), (day * drift).sum(axis=1))
        target_metrics = metrics(future @ target, day @ target)
        transaction_drag = float(np.median(np.abs(drift-target).sum(axis=1))) * cost_bps / 10000 * 2
        breaches = []
        for i, (name, direction) in enumerate(METRICS.items()):
            projected = float(np.median(hold_metrics[i]))
            comparator = float(np.median(target_metrics[i]))
            boundary = bands[i][0] if direction > 0 else bands[i][1]
            deteriorated = direction * (projected - boundary) < 0
            target_helps = direction * (comparator - projected) > 0
            if deteriorated and target_helps:
                breaches.append(name)
        median_net_gain = float(np.median(target_metrics[4] - hold_metrics[4])) - transaction_drag
        # Review a risk/quality breach only with a target that also clears the
        # doubled-cost median-return hurdle; mixed trade-offs expire at horizon.
        trigger = bool(breaches) and median_net_gain > 0
        timeline.append({"Sessions ahead": wait, "Metrics outside initial simulation bands": ", ".join(breaches) or "None", "Target median gain after 2× costs": median_net_gain})
        if trigger and chosen is None:
            chosen = {"sessions": wait, "reason": "projected deterioration: " + ", ".join(breaches)}
    result = chosen or {"sessions": horizon, "reason": "scenario evidence window expires; no earlier joint metric/cost trigger"}
    result.update({"as_of": str(matrix.index.max())[:10], "metrics": names, "timeline": timeline,
                   "method": "Joint historical bootstrap review window; fixed target, 32 simulation groups × 128 paths. Simulation bands are not calibrated forecast confidence intervals."})
    return result


def review_market_date(as_of, sessions, today):
    """Map evidence age to verified NSE sessions; unsupported years stay explicit."""
    from datetime import date
    anchor = date.fromisoformat(str(as_of)[:10])
    due = anchor
    remaining = max(0, int(sessions))
    try:
        while remaining:
            due += timedelta(days=1)
            if get_nse_equity_session(due)["is_open"]:
                remaining -= 1
        if due <= today:
            due = today
            while not get_nse_equity_session(due)["is_open"]:
                due += timedelta(days=1)
            return {"date": str(due), "sessions_remaining": 0, "due_now": True, "calendar": "NSE equity"}
        count, cursor = 0, today
        while cursor < due:
            cursor += timedelta(days=1)
            count += int(get_nse_equity_session(cursor)["is_open"])
        return {"date": str(due), "sessions_remaining": count, "due_now": False, "calendar": "NSE equity"}
    except ValueError as exc:
        return {"date": None, "sessions_remaining": None, "calendar_error": str(exc)}
