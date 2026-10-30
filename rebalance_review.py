"""Transparent, cost-aware comparison of saved portfolio alternatives."""
import math


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
