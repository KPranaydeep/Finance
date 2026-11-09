"""Final allocation checks. No downloads, optimizer calls or database writes."""

import math

import numpy as np
import pandas as pd

from trade_quantities import plan_quantity, quantity_step
from risk_aware_allocator import check_policy_weights


VERSION = "allocation-integrity-v1"
TOLERANCE = 1e-7


def allocation_check(weights, log_returns, *, bounds=None, max_drawdown=None,
                     target_volatility=None, covariance=None, policy_limits=None):
    """Check final weights against the requested limits, including the initial peak.

    Drawdown replays daily-rebalanced simple returns from the input log returns.
    Volatility follows the optimizer's daily log-return covariance convention.
    Cash may be present in an executable mix, whose risky weights sum below one.
    """
    errors = []
    w = np.asarray(weights, dtype=float)
    history = pd.DataFrame(log_returns).apply(pd.to_numeric, errors="coerce")
    report = {"version": VERSION, "passed": False, "violations": errors}
    if (w.ndim != 1 or len(w) != history.shape[1] or not len(w)
            or not np.isfinite(w).all() or not np.isfinite(history.to_numpy()).all()
            or len(history) < 2):
        errors.append("Incomplete or non-finite allocation/history")
        return report
    if w.min() < -TOLERANCE or w.sum() > 1 + TOLERANCE:
        errors.append("Negative exposure or total exposure above available capital")
    if bounds is not None:
        if len(bounds) != len(w):
            errors.append("Allocation bounds do not cover every security")
        else:
            for i, (lower, upper) in enumerate(bounds):
                if not float(lower) - TOLERANCE <= w[i] <= float(upper) + TOLERANCE:
                    errors.append(f"{history.columns[i]} violates its position/frozen-weight bounds")
    simple = np.expm1(history.to_numpy(dtype=float)) @ w
    if not np.isfinite(simple).all() or (simple <= -1).any():
        errors.append("Invalid portfolio return path")
        return report
    log_wealth = np.r_[0., np.cumsum(np.log1p(simple))]
    max_log_loss = float(np.max(np.maximum.accumulate(log_wealth) - log_wealth))
    drawdown = float(-np.expm1(-max_log_loss))
    report.update(weight_sum=float(w.sum()), historical_drawdown=drawdown)
    if max_drawdown is not None:
        limit = float(max_drawdown)
        if not math.isfinite(limit) or not 0 <= limit <= 1:
            errors.append("Invalid drawdown limit")
        elif drawdown > limit + TOLERANCE:
            errors.append("Final allocation exceeds the requested historical drawdown limit")
        report["drawdown_limit"] = limit
    if target_volatility is not None:
        cov = np.asarray(covariance, dtype=float)
        limit = float(target_volatility)
        if (cov.shape != (len(w), len(w)) or not np.isfinite(cov).all()
                or not math.isfinite(limit) or limit < 0):
            errors.append("Invalid volatility evidence or limit")
        else:
            variance = float(w @ cov @ w)
            if variance < -TOLERANCE:
                errors.append("Invalid covariance variance")
            volatility = math.sqrt(max(variance, 0.))
            report.update(daily_volatility=volatility, daily_volatility_limit=limit)
            if volatility > limit + TOLERANCE:
                errors.append("Final allocation exceeds the requested volatility limit")
    if policy_limits:
        policy = check_policy_weights(w, history, policy_limits)
        errors.extend(policy["violations"])
        report["allocation_policy"] = policy
    report["passed"] = not errors
    return report


def finalize_weights(raw_weights, cleaned_weights, log_returns, **limits):
    """Use cleanup only when it preserves feasibility; never repair by guessing."""
    raw = np.asarray(raw_weights, dtype=float)
    if (raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or raw.min() < -TOLERANCE
            or not np.isclose(raw.sum(), 1., atol=TOLERANCE, rtol=0)):
        return None, {"version": VERSION, "passed": False,
                      "violations": ["Invalid or incomplete raw optimizer allocation"]}
    checks = []
    for label, weights in (("cleaned", cleaned_weights), ("raw", raw_weights)):
        check = allocation_check(weights, log_returns, **limits)
        if not np.isclose(np.asarray(weights).sum(), 1., atol=TOLERANCE, rtol=0):
            check["violations"].append("Target weights do not sum to one")
            check["passed"] = False
        check["weights_used"] = label
        checks.append(check)
        if check["passed"]:
            if label == "raw":
                check["cleanup_skipped"] = checks[0]["violations"]
            return np.asarray(weights, dtype=float).copy(), check
    checks[-1]["cleanup_violations"] = checks[0]["violations"]
    return None, checks[-1]


def fund_buy_orders(orders, holdings, *, cost_bps, available_cash_inr=0.):
    """Round buys down to recorded cash plus net sale proceeds; never assume credit.

    Preserve the existing order priority. A sale must settle before its proceeds
    can fund a buy; this arithmetic check does not claim broker settlement.
    """
    orders = pd.DataFrame(orders).copy()
    cash, rate = float(available_cash_inr), float(cost_bps) / 10000.
    if not math.isfinite(cash) or cash < 0 or not 0 <= rate < 1:
        raise ValueError("Invalid recorded cash or transaction-cost assumption")
    held = pd.DataFrame(holdings).set_index("Yahoo Ticker")
    adjustments = []
    if orders.empty:
        return orders, adjustments
    cash += float(orders.loc[orders["Action"].eq("Sell"), "Executable Value"].sum()) * (1 - rate)
    for index, row in orders.loc[orders["Action"].eq("Buy")].iterrows():
        ticker = str(row["Yahoo Ticker"])
        price = float(row["Native Price"]) * float(row["FX to INR"])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid order price or FX")
        requested = float(row["Executable Quantity"])
        affordable = max(cash, 0.) / (price * (1 + rate))
        quantity = requested
        if requested > affordable + 1e-10:
            quantity = plan_quantity(affordable, float(held.loc[ticker, "Quantity"]),
                                     row["Currency"], "Buy", 1)
            quantity = min(quantity, requested)
            adjustments.append({"ticker": ticker, "requested_quantity": requested,
                                "funded_quantity": quantity})
        value = quantity * price
        orders.loc[index, ["Executable Quantity", "Executable Value"]] = [quantity, value]
        cash -= value * (1 + rate)
    return orders.loc[orders["Executable Quantity"].gt(0)].copy(), adjustments


def execution_check(holdings, orders, prices, log_returns, *, cost_bps,
                    available_cash_inr=0., excluded_tickers=(), frozen_weights=None,
                    max_weight=1., max_drawdown=None, target_volatility=None,
                    covariance=None, policy_limits=None):
    """Replay every marked holding, cash, fees and order; unavailable is not passed."""
    report = {"version": VERSION, "passed": False, "violations": []}
    errors = report["violations"]
    try:
        frame = pd.DataFrame(holdings).copy()
        frame["Yahoo Ticker"] = frame["Yahoo Ticker"].astype(str).str.strip().str.upper()
        if frame["Yahoo Ticker"].duplicated().any():
            raise ValueError("Duplicate holding tickers require consolidation")
        quantities, marked, steps = {}, {}, {}
        for row in frame.to_dict(orient="records"):
            ticker, quantity = row["Yahoo Ticker"], float(row["Quantity"])
            if not math.isfinite(quantity) or quantity < 0:
                raise ValueError("Invalid holding quantity")
            quantities[ticker] = quantity
            steps[ticker] = quantity_step(quantity, row["Currency"])
            if quantity == 0 and ticker not in prices:
                continue
            price = float(prices[ticker]) * float(row["FX to INR"])
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Missing positive current price/FX for " + ticker)
            marked[ticker] = price
        cash, rate = float(available_cash_inr), float(cost_bps) / 10000.
        if not math.isfinite(cash) or cash < 0 or not 0 <= rate < 1:
            raise ValueError("Invalid cash or cost assumption")
        if not math.isfinite(float(max_weight)) or not 0 < float(max_weight) <= 1:
            raise ValueError("Invalid position weight cap")
        initial = {t: q * marked[t] for t, q in quantities.items() if q > 0}
        base = sum(initial.values()) + cash
        if not math.isfinite(base) or base <= 0:
            raise ValueError("No positive marked portfolio capital")
        blocked = {str(t).strip().upper() for t in excluded_tickers}
        gross = 0.
        seen = set()
        for row in pd.DataFrame(orders).to_dict(orient="records"):
            ticker = str(row["Yahoo Ticker"]).strip().upper()
            action, quantity = row["Action"], float(row["Executable Quantity"])
            if ticker in seen or ticker in blocked or ticker not in quantities:
                raise ValueError("Duplicate, paused or unknown order ticker: " + ticker)
            seen.add(ticker)
            if action not in {"Buy", "Sell"} or not math.isfinite(quantity) or quantity < 0:
                raise ValueError("Invalid order action/quantity")
            step = steps[ticker]
            if not math.isclose(quantity / step, round(quantity / step), abs_tol=1e-6, rel_tol=0):
                raise ValueError("Order violates recorded quantity precision: " + ticker)
            value = quantity * marked[ticker]
            if not math.isclose(float(row["Executable Value"]), value, rel_tol=1e-8, abs_tol=1e-6):
                raise ValueError("Order value does not match quantity × price × FX")
            direction = 1 if action == "Buy" else -1
            quantities[ticker] += direction * quantity
            cash -= direction * value + rate * value
            gross += value
            if quantities[ticker] < -TOLERANCE:
                errors.append("Order sells more than owned: " + ticker)
        if cash < -max(base * 1e-10, 1e-6):
            errors.append("Orders require unrecorded cash or borrowing after costs")
        net_capital = base - rate * gross
        if net_capital <= 0:
            raise ValueError("Costs exhaust portfolio capital")
        values = {t: max(q, 0.) * marked[t] for t, q in quantities.items() if q > 0}
        history = pd.DataFrame(log_returns)
        missing = sorted(set(values) - set(history.columns))
        if missing:
            errors.append("Missing final holding history: " + ", ".join(missing))
        for ticker, value in values.items():
            if value / net_capital > float(max_weight) + TOLERANCE:
                errors.append("Final position exceeds its weight cap: " + ticker)
        for ticker in (frozen_weights or {}):
            # Fees/other paused positions must not redefine a frozen share count.
            if not math.isclose(values.get(ticker, 0.), initial.get(ticker, 0.),
                                abs_tol=1e-6, rel_tol=1e-8):
                errors.append("Frozen position changed: " + ticker)
        report.update(excluded_tickers=sorted(blocked),
                      cash_after_costs_inr=cash, estimated_cost_inr=rate * gross,
                      marked_capital_inr=base, final_weights={t: v / net_capital for t, v in values.items()},
                      settlement_assumption="Sale proceeds available before buys")
        if not missing:
            w = np.array([values.get(t, 0.) / net_capital for t in history.columns])
            risk = allocation_check(w, history, max_drawdown=max_drawdown,
                                    target_volatility=target_volatility, covariance=covariance)
            errors.extend(risk["violations"])
            report["risk"] = risk
            if policy_limits:
                policy_check = check_policy_weights(
                    w, history, policy_limits,
                    scenario_weights=w * net_capital / base,
                    cost_fraction=rate * gross / base * policy_limits["settings"]["cost_stress_multiple"],
                )
                errors.extend(policy_check["violations"])
                report["allocation_policy"] = policy_check
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        errors.append(str(exc))
    report["passed"] = not errors
    return report
