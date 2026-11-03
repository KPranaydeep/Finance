"""Cheap portfolio-wide gate for partial trims; no optimizer or market downloads."""

import math

import numpy as np
import pandas as pd
from portfolio_optimizer_config import TRADING_DAYS_PER_YEAR


def review_partial_sells(log_returns, current_values, orders, *, cost_bps,
                         risk_free_rate, trading_days=TRADING_DAYS_PER_YEAR, horizon=30):
    """Compare the rounded proposal with holding and with the same plan sans trims.

    Uses the app's annualized log-return Sharpe convention. One-time order costs
    are amortized over the review horizon as a return hurdle. Residual cash earns
    zero. This eligibility check is historical, not statistical confidence or a
    replacement for the existing funded-order/stability review.
    """
    result = {"allowed": False, "reason": "Incomplete portfolio history or sizing",
              "current_sharpe": None, "proposed_net_sharpe": None,
              "without_trims_net_sharpe": None, "horizon": horizon,
              "cost_bps": cost_bps}
    try:
        values = {str(t): float(v) for t, v in current_values.items()}
        if (not values or any(not math.isfinite(v) or v < 0 for v in values.values())
                or not all(math.isfinite(v) for v in (cost_bps, risk_free_rate, trading_days, horizon))
                or cost_bps < 0 or trading_days <= 0 or horizon <= 0):
            return result
        base = sum(values.values())
        if not math.isfinite(base) or base <= 0:
            return result
        orders = pd.DataFrame(orders).copy()
        partial = orders["Action"].eq("Sell") & orders["Optimal Weight"].gt(1e-12)
        symbols = sorted({t for t, v in values.items() if v > 0} | set(orders["Yahoo Ticker"].astype(str)))
        history = pd.DataFrame(log_returns)
        if any(t not in history for t in symbols):
            return result
        history = history[symbols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(history) < horizon:
            result["reason"] = "Insufficient common history for the partial-sell Sharpe check"
            return result
        initial = np.array([values.get(t, 0) / base for t in symbols])
        matrix = history.to_numpy(dtype=float)

        def sharpe(subset):
            weights = initial.copy()
            gross = 0.0
            for order in subset.to_dict(orient="records"):
                amount = float(order["Executable Value"]) / base
                if not math.isfinite(amount) or amount < 0:
                    return None
                direction = 1 if order["Action"] == "Buy" else -1
                weights[symbols.index(str(order["Yahoo Ticker"]))] += direction * amount
                gross += amount
            if weights.min() < -1e-9 or weights.sum() > 1 + 1e-9:
                return None
            cost = gross * cost_bps / 10000
            if cost >= 1:
                return None
            returns = matrix @ weights
            vol = float(returns.std(ddof=1)) * math.sqrt(trading_days)
            if vol <= 1e-12:
                return None
            annual = float(returns.mean()) * trading_days + math.log1p(-cost) * trading_days / horizon
            ratio = (annual - risk_free_rate) / vol
            return ratio if math.isfinite(ratio) else None

        before = sharpe(orders.iloc[:0])
        after = sharpe(orders)
        without = sharpe(orders.loc[~partial])
        result.update(current_sharpe=before, proposed_net_sharpe=after,
                      without_trims_net_sharpe=without, observations=len(history))
        if before is None or after is None:
            result["reason"] = "Invalid portfolio exposure or undefined Sharpe"
            return result
        result["allowed"] = after > before + 1e-9 and (without is None or after > without + 1e-9)
        result["reason"] = ("Partial sells improve overall Sharpe after estimated costs"
                            if result["allowed"] else "Partial sells do not improve overall Sharpe after estimated costs")
        return result
    except (ValueError, TypeError, KeyError, OverflowError):
        return result


def partial_sell_caption(check):
    if not check:
        return "Partial sells require an improvement in overall portfolio Sharpe after estimated costs."
    text = check["reason"]
    before, after = check.get("current_sharpe"), check.get("proposed_net_sharpe")
    if before is not None and after is not None:
        text += f" · Sharpe {before:.3f} → {after:.3f}"
    return text + f" · {check['cost_bps']:g} bp per side, costs spread over {check['horizon']} sessions."
