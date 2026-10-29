"""Transparent arithmetic for the saved portfolio reserve comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping


def calculate_leverage_stress_check(
    invested_base: float,
    expected_shortfall: float,
    portfolio_annual_return: float,
    borrowing_rate: float,
    loss_budget: float,
    maintenance_equity_ratio: float,
    horizon_sessions: int = 20,
    sessions_per_year: int = 252,
) -> dict[str, float]:
    """Estimate a leverage ceiling under explicitly supplied stress assumptions.

    The model scales the portfolio's historical ES linearly with gross exposure
    and charges simple interest over the stress horizon. It does not model gaps,
    security-level broker haircuts, changing margin rules, or future returns, so
    its result must never be interpreted as safe borrowing capacity.
    """
    values = {
        "invested_base": invested_base,
        "expected_shortfall": expected_shortfall,
        "portfolio_annual_return": portfolio_annual_return,
        "borrowing_rate": borrowing_rate,
        "loss_budget": loss_budget,
        "maintenance_equity_ratio": maintenance_equity_ratio,
    }
    normalized: dict[str, float] = {}
    for name, value in values.items():
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not math.isfinite(normalized[name]):
            raise ValueError(f"{name} must be finite")

    base = normalized["invested_base"]
    es = normalized["expected_shortfall"]
    annual_return = normalized["portfolio_annual_return"]
    rate = normalized["borrowing_rate"]
    budget = normalized["loss_budget"]
    maintenance = normalized["maintenance_equity_ratio"]
    if base <= 0:
        raise ValueError("invested_base must be positive")
    if es < 0 or es >= 1:
        raise ValueError("expected_shortfall must be between 0 and 1")
    if rate < 0:
        raise ValueError("borrowing_rate must be nonnegative")
    if budget < 0 or budget >= 1:
        raise ValueError("loss_budget must be between 0 and 1")
    if maintenance <= 0 or maintenance >= 1:
        raise ValueError("maintenance_equity_ratio must be between 0 and 1")
    if horizon_sessions <= 0 or sessions_per_year <= 0:
        raise ValueError("session counts must be positive")

    horizon_interest = rate * horizon_sessions / sessions_per_year
    stress_denominator = es + horizon_interest
    if budget <= es:
        debt_ratio_from_loss = 0.0
    elif stress_denominator <= 0:
        debt_ratio_from_loss = math.inf
    else:
        debt_ratio_from_loss = (budget - es) / stress_denominator

    maintenance_denominator = es + horizon_interest + maintenance * (1.0 - es)
    if maintenance_denominator <= 0:
        debt_ratio_from_margin = 0.0
    else:
        debt_ratio_from_margin = (
            (1.0 - maintenance) * (1.0 - es) / maintenance_denominator
        )

    debt_ratio = max(0.0, min(debt_ratio_from_loss, debt_ratio_from_margin))
    stress_loss_fraction = (
        (1.0 + debt_ratio) * es + debt_ratio * horizon_interest
    )
    stress_market_value = (1.0 + debt_ratio) * (1.0 - es) * base
    stress_equity_value = base * (1.0 - stress_loss_fraction)
    stressed_equity_ratio = (
        stress_equity_value / stress_market_value if stress_market_value > 0 else 0.0
    )

    return {
        "historical_return_minus_borrowing_rate": annual_return - rate,
        "debt_ratio_to_base": debt_ratio,
        "debt_amount": base * debt_ratio,
        "gross_exposure_multiple": 1.0 + debt_ratio,
        "modeled_horizon_loss_fraction": stress_loss_fraction,
        "stressed_equity_ratio": stressed_equity_ratio,
        "horizon_interest_fraction_of_base": debt_ratio * horizon_interest,
        "loss_budget_debt_ratio_limit": max(0.0, debt_ratio_from_loss),
        "maintenance_debt_ratio_limit": max(0.0, debt_ratio_from_margin),
    }


def calculate_portfolio_reserve_target(
    optimal_stats: Mapping[str, object],
    *,
    invested_base: float,
    horizon_sessions: int = 20,
    confidence: float = 0.95,
    sessions_per_year: int = 252,
) -> dict[str, float | str]:
    """Estimate additive reserve x from the portfolio's central path and downside tail.

    Preferred method: moving-block bootstrap median return plus one standard
    deviation of losses within the worst confidence tail. Older saved analyses
    lacking path summaries use a clearly identified Gaussian-volatility proxy.
    This is a liquidity/deployment reference, not insurance or an optimum.
    """
    try:
        base = float(invested_base)
    except (TypeError, ValueError) as exc:
        raise ValueError("invested_base must be numeric") from exc
    if not math.isfinite(base) or base <= 0:
        raise ValueError("invested_base must be positive and finite")
    if horizon_sessions <= 0 or sessions_per_year <= 0:
        raise ValueError("session counts must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    def finite_stat(key: str) -> float | None:
        try:
            value = float(optimal_stats[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    median_return = finite_stat("Block-Bootstrap Median Return (20 Sessions)")
    tail_sd = finite_stat("Block-Bootstrap Tail Loss SD 95% (20 Sessions)")
    if median_return is not None and tail_sd is not None and tail_sd >= 0:
        method = "moving-block bootstrap"
    else:
        annual_return = finite_stat("Annual Return")
        annual_volatility = finite_stat("Annual Volatility")
        if annual_return is None or annual_volatility is None or annual_volatility < 0:
            raise ValueError("optimal portfolio lacks usable return-distribution statistics")
        horizon_fraction = horizon_sessions / sessions_per_year
        median_return = math.expm1(annual_return * horizon_fraction)

        # Conditional standard deviation in the lower 5% tail of a normal core.
        # This only backfills older runs whose underlying bootstrap paths were not saved.
        alpha = 1.0 - confidence
        if abs(alpha - 0.05) > 1e-12:
            raise ValueError("the saved-statistics proxy is calibrated for a 95% tail")
        z = -1.6448536269514722
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        inverse_mills = phi / alpha
        conditional_variance_factor = max(
            0.0,
            1.0 - z * inverse_mills - inverse_mills * inverse_mills,
        )
        horizon_sd = annual_volatility * math.sqrt(horizon_fraction)
        tail_sd = horizon_sd * math.sqrt(conditional_variance_factor)
        method = "normal-tail volatility proxy (older saved run)"

    reserve_fraction = max(0.0, -median_return) + tail_sd
    return {
        "median_return": median_return,
        "tail_loss_sd": tail_sd,
        "reserve_fraction": reserve_fraction,
        "reserve_amount": base * reserve_fraction,
        "invested_base": base,
        "method": method,
        "horizon_sessions": float(horizon_sessions),
        "confidence": confidence,
    }
