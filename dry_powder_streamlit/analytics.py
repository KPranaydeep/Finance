"""Portfolio analytics, benchmark matching and dry-powder backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class DryPowderRecommendation:
    historical_max_drawdown: float
    stress_drawdown: float
    tolerance_drawdown: float
    formula_minimum_weight: float
    policy_floor_weight: float
    recommended_weight: float
    estimated_stress_drawdown: float


@dataclass(frozen=True)
class BacktestSummary:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_value: float
    fully_invested_end: float
    strategy_end: float
    idle_cash_end: float
    fully_invested_return: float
    strategy_return: float
    fully_invested_max_drawdown: float
    strategy_max_drawdown: float
    cash_deployed: float
    triggers_hit: int
    deployment_value_add: float
    opportunity_cost_or_gain: float


def drawdown_series(values: pd.Series) -> pd.Series:
    values = values.dropna().astype(float)
    if values.empty:
        return values
    return values / values.cummax() - 1.0


def max_drawdown(values: pd.Series) -> float:
    dd = drawdown_series(values)
    return float(-dd.min()) if not dd.empty else 0.0


def annualized_volatility(prices: pd.Series) -> float:
    returns = prices.pct_change().dropna()
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else 0.0


def recommend_dry_powder(
    prices: pd.Series,
    tolerance_drawdown: float,
    policy_floor: float,
    custom_stress_drawdown: float | None = None,
    recommendation_cap: float = 0.50,
) -> DryPowderRecommendation:
    historical_mdd = max_drawdown(prices)
    stress = custom_stress_drawdown if custom_stress_drawdown is not None else historical_mdd
    stress = max(float(stress), 0.01)
    tolerance = max(float(tolerance_drawdown), 0.0)

    # Approximation: if cash is stable during the shock, portfolio DD ~= equity_weight * index DD.
    formula_min = max(0.0, 1.0 - tolerance / stress)
    recommended = min(max(formula_min, policy_floor), recommendation_cap)
    # Round up to a usable 2.5 percentage-point allocation.
    recommended = min(math.ceil(recommended / 0.025) * 0.025, recommendation_cap)
    estimated = (1.0 - recommended) * stress

    return DryPowderRecommendation(
        historical_max_drawdown=historical_mdd,
        stress_drawdown=stress,
        tolerance_drawdown=tolerance,
        formula_minimum_weight=formula_min,
        policy_floor_weight=policy_floor,
        recommended_weight=recommended,
        estimated_stress_drawdown=estimated,
    )


def previous_completed_quarter(reference_date: date | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    ref = pd.Timestamp(reference_date).normalize()
    current_q_start = ref.to_period("Q").start_time
    previous_q_end = current_q_start - pd.Timedelta(days=1)
    previous_q_start = previous_q_end.to_period("Q").start_time
    return previous_q_start, previous_q_end


def available_completed_quarters(prices: pd.Series, count: int = 12) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if prices.empty:
        return []
    end_ref = min(pd.Timestamp.today().normalize(), prices.index.max())
    start, end = previous_completed_quarter(end_ref)
    quarters = []
    for _ in range(count):
        if start <= prices.index.max() and end >= prices.index.min():
            quarters.append((start, end))
        end = start - pd.Timedelta(days=1)
        start = end.to_period("Q").start_time
    return quarters


def run_dry_powder_backtest(
    prices: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    dry_powder_weight: float,
    thresholds: list[float],
    annual_cash_yield: float = 0.0,
) -> tuple[pd.DataFrame, BacktestSummary, pd.DataFrame]:
    q = prices.loc[(prices.index >= start_date) & (prices.index <= end_date)].dropna().astype(float)
    if len(q) < 2:
        raise ValueError("Not enough benchmark observations in the selected quarter.")

    thresholds = sorted(set(float(x) for x in thresholds if 0 < float(x) < 1))
    dry_powder_weight = min(max(float(dry_powder_weight), 0.0), 1.0)
    initial_capital = float(initial_capital)

    first_price = float(q.iloc[0])
    fully_units = initial_capital / first_price
    strategy_units = initial_capital * (1.0 - dry_powder_weight) / first_price
    idle_units = strategy_units
    cash = initial_capital * dry_powder_weight
    idle_cash = cash
    original_cash = cash
    tranche = original_cash / len(thresholds) if thresholds else 0.0
    next_trigger = 0
    running_peak = first_price
    previous_date = q.index[0]
    rows = []
    events = []

    for dt, price in q.items():
        days = max((dt - previous_date).days, 0)
        if days:
            growth = (1.0 + annual_cash_yield) ** (days / 365.0)
            cash *= growth
            idle_cash *= growth
        previous_date = dt
        running_peak = max(running_peak, float(price))
        current_dd = 1.0 - float(price) / running_peak

        while next_trigger < len(thresholds) and current_dd >= thresholds[next_trigger] and cash > 0:
            spend = min(tranche, cash)
            units_bought = spend / float(price)
            strategy_units += units_bought
            cash -= spend
            events.append(
                {
                    "Date": dt,
                    "Trigger": thresholds[next_trigger],
                    "Index price": float(price),
                    "Invested": spend,
                    "Units bought": units_bought,
                }
            )
            next_trigger += 1

        rows.append(
            {
                "Date": dt,
                "Index": float(price),
                "Fully invested": fully_units * float(price),
                "Dry powder deployed": strategy_units * float(price) + cash,
                "Dry powder kept idle": idle_units * float(price) + idle_cash,
                "Cash remaining": cash,
                "Index drawdown": -current_dd,
            }
        )

    result = pd.DataFrame(rows).set_index("Date")
    events_df = pd.DataFrame(events)
    deployed = original_cash - cash
    fully_dd = max_drawdown(result["Fully invested"])
    strategy_dd = max_drawdown(result["Dry powder deployed"])

    summary = BacktestSummary(
        start_date=result.index[0],
        end_date=result.index[-1],
        start_value=initial_capital,
        fully_invested_end=float(result["Fully invested"].iloc[-1]),
        strategy_end=float(result["Dry powder deployed"].iloc[-1]),
        idle_cash_end=float(result["Dry powder kept idle"].iloc[-1]),
        fully_invested_return=float(result["Fully invested"].iloc[-1] / initial_capital - 1),
        strategy_return=float(result["Dry powder deployed"].iloc[-1] / initial_capital - 1),
        fully_invested_max_drawdown=fully_dd,
        strategy_max_drawdown=strategy_dd,
        cash_deployed=float(deployed),
        triggers_hit=len(events),
        deployment_value_add=float(
            result["Dry powder deployed"].iloc[-1] - result["Dry powder kept idle"].iloc[-1]
        ),
        opportunity_cost_or_gain=float(
            result["Dry powder deployed"].iloc[-1] - result["Fully invested"].iloc[-1]
        ),
    )
    return result, summary, events_df


def weighted_portfolio_returns(price_frame: pd.DataFrame, holdings: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    available = [t for t in holdings["ticker"] if t in price_frame.columns and price_frame[t].notna().sum() >= 40]
    missing = [t for t in holdings["ticker"] if t not in available]
    if not available:
        raise ValueError("None of the uploaded holding tickers returned enough price history.")

    weights = holdings.set_index("ticker").loc[available, "weight"]
    weights = weights / weights.sum()
    returns = price_frame[available].pct_change(fill_method=None)
    # Renormalize weights across holdings available on each date.
    valid = returns.notna().astype(float)
    weighted = returns.fillna(0).mul(weights, axis=1).sum(axis=1)
    denominator = valid.mul(weights, axis=1).sum(axis=1)
    portfolio = (weighted / denominator.replace(0, np.nan)).dropna()
    portfolio.name = "Portfolio"
    return portfolio, missing


def score_benchmarks(portfolio_returns: pd.Series, benchmark_prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in benchmark_prices.columns:
        benchmark_returns = benchmark_prices[symbol].pct_change(fill_method=None).dropna()
        aligned = pd.concat([portfolio_returns, benchmark_returns.rename("Benchmark")], axis=1).dropna()
        if len(aligned) < 40 or aligned["Benchmark"].var() <= 0:
            continue

        corr = float(aligned.iloc[:, 0].corr(aligned["Benchmark"]))
        beta = float(aligned.iloc[:, 0].cov(aligned["Benchmark"]) / aligned["Benchmark"].var())
        active = aligned.iloc[:, 0] - aligned["Benchmark"]
        tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        beta_fit = max(0.0, 1.0 - min(abs(beta - 1.0), 1.0))
        corr_fit = max(0.0, corr)
        te_fit = 1.0 / (1.0 + 5.0 * tracking_error)
        score = 100.0 * (0.60 * corr_fit + 0.25 * beta_fit + 0.15 * te_fit)
        rows.append(
            {
                "symbol": symbol,
                "match_score": score,
                "correlation": corr,
                "beta": beta,
                "tracking_error": tracking_error,
                "overlap_days": len(aligned),
            }
        )
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True)
