"""Self-contained Streamlit app for Indian portfolio dry-powder planning.

This file intentionally contains its analytics, configuration, and market-data
helpers so Streamlit Community Cloud can deploy it as a single entrypoint.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INDEX_CATALOG = {
    "NIFTY 50": {
        "symbol": "^NSEI",
        "nse_name": "NIFTY 50",
        "kind": "Index",
        "description": "Large-cap diversified Indian equities; suitable for portfolios dominated by major companies.",
    },
    "NIFTY 500": {
        "symbol": "^CRSLDX",
        "nse_name": "NIFTY 500",
        "kind": "Index",
        "description": "Broad-market proxy spanning large, mid and small companies. Availability is validated at runtime.",
    },
    "NIFTY Midcap 100": {
        "symbol": "^CNXMidcap",
        "nse_name": "NIFTY MIDCAP 100",
        "kind": "Index",
        "description": "Mid-cap benchmark; useful when the portfolio has a substantial mid-cap allocation.",
    },
    "NIFTY Next 50 (ETF proxy)": {
        "symbol": "JUNIORBEES.NS",
        "nse_name": "NIFTY NEXT 50",
        "kind": "ETF proxy",
        "description": "Tradable proxy for companies immediately below the NIFTY 50 universe.",
    },
    "NIFTY Midcap 150 (ETF proxy)": {
        "symbol": "MID150BEES.NS",
        "nse_name": "NIFTY MIDCAP 150",
        "kind": "ETF proxy",
        "description": "Tradable proxy for a diversified mid-cap portfolio.",
    },
    "NIFTY Smallcap 250 (ETF proxy)": {
        "symbol": "SMALLCAP.NS",
        "nse_name": "NIFTY SMALLCAP 250",
        "kind": "ETF proxy",
        "description": "Tradable proxy for diversified small-cap exposure; validate symbol/data availability.",
    },
    "NIFTY Bank": {
        "symbol": "^NSEBANK",
        "nse_name": "NIFTY BANK",
        "kind": "Index",
        "description": "For portfolios concentrated in large and liquid banking stocks.",
    },
    "NIFTY IT": {
        "symbol": "^CNXIT",
        "nse_name": "NIFTY IT",
        "kind": "Index",
        "description": "For portfolios concentrated in Indian information-technology companies.",
    },
    "NIFTY Auto": {
        "symbol": "^CNXAUTO",
        "nse_name": "NIFTY AUTO",
        "kind": "Index",
        "description": "For portfolios concentrated in automobile and related companies.",
    },
    "NIFTY FMCG": {
        "symbol": "^CNXFMCG",
        "nse_name": "NIFTY FMCG",
        "kind": "Index",
        "description": "For portfolios concentrated in fast-moving consumer-goods companies.",
    },
    "NIFTY Pharma": {
        "symbol": "^CNXPHARMA",
        "nse_name": "NIFTY PHARMA",
        "kind": "Index",
        "description": "For portfolios concentrated in pharmaceutical companies.",
    },
    "NIFTY Metal": {
        "symbol": "^CNXMETAL",
        "nse_name": "NIFTY METAL",
        "kind": "Index",
        "description": "For portfolios concentrated in metals and mining companies.",
    },
    "NIFTY Realty": {
        "symbol": "^CNXREALTY",
        "nse_name": "NIFTY REALTY",
        "kind": "Index",
        "description": "For portfolios concentrated in listed real-estate companies.",
    },
    "NIFTY PSU Bank": {
        "symbol": "^CNXPSUBANK",
        "nse_name": "NIFTY PSU BANK",
        "kind": "Index",
        "description": "For portfolios concentrated in public-sector banks.",
    },
    "NIFTY Energy": {
        "symbol": "^CNXENERGY",
        "nse_name": "NIFTY ENERGY",
        "kind": "Index",
        "description": "For portfolios concentrated in energy-sector companies.",
    },
    "NIFTY Infrastructure": {
        "symbol": "^CNXINFRA",
        "nse_name": "NIFTY INFRASTRUCTURE",
        "kind": "Index",
        "description": "For portfolios concentrated in infrastructure-related companies.",
    },
}

POLICY_FLOORS = {
    "Aggressive — 5% minimum reserve": 0.05,
    "Balanced — 10% minimum reserve": 0.10,
    "Defensive — 20% minimum reserve": 0.20,
    "Formula only — no policy floor": 0.00,
}

DEFAULT_THRESHOLDS = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]


# -----------------------------------------------------------------------------
# Portfolio analytics
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from datetime import date
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class DryPowderRecommendation:
    lookback_max_drawdown: float
    all_history_max_drawdown: float
    index_stress_drawdown: float
    portfolio_beta: float
    portfolio_stress_drawdown: float
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
    reserve_start: float
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
    initial_equity_gain: float
    cash_interest_earned: float


@dataclass(frozen=True)
class LiveDeploymentRecommendation:
    latest_date: pd.Timestamp
    latest_price: float
    peak_date: pd.Timestamp
    peak_price: float
    current_drawdown: float
    triggers_crossed: tuple[float, ...]
    target_cumulative_deployment: float
    mechanical_deploy_now: float
    keep_as_dry_powder_after_mechanical: float
    next_trigger: float | None
    overdeployment_amount: float
    status: str


def drawdown_series(values: pd.Series) -> pd.Series:
    values = values.dropna().astype(float)
    if values.empty:
        return values
    return values / values.cummax() - 1.0


def max_drawdown(values: pd.Series) -> float:
    dd = drawdown_series(values)
    return float(-dd.min()) if not dd.empty else 0.0


def required_reserve_weight(
    stress_drawdown: float,
    tolerance_drawdown: float,
    portfolio_beta: float = 1.0,
    cap: float = 0.95,
) -> float:
    """Estimate the cash weight needed to constrain a specified portfolio shock.

    This is a drawdown-control identity, not an optimisation model:
    portfolio drawdown ~= equity weight * benchmark stress * beta.
    """
    stress = max(float(stress_drawdown), 0.0)
    beta = max(float(portfolio_beta), 0.0)
    portfolio_stress = min(stress * beta, 0.99)
    tolerance = max(float(tolerance_drawdown), 0.0)
    if portfolio_stress <= 0:
        return 0.0
    return min(max(0.0, 1.0 - tolerance / portfolio_stress), cap)


def recommend_dry_powder(
    prices: pd.Series,
    all_prices: pd.Series,
    tolerance_drawdown: float,
    policy_floor: float,
    stress_drawdown: float,
    portfolio_beta: float = 1.0,
    recommendation_cap: float = 0.80,
) -> DryPowderRecommendation:
    lookback_mdd = max_drawdown(prices)
    all_history_mdd = max_drawdown(all_prices)
    stress = max(float(stress_drawdown), 0.01)
    beta = max(float(portfolio_beta), 0.0)
    portfolio_stress = min(stress * beta, 0.99)
    formula_min = required_reserve_weight(
        stress_drawdown=stress,
        tolerance_drawdown=tolerance_drawdown,
        portfolio_beta=beta,
        cap=recommendation_cap,
    )
    recommended = min(max(formula_min, float(policy_floor)), recommendation_cap)
    estimated = (1.0 - recommended) * portfolio_stress

    return DryPowderRecommendation(
        lookback_max_drawdown=lookback_mdd,
        all_history_max_drawdown=all_history_mdd,
        index_stress_drawdown=stress,
        portfolio_beta=beta,
        portfolio_stress_drawdown=portfolio_stress,
        tolerance_drawdown=float(tolerance_drawdown),
        formula_minimum_weight=formula_min,
        policy_floor_weight=float(policy_floor),
        recommended_weight=recommended,
        estimated_stress_drawdown=estimated,
    )


def normalize_ladder(
    thresholds: list[float],
    weights: list[float],
) -> tuple[list[float], list[float]]:
    clean_thresholds = [float(x) for x in thresholds if 0 < float(x) < 1]
    clean_weights = [float(x) for x in weights if float(x) >= 0]
    if not clean_thresholds:
        raise ValueError("At least one deployment trigger is required.")
    if len(clean_thresholds) != len(clean_weights):
        raise ValueError("The number of deployment weights must equal the number of drawdown triggers.")
    pairs = sorted(zip(clean_thresholds, clean_weights), key=lambda item: item[0])
    deduped: dict[float, float] = {}
    for threshold, weight in pairs:
        deduped[threshold] = weight
    thresholds_out = list(deduped)
    weights_out = [deduped[t] for t in thresholds_out]
    total = sum(weights_out)
    if total <= 0:
        raise ValueError("Deployment weights must add to more than zero.")
    weights_out = [w / total for w in weights_out]
    return thresholds_out, weights_out


def recommend_live_deployment(
    prices: pd.Series,
    cycle_start_reserve: float,
    current_dry_powder: float,
    already_deployed: float,
    thresholds: list[float],
    weights: list[float],
) -> LiveDeploymentRecommendation:
    """Calculate mechanical permission from the actual funded cycle reserve.

    The aspirational drawdown-control reserve is deliberately not used here.
    This prevents the app from recommending deployment of money that was never
    funded at the start of the current drawdown cycle.
    """
    series = prices.dropna().astype(float).sort_index()
    if len(series) < 2:
        raise ValueError("At least two valid prices are required for a closing-price signal.")

    thresholds, weights = normalize_ladder(thresholds, weights)
    latest_date = pd.Timestamp(series.index[-1])
    latest_price = float(series.iloc[-1])
    peak_price = float(series.max())
    peak_date = pd.Timestamp(series[series == peak_price].index[-1])
    current_drawdown = max(0.0, 1.0 - latest_price / peak_price)

    cycle_start = max(float(cycle_start_reserve), 0.0)
    current_cash = max(float(current_dry_powder), 0.0)
    deployed = max(float(already_deployed), 0.0)

    crossed_indices = [i for i, trigger in enumerate(thresholds) if current_drawdown >= trigger]
    crossed = tuple(thresholds[i] for i in crossed_indices)
    cumulative_weight = sum(weights[i] for i in crossed_indices)
    target_cumulative = cycle_start * cumulative_weight
    required_now = max(0.0, target_cumulative - deployed)
    mechanical = min(current_cash, required_now)
    overdeployment = max(0.0, deployed - target_cumulative)
    keep = max(0.0, current_cash - mechanical)
    next_trigger = next((t for t in thresholds if t > current_drawdown), None)

    if mechanical > 0:
        status = "MECHANICAL DEPLOYMENT PERMITTED"
    elif overdeployment > 0:
        status = "HOLD — ALREADY AHEAD OF LADDER"
    elif crossed:
        status = "HOLD — CROSSED TRIGGERS ALREADY FUNDED"
    else:
        status = "HOLD — FIRST TRIGGER NOT REACHED"

    return LiveDeploymentRecommendation(
        latest_date=latest_date,
        latest_price=latest_price,
        peak_date=peak_date,
        peak_price=peak_price,
        current_drawdown=current_drawdown,
        triggers_crossed=crossed,
        target_cumulative_deployment=target_cumulative,
        mechanical_deploy_now=mechanical,
        keep_as_dry_powder_after_mechanical=keep,
        next_trigger=next_trigger,
        overdeployment_amount=overdeployment,
        status=status,
    )


def available_completed_years(
    prices: pd.Series,
    count: int = 15,
    minimum_observations: int = 180,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    prices = prices.dropna()
    if prices.empty:
        return []

    last_candidate_year = min(pd.Timestamp.today().year - 1, prices.index.max().year)
    first_data_year = prices.index.min().year
    years: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for year in range(last_candidate_year, first_data_year - 1, -1):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        yearly_prices = prices.loc[(prices.index >= start) & (prices.index <= end)]
        if yearly_prices.empty:
            continue
        covers_start = yearly_prices.index.min() <= pd.Timestamp(year=year, month=3, day=31)
        covers_end = yearly_prices.index.max() >= pd.Timestamp(year=year, month=10, day=1)
        if len(yearly_prices) >= minimum_observations and covers_start and covers_end:
            years.append((start, end))
        if len(years) >= count:
            break
    return years


def run_dry_powder_backtest(
    prices: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    dry_powder_weight: float,
    thresholds: list[float],
    weights: list[float],
    annual_cash_yield: float = 0.0,
) -> tuple[pd.DataFrame, BacktestSummary, pd.DataFrame]:
    q = prices.loc[(prices.index >= start_date) & (prices.index <= end_date)].dropna().astype(float)
    if len(q) < 2:
        raise ValueError("Not enough benchmark observations in the selected year.")

    thresholds, weights = normalize_ladder(thresholds, weights)
    dry_powder_weight = min(max(float(dry_powder_weight), 0.0), 1.0)
    initial_capital = float(initial_capital)

    first_price = float(q.iloc[0])
    initial_equity_capital = initial_capital * (1.0 - dry_powder_weight)
    fully_units = initial_capital / first_price
    strategy_units = initial_equity_capital / first_price
    idle_units = strategy_units
    cash = initial_capital * dry_powder_weight
    idle_cash = cash
    original_cash = cash
    tranche_amounts = [original_cash * weight for weight in weights]
    next_trigger = 0
    running_peak = first_price
    previous_date = q.index[0]
    rows: list[dict] = []
    events: list[dict] = []
    interest_earned = 0.0

    for dt, price in q.items():
        days = max((dt - previous_date).days, 0)
        if days:
            growth = (1.0 + annual_cash_yield) ** (days / 365.0)
            interest_earned += cash * (growth - 1.0)
            cash *= growth
            idle_cash *= growth
        previous_date = dt
        running_peak = max(running_peak, float(price))
        current_dd = 1.0 - float(price) / running_peak

        while next_trigger < len(thresholds) and current_dd >= thresholds[next_trigger] and cash > 0:
            spend = min(tranche_amounts[next_trigger], cash)
            units_bought = spend / float(price)
            strategy_units += units_bought
            cash -= spend
            events.append(
                {
                    "Date": dt,
                    "Trigger": thresholds[next_trigger],
                    "Ladder weight": weights[next_trigger],
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
    deployed = sum(float(event["Invested"]) for event in events)
    fully_dd = max_drawdown(result["Fully invested"])
    strategy_dd = max_drawdown(result["Dry powder deployed"])
    initial_equity_end = idle_units * float(q.iloc[-1])

    summary = BacktestSummary(
        start_date=result.index[0],
        end_date=result.index[-1],
        start_value=initial_capital,
        reserve_start=original_cash,
        fully_invested_end=float(result["Fully invested"].iloc[-1]),
        strategy_end=float(result["Dry powder deployed"].iloc[-1]),
        idle_cash_end=float(result["Dry powder kept idle"].iloc[-1]),
        fully_invested_return=float(result["Fully invested"].iloc[-1] / initial_capital - 1),
        strategy_return=float(result["Dry powder deployed"].iloc[-1] / initial_capital - 1),
        fully_invested_max_drawdown=fully_dd,
        strategy_max_drawdown=strategy_dd,
        cash_deployed=deployed,
        triggers_hit=len(events),
        deployment_value_add=float(
            result["Dry powder deployed"].iloc[-1] - result["Dry powder kept idle"].iloc[-1]
        ),
        opportunity_cost_or_gain=float(
            result["Dry powder deployed"].iloc[-1] - result["Fully invested"].iloc[-1]
        ),
        initial_equity_gain=float(initial_equity_end - initial_equity_capital),
        cash_interest_earned=float(interest_earned),
    )
    return result, summary, events_df


def weighted_portfolio_returns(price_frame: pd.DataFrame, holdings: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    available = [t for t in holdings["ticker"] if t in price_frame.columns and price_frame[t].notna().sum() >= 40]
    missing = [t for t in holdings["ticker"] if t not in available]
    if not available:
        attempted = ", ".join(holdings["ticker"].astype(str).tolist()[:8])
        raise ValueError(
            "None of the uploaded holding tickers returned enough Yahoo price history. "
            f"Tried: {attempted}. Bond, ISIN-style, delisted or unsupported BSE symbols may not be available."
        )

    weights = holdings.set_index("ticker").loc[available, "weight"]
    weights = weights / weights.sum()
    returns = price_frame[available].pct_change(fill_method=None)
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
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "match_score", "correlation", "beta", "tracking_error", "overlap_days"]
        )
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True)
# -----------------------------------------------------------------------------
# Market data and CSV parsing
# -----------------------------------------------------------------------------
from datetime import date, timedelta
from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf


NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
}


def download_nse_index_close(index_name: str, start: date | str, end: date | str) -> pd.Series:
    """Download daily index closes from NSE's public historical-index endpoint.

    NSE may throttle or block automated cloud traffic. Callers should expose a CSV
    fallback and present a clear error rather than silently fabricating data.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts >= end_ts:
        raise ValueError("Start date must be before end date.")

    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    origin = "https://www.nseindia.com/reports-indices-historical-index-data"
    try:
        session.get(origin, timeout=20)
    except requests.RequestException:
        pass

    chunks = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=364), end_ts)
        url = (
            "https://www.nseindia.com/api/historical/indicesHistory"
            f"?indexType={quote(index_name.upper())}"
            f"&from={cursor.strftime('%d-%m-%Y')}"
            f"&to={chunk_end.strftime('%d-%m-%Y')}"
        )
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            chunks.append(_nse_payload_to_close(payload, index_name))
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise ValueError(f"NSE historical-data request failed for {index_name}: {exc}") from exc
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise ValueError(f"NSE returned no history for {index_name}.")
    close = pd.concat(chunks).sort_index()
    close = close[~close.index.duplicated(keep="last")].dropna()
    if close.empty:
        raise ValueError(f"NSE returned no usable close values for {index_name}.")
    close.name = index_name
    return close


def _nse_payload_to_close(payload: dict, index_name: str) -> pd.Series:
    data = payload.get("data", payload)
    records = []
    if isinstance(data, dict):
        for key in ("indexCloseOnlineRecords", "indexCloseRecords", "data"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
    elif isinstance(data, list):
        records = data
    if not records:
        raise ValueError("response contained no close records")

    df = pd.DataFrame(records)
    date_col = next((c for c in ("EOD_TIMESTAMP", "TIMESTAMP", "HistoricalDate", "Date", "date") if c in df.columns), None)
    close_col = next((c for c in ("EOD_CLOSE_INDEX_VAL", "CLOSE_INDEX_VAL", "CLOSE", "Close", "close") if c in df.columns), None)
    if not date_col or not close_col:
        raise ValueError(f"unexpected NSE response fields: {', '.join(map(str, df.columns[:12]))}")

    dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    values = pd.to_numeric(df[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=index_name).dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


def download_benchmark_close(
    nse_name: str,
    yahoo_symbol: str,
    start: date | str,
    end: date | str,
    provider: str = "Automatic",
) -> tuple[pd.Series, str]:
    if provider == "NSE official":
        return download_nse_index_close(nse_name, start, end), "NSE official historical data"
    if provider == "Yahoo/yfinance":
        return download_close(yahoo_symbol, start=start, end=end), "Yahoo Finance via yfinance"

    nse_error = None
    try:
        return download_nse_index_close(nse_name, start, end), "NSE official historical data"
    except Exception as exc:
        nse_error = exc
    try:
        return download_close(yahoo_symbol, start=start, end=end), "Yahoo Finance via yfinance (NSE fallback failed)"
    except Exception as yahoo_exc:
        raise ValueError(f"Both providers failed. NSE: {nse_error}. Yahoo: {yahoo_exc}") from yahoo_exc


def _extract_close(raw: pd.DataFrame, symbol: str | None = None) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance can return (Price, Ticker) or (Ticker, Price) columns.
        level0 = set(map(str, raw.columns.get_level_values(0)))
        level1 = set(map(str, raw.columns.get_level_values(1)))
        if "Close" in level0:
            close = raw["Close"]
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1)
        else:
            return pd.Series(dtype=float)
        if isinstance(close, pd.DataFrame):
            if symbol and symbol in close.columns:
                close = close[symbol]
            else:
                close = close.iloc[:, 0]
    elif "Close" in raw.columns:
        close = raw["Close"]
    elif "Adj Close" in raw.columns:
        close = raw["Adj Close"]
    else:
        return pd.Series(dtype=float)

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = symbol or "Close"
    return close.sort_index()


def download_close(
    symbol: str,
    start: date | str | None = None,
    end: date | str | None = None,
    period: str | None = None,
) -> pd.Series:
    kwargs = {
        "tickers": symbol,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
        "timeout": 20,
    }
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    raw = yf.download(**kwargs)
    close = _extract_close(raw, symbol)
    if close.empty:
        raise ValueError(
            f"No price history was returned for {symbol}. Try another symbol or upload a Date/Close CSV."
        )
    return close


def download_many(symbols: Iterable[str], start: date | str, end: date | str) -> pd.DataFrame:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
        timeout=30,
    )
    if raw.empty:
        return pd.DataFrame()

    output = {}
    for symbol in symbols:
        try:
            series = _extract_symbol_from_multi(raw, symbol, len(symbols))
            if not series.empty:
                output[symbol] = series
        except (KeyError, IndexError, TypeError):
            continue
    return pd.DataFrame(output).sort_index()


def _extract_symbol_from_multi(raw: pd.DataFrame, symbol: str, symbol_count: int) -> pd.Series:
    if symbol_count == 1:
        return _extract_close(raw, symbol)
    if not isinstance(raw.columns, pd.MultiIndex):
        return pd.Series(dtype=float)

    level0 = set(map(str, raw.columns.get_level_values(0)))
    if "Close" in level0:
        close_df = raw["Close"]
        if symbol not in close_df.columns:
            return pd.Series(dtype=float)
        series = close_df[symbol]
    else:
        # Some yfinance versions may return ticker as the first level.
        if symbol not in level0:
            return pd.Series(dtype=float)
        sub = raw[symbol]
        if "Close" not in sub.columns:
            return pd.Series(dtype=float)
        series = sub["Close"]

    series = pd.to_numeric(series, errors="coerce").dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = symbol
    return series


def parse_benchmark_csv(uploaded_file) -> pd.Series:
    df = pd.read_csv(uploaded_file)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "date" not in normalized:
        raise ValueError("Benchmark CSV must contain a Date column.")
    close_key = next((key for key in ("close", "adj close", "adjusted close", "price") if key in normalized), None)
    if not close_key:
        raise ValueError("Benchmark CSV must contain Close, Adj Close, Adjusted Close or Price.")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[normalized["date"]], errors="coerce"),
            "Close": pd.to_numeric(df[normalized[close_key]], errors="coerce"),
        }
    ).dropna()
    out = out.drop_duplicates("Date", keep="last").sort_values("Date")
    if len(out) < 20:
        raise ValueError("Benchmark CSV needs at least 20 valid daily observations.")
    return out.set_index("Date")["Close"]


def parse_holdings_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    ticker_key = next((k for k in ("ticker", "symbol", "nse symbol") if k in normalized), None)
    if not ticker_key:
        raise ValueError("Holdings CSV must contain a ticker or symbol column.")

    value_key = next((k for k in ("value", "market value", "current value", "amount") if k in normalized), None)
    weight_key = next((k for k in ("weight", "weight %", "allocation", "allocation %") if k in normalized), None)
    if not value_key and not weight_key:
        raise ValueError("Holdings CSV must contain either value/amount or weight/allocation.")

    out = pd.DataFrame()
    out["ticker"] = df[normalized[ticker_key]].astype(str).str.strip().str.upper()
    out = out[out["ticker"].ne("") & out["ticker"].ne("NAN")]
    out["ticker"] = out["ticker"].map(_normalize_indian_ticker)

    if value_key:
        out["raw_weight"] = pd.to_numeric(df.loc[out.index, normalized[value_key]], errors="coerce")
    else:
        out["raw_weight"] = pd.to_numeric(df.loc[out.index, normalized[weight_key]], errors="coerce")

    out = out.dropna(subset=["raw_weight"])
    out = out[out["raw_weight"] > 0]
    out = out.groupby("ticker", as_index=False)["raw_weight"].sum()
    if out.empty:
        raise ValueError("No positive holdings were found in the CSV.")
    out["weight"] = out["raw_weight"] / out["raw_weight"].sum()
    return out[["ticker", "weight"]].sort_values("weight", ascending=False)


def _normalize_indian_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith("^") or ticker.endswith((".NS", ".BO")):
        return ticker
    return f"{ticker}.NS"


# -----------------------------------------------------------------------------
# Streamlit user interface
# -----------------------------------------------------------------------------
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Dry Powder Planner — India", page_icon="🛡️", layout="wide")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_benchmark(nse_name: str, symbol: str, start: date, end: date, provider: str):
    return download_benchmark_close(
        nse_name=nse_name,
        yahoo_symbol=symbol,
        start=start,
        end=end,
        provider=provider,
    )


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_many(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return download_many(symbols=symbols, start=start, end=end)


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_percentage_list(text: str, label: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip().replace("%", "")
        if not token:
            continue
        value = float(token) / 100.0
        if value < 0 or value >= 1:
            raise ValueError(f"Each {label.lower()} must be between 0% and 100%.")
        values.append(value)
    if not values:
        raise ValueError(f"Enter at least one {label.lower()}.")
    return values


def year_label(item: tuple[pd.Timestamp, pd.Timestamp]) -> str:
    start, end = item
    return f"{start.year} — {start:%d %b} to {end:%d %b}"


def render_matcher(start: date, end: date) -> dict | None:
    st.subheader("Find the benchmark that resembles your holdings")
    uploaded = st.file_uploader(
        "Upload holdings CSV",
        type=["csv"],
        help="Required: ticker/symbol plus value/amount or weight/allocation. NSE tickers may be entered without .NS.",
    )
    if uploaded is None:
        st.caption(
            "Upload holdings to validate the benchmark statistically. Without a successful match, "
            "the closing-price deployment signal requires manual benchmark confirmation."
        )
        return None

    try:
        holdings = parse_holdings_csv(uploaded)
        st.dataframe(
            holdings.assign(weight=lambda x: (100 * x["weight"]).round(2)).rename(columns={"weight": "Weight %"}),
            use_container_width=True,
            hide_index=True,
        )
        candidate_names = st.multiselect(
            "Candidate benchmarks to test",
            options=list(INDEX_CATALOG),
            default=["NIFTY 50", "NIFTY 500", "NIFTY Midcap 100", "NIFTY Bank", "NIFTY IT"],
        )
        if not candidate_names:
            st.warning("Choose at least one candidate benchmark.")
            return None

        symbols = tuple(holdings["ticker"].tolist() + [INDEX_CATALOG[n]["symbol"] for n in candidate_names])
        with st.spinner("Matching portfolio behaviour against candidate benchmarks…"):
            prices = cached_many(symbols, start, end)
            portfolio_returns, missing = weighted_portfolio_returns(prices, holdings)
            candidate_symbols = [INDEX_CATALOG[n]["symbol"] for n in candidate_names]
            available_candidates = [s for s in candidate_symbols if s in prices.columns]
            scores = score_benchmarks(portfolio_returns, prices[available_candidates])

        if missing:
            st.warning(f"Skipped {len(missing)} ticker(s) with insufficient data: {', '.join(missing[:8])}")
        if scores.empty:
            st.error(
                "Benchmark validation failed because no candidate had enough overlapping data. "
                "The deployment recommendation will remain disabled until you explicitly confirm a manual proxy."
            )
            return None

        reverse_names = {v["symbol"]: k for k, v in INDEX_CATALOG.items()}
        scores.insert(0, "Benchmark", scores["symbol"].map(reverse_names).fillna(scores["symbol"]))
        display = scores[["Benchmark", "match_score", "correlation", "beta", "tracking_error", "overlap_days"]].copy()
        display.columns = ["Benchmark", "Match score", "Correlation", "Beta", "Tracking error", "Days"]
        display["Match score"] = display["Match score"].round(1)
        display["Correlation"] = display["Correlation"].round(3)
        display["Beta"] = display["Beta"].round(2)
        display["Tracking error"] = (100 * display["Tracking error"]).round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True, hide_index=True)

        best = scores.iloc[0]
        best_symbol = str(best["symbol"])
        best_name = reverse_names.get(best_symbol, best_symbol)
        strong_enough = float(best["correlation"]) >= 0.70 and float(best["overlap_days"]) >= 120
        if strong_enough:
            st.success(
                f"Validated statistical proxy: **{best_name}** "
                f"(correlation {float(best['correlation']):.2f}, beta {float(best['beta']):.2f})."
            )
        else:
            st.warning(
                f"Best available match is **{best_name}**, but the evidence is weak "
                f"(correlation {float(best['correlation']):.2f}). Manual confirmation is still required."
            )
        return {
            "name": best_name if best_name in INDEX_CATALOG else None,
            "beta": max(float(best["beta"]), 0.0),
            "validated": strong_enough,
        }
    except Exception as exc:
        st.error(
            f"Could not validate the holdings benchmark: {exc} "
            "The app will not silently treat NIFTY 50 as validated."
        )
        return None


st.title("🛡️ Dry Powder Planner for Indian Equity Portfolios")
st.write(
    "Separate net worth, emergency liquidity, target drawdown control and the actual funded drawdown-cycle reserve. "
    "The app first calculates mechanical permission, then applies benchmark, liquidity and valuation gates."
)
st.info(
    "Dry powder is investable liquidity—not your emergency fund. The closing-price signal is an auditable rule, "
    "not a market-bottom prediction or personalised investment advice."
)

with st.sidebar:
    st.header("1. Net worth and liquidity")
    portfolio_value = st.number_input(
        "Total investable portfolio — equity + current dry powder (₹)",
        min_value=100_000.0,
        max_value=1_000_000_000.0,
        value=2_000_000.0,
        step=50_000.0,
    )
    current_dry_powder = st.number_input(
        "Current deployable dry powder (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=110_000.0,
        step=5_000.0,
    )
    already_deployed = st.number_input(
        "Already deployed in current drawdown cycle (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=211_000.0,
        step=5_000.0,
    )
    use_identifiable_cycle_reserve = st.checkbox(
        "Use identifiable cycle reserve = current cash + already deployed",
        value=True,
        help="Recommended. It prevents an aspirational target reserve from becoming fictional deployable money.",
    )
    identifiable_cycle_reserve = float(current_dry_powder) + float(already_deployed)
    if use_identifiable_cycle_reserve:
        cycle_start_reserve = identifiable_cycle_reserve
    else:
        cycle_start_reserve = st.number_input(
            "Actual funded reserve at cycle start (₹)",
            min_value=0.0,
            max_value=1_000_000_000.0,
            value=float(identifiable_cycle_reserve),
            step=5_000.0,
        )
    net_cycle_flows = st.number_input(
        "Net reserve additions (+) or withdrawals (−) since peak (₹)",
        min_value=-1_000_000_000.0,
        max_value=1_000_000_000.0,
        value=0.0,
        step=5_000.0,
        help="Exclude deployments. Add contributions to dry powder as positive; unrelated withdrawals as negative.",
    )

    emergency_fund = st.number_input(
        "Emergency fund — never deploy (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=12_000.0,
        step=10_000.0,
    )
    monthly_essential_expenses = st.number_input(
        "Monthly essential expenses (₹)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=30_000.0,
        step=5_000.0,
    )
    emergency_months = st.slider("Required emergency coverage (months)", 0, 24, 6)
    near_term_obligations = st.number_input(
        "Near-term obligations outside monthly expenses (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=0.0,
        step=10_000.0,
        help="Examples: insurance premiums, tuition, medical costs or loan payments due soon.",
    )
    other_assets = st.number_input("Other net-worth assets (₹)", 0.0, 10_000_000_000.0, 0.0, 50_000.0)
    liabilities = st.number_input("Outstanding liabilities (₹)", 0.0, 10_000_000_000.0, 0.0, 50_000.0)

    st.divider()
    st.header("2. Strategy assumptions")
    lookback_years = st.slider("Risk lookback", 3, 20, 15)
    tolerance = st.slider("Maximum portfolio drawdown you can tolerate", 5, 50, 23, 1) / 100
    portfolio_beta_input = st.number_input(
        "Portfolio beta to selected benchmark",
        min_value=0.25,
        max_value=3.00,
        value=1.00,
        step=0.05,
        help="Use the matched beta when available. A concentrated portfolio may have beta above 1.",
    )
    policy_options = list(POLICY_FLOORS)
    policy_name = st.selectbox(
        "Reserve policy",
        policy_options,
        index=policy_options.index("Formula only — no policy floor"),
    )
    stress_method = st.radio(
        "Stress basis",
        ["Selected-lookback maximum", "Downloaded-history maximum", "Custom crash assumption"],
    )
    custom_stress = None
    if stress_method == "Custom crash assumption":
        custom_stress = st.slider("Assumed benchmark crash", 10, 80, 40, 1) / 100

    cash_yield = st.number_input("Annual yield on dry powder (%)", 0.0, 20.0, 11.0, 0.25) / 100
    reserve_is_liquid = st.checkbox(
        "Reserve instrument is capital-stable and redeemable within one business day",
        value=False,
        help="A high-yielding credit or locked instrument is not equivalent to instant dry powder.",
    )
    trigger_text = st.text_input("Drawdown triggers (%)", "5, 10, 15, 20, 25, 30, 35, 40")
    weight_text = st.text_input(
        "Reserve deployed at each trigger (%)",
        "5, 10, 15, 20, 20, 15, 10, 5",
        help="Weights are normalised to 100%. Later declines receive more capital than ordinary market noise.",
    )
    custom_symbol = st.text_input(
        "Optional custom Yahoo Finance symbol",
        placeholder="Example: ^NSEI or an ETF ticker",
    )

if current_dry_powder > portfolio_value:
    st.error("Current dry powder cannot exceed the total investable portfolio.")
    st.stop()
if already_deployed > portfolio_value:
    st.error("Already deployed capital cannot exceed the total investable portfolio.")
    st.stop()

try:
    thresholds_raw = parse_percentage_list(trigger_text, "drawdown trigger")
    weights_raw = parse_percentage_list(weight_text, "deployment weight")
    thresholds, ladder_weights = normalize_ladder(thresholds_raw, weights_raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

current_equity_value = max(float(portfolio_value) - float(current_dry_powder), 0.0)
required_emergency_fund = float(monthly_essential_expenses) * emergency_months + float(near_term_obligations)
emergency_gap = float(emergency_fund) - required_emergency_fund
emergency_adequate = emergency_gap >= 0
calculated_net_worth = float(portfolio_value) + float(emergency_fund) + float(other_assets) - float(liabilities)

expected_current_cash = float(cycle_start_reserve) + float(net_cycle_flows) - float(already_deployed)
reconciliation_gap = float(current_dry_powder) - expected_current_cash
reconciliation_tolerance = max(1_000.0, 0.01 * max(float(cycle_start_reserve), 1.0))
cycle_reconciled = abs(reconciliation_gap) <= reconciliation_tolerance

st.header("1. Net worth, emergency liquidity and cycle funding")
nw = st.columns(7)
nw[0].metric("Calculated net worth", money(calculated_net_worth))
nw[1].metric("Investable portfolio", money(portfolio_value))
nw[2].metric("Currently invested", money(current_equity_value))
nw[3].metric("Current dry powder", money(current_dry_powder))
nw[4].metric("Funded cycle reserve", money(cycle_start_reserve))
nw[5].metric("Emergency fund", money(emergency_fund))
nw[6].metric("Required emergency fund", money(required_emergency_fund))

if emergency_adequate:
    st.success(f"Emergency liquidity gate: PASS. Buffer above requirement: **{money(emergency_gap)}**.")
else:
    st.error(
        f"Emergency liquidity gate: FAIL. Emergency cash is **{money(abs(emergency_gap))} below** the requirement. "
        "The final deployment recommendation will be ₹0."
    )

if cycle_reconciled:
    st.success("Cycle-reserve reconciliation: PASS. Current cash, deployments and net cycle flows reconcile.")
else:
    st.error(
        f"Cycle-reserve reconciliation: FAIL by **{money(abs(reconciliation_gap))}**. "
        "Correct the funded cycle reserve or net additions/withdrawals before relying on a deployment amount."
    )

end_download = date.today() + timedelta(days=1)
matcher_start = date.today() - timedelta(days=365 * 5 + 30)
# Download at least 20 years for the benchmark so the downloaded-history stress can include 2008 when available.
start_download = date.today() - timedelta(days=365 * max(lookback_years, 20) + 180)

mode = st.radio(
    "Benchmark selection mode",
    ["Choose manually", "Match using holdings CSV"],
    horizontal=True,
)
match_result = None
if mode == "Match using holdings CSV":
    match_result = render_matcher(matcher_start, end_download)

index_names = list(INDEX_CATALOG)
matched_name = match_result.get("name") if match_result else None
default_index = matched_name or "NIFTY 50"
selected_name = st.selectbox(
    "Benchmark for risk and backtest",
    index_names,
    index=index_names.index(default_index),
)
selected = INDEX_CATALOG[selected_name]
selected_symbol = custom_symbol.strip() or selected["symbol"]
st.caption(
    f"**{selected_name}** · {selected['kind']} · {selected['description']} · Data symbol: `{selected_symbol}`"
)

statistically_validated = bool(
    match_result
    and match_result.get("validated")
    and match_result.get("name") == selected_name
    and not custom_symbol.strip()
)
manual_benchmark_confirmation = st.checkbox(
    "I confirm this benchmark is a reasonable proxy for my actual holdings",
    value=False,
    help="Required when the holdings match failed, was weak, or the benchmark was selected manually.",
)
benchmark_validated = statistically_validated or manual_benchmark_confirmation
if statistically_validated:
    effective_beta = float(match_result["beta"])
    st.success(f"Benchmark validation gate: PASS. Matched portfolio beta: **{effective_beta:.2f}**.")
else:
    effective_beta = float(portfolio_beta_input)
    if benchmark_validated:
        st.success(f"Benchmark validation gate: PASS by explicit confirmation. Beta assumption: **{effective_beta:.2f}**.")
    else:
        st.error(
            "Benchmark validation gate: FAIL. The app will calculate drawdown statistics, "
            "but the final deployment recommendation will remain ₹0."
        )

source_mode = st.radio(
    "Price-data source",
    ["Automatic — NSE official, then Yahoo fallback", "NSE official", "Yahoo/yfinance", "Upload benchmark CSV"],
    horizontal=True,
)
benchmark_csv = None
if source_mode == "Upload benchmark CSV":
    benchmark_csv = st.file_uploader("Upload Date/Close benchmark CSV", type=["csv"], key="benchmark_csv")

try:
    with st.spinner("Loading benchmark history…"):
        if source_mode == "Upload benchmark CSV":
            if benchmark_csv is None:
                st.warning("Upload a benchmark CSV to continue.")
                st.stop()
            benchmark = parse_benchmark_csv(benchmark_csv)
            data_source_label = "Uploaded benchmark CSV"
        else:
            provider = {
                "Automatic — NSE official, then Yahoo fallback": "Automatic",
                "NSE official": "NSE official",
                "Yahoo/yfinance": "Yahoo/yfinance",
            }[source_mode]
            if custom_symbol.strip() and provider == "Automatic":
                provider = "Yahoo/yfinance"
            benchmark, data_source_label = cached_benchmark(
                selected["nse_name"], selected_symbol, start_download, end_download, provider
            )
except Exception as exc:
    st.error(f"Benchmark data could not be loaded: {exc}")
    st.info("Use the CSV option with Date and Close columns if a cloud host blocks NSE or Yahoo.")
    st.stop()

latest_benchmark_date = pd.Timestamp(benchmark.index.max())
st.caption(
    f"Loaded {len(benchmark):,} observations from **{data_source_label}**; "
    f"latest closing date: **{latest_benchmark_date:%d %b %Y}**."
)

risk_window_start = latest_benchmark_date - pd.DateOffset(years=lookback_years)
risk_prices = benchmark.loc[benchmark.index >= risk_window_start]
if len(risk_prices) < 100:
    st.error("The selected benchmark has too little history for the chosen lookback.")
    st.stop()

lookback_mdd = max_drawdown(risk_prices)
all_history_mdd = max_drawdown(benchmark)
if stress_method == "Selected-lookback maximum":
    chosen_stress = lookback_mdd
elif stress_method == "Downloaded-history maximum":
    chosen_stress = all_history_mdd
else:
    chosen_stress = float(custom_stress)

recommendation = recommend_dry_powder(
    prices=risk_prices,
    all_prices=benchmark,
    tolerance_drawdown=tolerance,
    policy_floor=POLICY_FLOORS[policy_name],
    stress_drawdown=chosen_stress,
    portfolio_beta=effective_beta,
)
planning_reserve_amount = float(portfolio_value) * recommendation.recommended_weight
planning_gap = float(current_dry_powder) - planning_reserve_amount

st.header("2. Drawdown-control reserve estimate — planning target, not deployable cash")
reserve_cols = st.columns(7)
reserve_cols[0].metric("Planning reserve estimate", pct(recommendation.recommended_weight), money(planning_reserve_amount))
reserve_cols[1].metric("Actual funded cycle reserve", money(cycle_start_reserve))
reserve_cols[2].metric("Current dry powder", money(current_dry_powder))
reserve_cols[3].metric("Planning gap", money(planning_gap))
reserve_cols[4].metric("Lookback max DD", pct(recommendation.lookback_max_drawdown))
reserve_cols[5].metric("Downloaded-history max DD", pct(recommendation.all_history_max_drawdown))
reserve_cols[6].metric("Portfolio stress used", pct(recommendation.portfolio_stress_drawdown))

st.warning(
    "This reserve figure answers a narrow drawdown-control question. It is not an optimal allocation, "
    "a mandatory cash balance or the source used for today's deployment ladder."
)
st.write(
    f"Using a **{pct(recommendation.index_stress_drawdown)} benchmark shock**, beta **{effective_beta:.2f}**, "
    f"and a **{pct(tolerance)} portfolio drawdown tolerance**, the formula estimates "
    f"**{pct(recommendation.formula_minimum_weight)}** outside equity."
)

scenario_values = sorted(set([0.30, 0.35, lookback_mdd, all_history_mdd, 0.45, 0.50]))
scenario_rows = []
for stress in scenario_values:
    scenario_rows.append(
        {
            "Benchmark stress": pct(stress),
            "Beta-adjusted portfolio stress": pct(min(stress * effective_beta, 0.99)),
            "Reserve estimate": pct(required_reserve_weight(stress, tolerance, effective_beta, cap=0.95)),
            "Reserve amount": money(
                portfolio_value * required_reserve_weight(stress, tolerance, effective_beta, cap=0.95)
            ),
        }
    )
st.subheader("Stress-scenario range")
st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)
st.caption(
    f"Selected lookback window: {risk_prices.index.min():%d %b %Y} to {risk_prices.index.max():%d %b %Y}. "
    "A 15-year window does not include 2008; the all-history and custom scenarios help expose that limitation."
)

if cash_yield > 0.08 and not reserve_is_liquid:
    st.error(
        f"The assumed dry-powder yield is {pct(cash_yield)} but the reserve instrument has not been confirmed as "
        "capital-stable and immediately liquid. The final deployment recommendation will be blocked."
    )
elif reserve_is_liquid:
    st.success("Reserve-instrument liquidity gate: PASS.")
else:
    st.warning("Confirm that the reserve instrument is capital-stable and immediately liquid before deployment.")

chart_df = pd.DataFrame({"Benchmark": risk_prices / risk_prices.iloc[0] * 100})
fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Benchmark"], name="Growth of 100"))
fig_risk.update_layout(
    title=f"{selected_name}: history used for the current risk estimate",
    yaxis_title="Index level (start = 100)",
)
st.plotly_chart(fig_risk, use_container_width=True)

st.header("3. Latest closing-price deployment decision")
st.write(
    "The app now separates three questions: **liquidity status**, **mechanical drawdown permission**, "
    "and **valuation confirmation**. A crossed threshold alone is not a buy decision."
)

reference_mode = st.selectbox(
    "Drawdown reference peak",
    ["52-week peak", "Current calendar-year peak", "Risk-lookback peak"],
    index=1,
)
latest_date = pd.Timestamp(benchmark.index.max())
if reference_mode == "52-week peak":
    reference_start = latest_date - pd.Timedelta(days=365)
elif reference_mode == "Current calendar-year peak":
    reference_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
else:
    reference_start = risk_window_start
reference_prices = benchmark.loc[(benchmark.index >= reference_start) & (benchmark.index <= latest_date)]

valuation_status = st.selectbox(
    "Valuation and business-quality review",
    [
        "Not reviewed",
        "Reasonable or undervalued — margin of safety exists",
        "Approximately fair value — deploy only half of mechanical permission",
        "Expensive — no margin of safety",
        "Fundamentals impaired or investment thesis broken",
    ],
    help="A drawdown is not automatically undervaluation. Review earnings, balance sheet, valuation and the investment thesis.",
)
if valuation_status.startswith("Reasonable"):
    valuation_multiplier = 1.0
elif valuation_status.startswith("Approximately"):
    valuation_multiplier = 0.5
else:
    valuation_multiplier = 0.0

live = recommend_live_deployment(
    prices=reference_prices,
    cycle_start_reserve=cycle_start_reserve,
    current_dry_powder=current_dry_powder,
    already_deployed=already_deployed,
    thresholds=thresholds,
    weights=ladder_weights,
)

staleness_days = max((pd.Timestamp.today().normalize() - live.latest_date.normalize()).days, 0)
data_fresh = staleness_days <= 7
liquidity_gate = emergency_adequate and cycle_reconciled and reserve_is_liquid
all_hard_gates = liquidity_gate and benchmark_validated and data_fresh
valuation_adjusted_permission = live.mechanical_deploy_now * valuation_multiplier
final_deploy_now = valuation_adjusted_permission if all_hard_gates else 0.0
final_keep_liquid = max(float(current_dry_powder) - final_deploy_now, 0.0)
post_deployment_equity = current_equity_value + final_deploy_now

status_cols = st.columns(3)
status_cols[0].metric(
    "Liquidity status",
    "PASS" if liquidity_gate else "BLOCKED",
    "Emergency + reconciliation + liquidity",
)
status_cols[1].metric(
    "Mechanical permission",
    money(live.mechanical_deploy_now),
    live.status,
)
status_cols[2].metric(
    "Final deploy recommendation",
    money(final_deploy_now),
    valuation_status,
)

live_cols = st.columns(8)
live_cols[0].metric("Current drawdown", pct(live.current_drawdown))
live_cols[1].metric("Crossed triggers", f"{len(live.triggers_crossed)} / {len(thresholds)}")
live_cols[2].metric("Funded cycle reserve", money(cycle_start_reserve))
live_cols[3].metric("Ladder target deployed", money(live.target_cumulative_deployment))
live_cols[4].metric("Already deployed", money(already_deployed))
live_cols[5].metric("Keep liquid", money(final_keep_liquid))
live_cols[6].metric("Equity after action", money(post_deployment_equity))
live_cols[7].metric("Latest data", f"{live.latest_date:%d %b %Y}")

if live.overdeployment_amount > 0:
    st.info(
        f"You are already **{money(live.overdeployment_amount)} ahead of the funded-reserve ladder**. "
        f"Mechanical deployment is ₹0; retain the remaining **{money(current_dry_powder)}**."
    )

gate_failures = []
if not emergency_adequate:
    gate_failures.append("emergency fund is below requirement")
if not cycle_reconciled:
    gate_failures.append("cycle reserve does not reconcile")
if not reserve_is_liquid:
    gate_failures.append("reserve instrument liquidity is unconfirmed")
if not benchmark_validated:
    gate_failures.append("benchmark is not validated")
if not data_fresh:
    gate_failures.append(f"latest closing data is {staleness_days} days old")
if valuation_multiplier == 0:
    gate_failures.append("valuation/business-quality review does not permit deployment")

if final_deploy_now > 0:
    st.success(
        f"**Final recommendation: deploy {money(final_deploy_now)} and retain {money(final_keep_liquid)}.** "
        f"The benchmark is {pct(live.current_drawdown)} below its {reference_mode.lower()} "
        f"of {live.peak_price:,.2f} on {live.peak_date:%d %b %Y}. "
        f"The funded-reserve ladder permits {money(live.mechanical_deploy_now)}; "
        f"the valuation gate applies a {valuation_multiplier:.0%} multiplier."
    )
else:
    reason_text = "; ".join(gate_failures) if gate_failures else live.status.lower()
    st.warning(
        f"**Final recommendation: deploy ₹0 and retain {money(current_dry_powder)}.** "
        f"Reason: {reason_text}."
    )

next_trigger_text = pct(live.next_trigger) if live.next_trigger is not None else "All configured triggers crossed"
st.caption(f"Next trigger: **{next_trigger_text}**. Closing-price signal only; not an intraday quote.")

ladder_rows = []
cumulative_weight = 0.0
for trigger, weight in zip(thresholds, ladder_weights):
    cumulative_weight += weight
    cumulative_target = cycle_start_reserve * cumulative_weight
    if live.current_drawdown >= trigger:
        state = "Crossed"
    elif trigger == live.next_trigger:
        state = "Next"
    else:
        state = "Waiting"
    ladder_rows.append(
        {
            "Drawdown trigger": pct(trigger),
            "Reserve share at trigger": pct(weight),
            "Tranche from funded reserve": money(cycle_start_reserve * weight),
            "Cumulative funded-reserve target": money(cumulative_target),
            "Status": state,
        }
    )
st.dataframe(pd.DataFrame(ladder_rows), use_container_width=True, hide_index=True)

live_drawdowns = drawdown_series(reference_prices) * 100
fig_live = go.Figure()
fig_live.add_trace(go.Scatter(x=live_drawdowns.index, y=live_drawdowns, name="Drawdown from running peak"))
for trigger in thresholds:
    fig_live.add_hline(y=-trigger * 100, line_dash="dot", annotation_text=f"{trigger * 100:g}%")
fig_live.update_layout(
    title=f"{selected_name}: closing-price drawdown path",
    yaxis_title="Drawdown (%)",
)
st.plotly_chart(fig_live, use_container_width=True)

st.header("4. Look-ahead-safe completed-year test")
years = available_completed_years(benchmark, count=15)
if not years:
    st.error("No complete calendar year is available in the loaded data.")
    st.stop()
selected_year = st.selectbox(
    "Calendar year",
    years,
    index=0,
    format_func=year_label,
    key="calendar_year_selector_buffett_v1",
)

history_cutoff = selected_year[0] - pd.Timedelta(days=1)
historical_start = history_cutoff - pd.DateOffset(years=lookback_years)
pre_year_lookback = benchmark.loc[
    (benchmark.index >= historical_start) & (benchmark.index <= history_cutoff)
]
pre_year_all = benchmark.loc[benchmark.index <= history_cutoff]
if len(pre_year_lookback) < 100:
    st.error(
        f"Insufficient pre-{selected_year[0].year} history to calculate a look-ahead-safe reserve. "
        "Select a more recent year or upload longer history."
    )
    st.stop()

pre_lookback_mdd = max_drawdown(pre_year_lookback)
pre_all_mdd = max_drawdown(pre_year_all)
if stress_method == "Selected-lookback maximum":
    backtest_stress = pre_lookback_mdd
elif stress_method == "Downloaded-history maximum":
    backtest_stress = pre_all_mdd
else:
    backtest_stress = float(custom_stress)

backtest_rec = recommend_dry_powder(
    prices=pre_year_lookback,
    all_prices=pre_year_all,
    tolerance_drawdown=tolerance,
    policy_floor=POLICY_FLOORS[policy_name],
    stress_drawdown=backtest_stress,
    portfolio_beta=1.0,
)

result, summary, events = run_dry_powder_backtest(
    prices=benchmark,
    start_date=selected_year[0],
    end_date=selected_year[1],
    initial_capital=portfolio_value,
    dry_powder_weight=backtest_rec.recommended_weight,
    thresholds=thresholds,
    weights=ladder_weights,
    annual_cash_yield=cash_yield,
)
_, zero_yield_summary, _ = run_dry_powder_backtest(
    prices=benchmark,
    start_date=selected_year[0],
    end_date=selected_year[1],
    initial_capital=portfolio_value,
    dry_powder_weight=backtest_rec.recommended_weight,
    thresholds=thresholds,
    weights=ladder_weights,
    annual_cash_yield=0.0,
)
cash_yield_value_add = summary.strategy_end - zero_yield_summary.strategy_end

st.success(
    f"No look-ahead: the {selected_year[0].year} starting reserve was calculated only from data available through "
    f"**{history_cutoff:%d %b %Y}**. Because the simulated equity asset is the benchmark itself, the historical test uses beta 1.00. Frozen starting reserve: "
    f"**{pct(backtest_rec.recommended_weight)} ({money(summary.reserve_start)})**."
)

if selected["kind"] == "Index":
    st.warning(
        "The selected series is an index and may exclude dividends, while the reserve earns interest. "
        "This can structurally favour the dry-powder strategy. Prefer an adjusted ETF/total-return proxy or upload TRI data."
    )

bt = st.columns(6)
bt[0].metric("Fully invested end", money(summary.fully_invested_end), pct(summary.fully_invested_return))
bt[1].metric("Strategy end", money(summary.strategy_end), pct(summary.strategy_return))
bt[2].metric("Vs fully invested", money(summary.opportunity_cost_or_gain))
bt[3].metric(
    "Max-DD reduction",
    f"{(summary.fully_invested_max_drawdown - summary.strategy_max_drawdown) * 100:.1f} pp",
)
bt[4].metric("Cash deployed", money(summary.cash_deployed), f"{summary.triggers_hit} trigger(s)")
bt[5].metric("Cash-yield value added", money(cash_yield_value_add), f"Assumption: {pct(cash_yield)}")

st.subheader("Performance attribution")
attr = st.columns(3)
attr[0].metric("Initial-equity gain/loss", money(summary.initial_equity_gain))
attr[1].metric("Cash interest accrued", money(summary.cash_interest_earned))
attr[2].metric("Deployment vs leaving reserve idle", money(summary.deployment_value_add))

normalized = result[["Fully invested", "Dry powder deployed", "Dry powder kept idle"]] / portfolio_value * 100
fig = go.Figure()
for column in normalized.columns:
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[column], name=column))
fig.update_layout(
    title=f"Portfolio paths during {year_label(selected_year)}",
    yaxis_title="Portfolio value (start = 100)",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Deployment events")
if events.empty:
    st.caption("No deployment event occurred in the selected year.")
else:
    event_display = events.copy()
    event_display["Trigger"] = (event_display["Trigger"] * 100).round(1).astype(str) + "%"
    event_display["Ladder weight"] = (event_display["Ladder weight"] * 100).round(1).astype(str) + "%"
    event_display["Invested"] = event_display["Invested"].map(money)
    event_display["Index price"] = event_display["Index price"].round(2)
    event_display["Units bought"] = event_display["Units bought"].round(4)
    st.dataframe(event_display, use_container_width=True, hide_index=True)

export = result.reset_index().copy()
export.insert(1, "Benchmark", selected_name)
export.insert(2, "Symbol", selected_symbol)
export.insert(3, "Lookahead safe reserve weight", backtest_rec.recommended_weight)
st.download_button(
    "Download annual simulation CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"dry_powder_{selected_year[0].year}.csv",
    mime="text/csv",
)

st.header("5. Interpretation")
st.write(
    "The planning reserve estimate controls a hypothetical drawdown; the funded cycle reserve controls the deployment ladder. "
    "They are deliberately separate. The final recommendation can only be positive when emergency liquidity is adequate, "
    "the cycle reserve reconciles, the reserve is genuinely liquid, the benchmark is validated, closing data is fresh, "
    "and valuation/business quality provides a margin of safety."
)
st.caption(
    "Educational tool only. Review taxes, credit risk, exit loads, liquidity, dividends, portfolio concentration, "
    "fundamentals and valuation before acting."
)
