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


@dataclass(frozen=True)
class LiveDeploymentRecommendation:
    latest_date: pd.Timestamp
    latest_price: float
    peak_date: pd.Timestamp
    peak_price: float
    current_drawdown: float
    triggers_crossed: tuple[float, ...]
    target_cumulative_deployment: float
    deploy_now: float
    keep_as_dry_powder: float
    next_trigger: float | None
    tranche_amount: float
    status: str


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


def recommend_live_deployment(
    prices: pd.Series,
    target_reserve: float,
    current_dry_powder: float,
    already_deployed: float,
    thresholds: list[float],
) -> LiveDeploymentRecommendation:
    """Return an auditable tranche recommendation from the latest closing drawdown.

    Each threshold receives an equal share of the target reserve. The recommendation
    is cumulative: at a 10% drawdown with triggers 5%, 10%, 15%, 20%, half of the
    target reserve should have been deployed. Any amount the user says was already
    deployed is subtracted, preventing duplicate deployment.
    """
    series = prices.dropna().astype(float).sort_index()
    if len(series) < 2:
        raise ValueError("At least two valid prices are required for a live deployment signal.")

    cleaned_thresholds = sorted(set(float(x) for x in thresholds if 0 < float(x) < 1))
    if not cleaned_thresholds:
        raise ValueError("At least one deployment trigger is required.")

    latest_date = pd.Timestamp(series.index[-1])
    latest_price = float(series.iloc[-1])
    peak_price = float(series.max())
    peak_candidates = series[series == peak_price]
    peak_date = pd.Timestamp(peak_candidates.index[-1])
    current_drawdown = max(0.0, 1.0 - latest_price / peak_price)

    target_reserve = max(float(target_reserve), 0.0)
    current_dry_powder = max(float(current_dry_powder), 0.0)
    already_deployed = max(float(already_deployed), 0.0)
    tranche_amount = target_reserve / len(cleaned_thresholds)
    crossed = tuple(t for t in cleaned_thresholds if current_drawdown >= t)
    target_cumulative = min(target_reserve, tranche_amount * len(crossed))
    required_now = max(0.0, target_cumulative - already_deployed)
    deploy_now = min(current_dry_powder, required_now)
    keep = max(0.0, current_dry_powder - deploy_now)
    next_trigger = next((t for t in cleaned_thresholds if t > current_drawdown), None)

    if deploy_now > 0:
        status = "DEPLOY A TRANCHE"
    elif crossed and already_deployed >= target_cumulative:
        status = "HOLD — CROSSED TRIGGERS ALREADY FUNDED"
    else:
        status = "HOLD DRY POWDER"

    return LiveDeploymentRecommendation(
        latest_date=latest_date,
        latest_price=latest_price,
        peak_date=peak_date,
        peak_price=peak_price,
        current_drawdown=current_drawdown,
        triggers_crossed=crossed,
        target_cumulative_deployment=target_cumulative,
        deploy_now=deploy_now,
        keep_as_dry_powder=keep,
        next_trigger=next_trigger,
        tranche_amount=tranche_amount,
        status=status,
    )


def available_completed_years(
    prices: pd.Series,
    count: int = 15,
    minimum_observations: int = 180,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return completed calendar years with enough data for a meaningful test.

    The current calendar year is excluded. A year must contain enough trading
    observations and data in both the first and final quarters, which prevents
    a partially uploaded year from being treated as complete.
    """
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
    annual_cash_yield: float = 0.0,
) -> tuple[pd.DataFrame, BacktestSummary, pd.DataFrame]:
    q = prices.loc[(prices.index >= start_date) & (prices.index <= end_date)].dropna().astype(float)
    if len(q) < 2:
        raise ValueError("Not enough benchmark observations in the selected year.")

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
        attempted = ", ".join(holdings["ticker"].astype(str).tolist()[:8])
        raise ValueError(
            "None of the uploaded holding tickers returned enough Yahoo price history. "
            f"Tried: {attempted}. Bond, ISIN-style, delisted or unsupported BSE symbols may not be available; "
            "select the closest benchmark manually or use supported NSE equity tickers."
        )

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
    return download_benchmark_close(nse_name=nse_name, yahoo_symbol=symbol, start=start, end=end, provider=provider)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_many(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return download_many(symbols=symbols, start=start, end=end)


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_threshold_text(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip().replace("%", "")
        if not token:
            continue
        value = float(token) / 100.0
        if not 0 < value < 1:
            raise ValueError("Each deployment trigger must be between 0% and 100%.")
        values.append(value)
    if not values:
        raise ValueError("Enter at least one deployment trigger.")
    return sorted(set(values))


def year_label(item: tuple[pd.Timestamp, pd.Timestamp]) -> str:
    start, end = item
    return f"{start.year} — {start:%d %b} to {end:%d %b}"


def render_matcher(start: date, end: date) -> str | None:
    st.subheader("Find the benchmark that actually resembles your holdings")
    uploaded = st.file_uploader(
        "Upload holdings CSV",
        type=["csv"],
        help="Required: ticker/symbol plus value/amount or weight/allocation. NSE tickers may be entered without .NS.",
    )
    if uploaded is None:
        st.caption("Use the included sample_holdings.csv as a template. Until a file is uploaded, select the index manually below.")
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
            scores = score_benchmarks(portfolio_returns, prices[[s for s in candidate_symbols if s in prices.columns]])

        if missing:
            st.warning(f"Skipped {len(missing)} ticker(s) with insufficient data: {', '.join(missing[:8])}")
        if scores.empty:
            st.error("No candidate had enough overlapping data. Select different candidates or use manual selection.")
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
        best_symbol = str(scores.iloc[0]["symbol"])
        best_name = reverse_names.get(best_symbol, best_symbol)
        st.success(f"Best statistical match in the tested set: **{best_name}**. You can still override it.")
        return best_name if best_name in INDEX_CATALOG else None
    except Exception as exc:
        st.error(f"Could not analyse holdings: {exc}")
        return None


st.title("🛡️ Dry Powder Planner for Indian Equity Portfolios")
st.write(
    "Choose or statistically match an NSE benchmark, estimate a transparent minimum cash reserve, "
    "track net worth and current liquidity, get an instant rule-based deployment amount, and test the reserve during a completed calendar year."
)
st.info(
    "Dry powder is investable liquidity—not your emergency fund. This app does not model taxes, brokerage, "
    "slippage, debt-fund taxation, or option hedges. Market data may be delayed or revised."
)

with st.sidebar:
    st.header("1. Net worth and liquidity")
    portfolio_value = st.number_input(
        "Total investable portfolio — equity + dry powder (₹)",
        min_value=100_000,
        max_value=1_000_000_000,
        value=2_000_000,
        step=50_000,
        help="This is the capital covered by the dry-powder plan. Do not add the current dry powder again outside this amount.",
    )
    current_dry_powder = st.number_input(
        "Current deployable dry powder (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=0.0,
        step=5_000.0,
        help="Liquid capital available to invest now. Exclude the emergency fund.",
        key="sidebar_current_dry_powder_v2",
    )
    already_deployed = st.number_input(
        "Already deployed in the current drawdown cycle (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=0.0,
        step=5_000.0,
        help="Capital already invested after declines from the selected live reference peak.",
        key="sidebar_already_deployed_v2",
    )
    emergency_fund = st.number_input(
        "Emergency fund — never deploy (₹)",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=0.0,
        step=10_000.0,
    )
    other_assets = st.number_input(
        "Other net-worth assets (₹)",
        min_value=0.0,
        max_value=10_000_000_000.0,
        value=0.0,
        step=50_000.0,
        help="Examples: EPF, PPF, property equity, gold or other assets not included in the investable portfolio.",
    )
    liabilities = st.number_input(
        "Outstanding liabilities (₹)",
        min_value=0.0,
        max_value=10_000_000_000.0,
        value=0.0,
        step=50_000.0,
    )

    st.divider()
    st.header("2. Strategy assumptions")
    lookback_years = st.slider("Risk lookback", 3, 15, 15)
    tolerance = st.slider("Maximum drawdown you can tolerate", 5, 50, 23, 1) / 100
    policy_options = list(POLICY_FLOORS)
    policy_name = st.selectbox(
        "Reserve policy",
        policy_options,
        index=policy_options.index("Formula only — no policy floor"),
    )
    stress_method = st.radio("Stress basis", ["Historical maximum drawdown", "Custom crash assumption"])
    custom_stress = None
    if stress_method == "Custom crash assumption":
        custom_stress = st.slider("Assumed benchmark crash", 10, 70, 35, 1) / 100
    cash_yield = st.number_input("Annual yield on dry powder (%)", 0.0, 20.0, 11.0, 0.25) / 100
    trigger_text = st.text_input(
        "Deploy at drawdowns (%)",
        "2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20",
    )
    custom_symbol = st.text_input("Optional custom Yahoo Finance symbol", placeholder="Example: ^NSEI or an ETF ticker")

if current_dry_powder > portfolio_value:
    st.error("Current deployable dry powder cannot exceed the total investable portfolio.")
    st.stop()
if already_deployed > portfolio_value:
    st.error("Already deployed capital cannot exceed the total investable portfolio.")
    st.stop()

current_equity_value = max(float(portfolio_value) - float(current_dry_powder), 0.0)
calculated_net_worth = float(portfolio_value) + float(emergency_fund) + float(other_assets) - float(liabilities)

st.header("1. Net worth snapshot")
networth_cols = st.columns(6)
networth_cols[0].metric("Calculated net worth", money(calculated_net_worth))
networth_cols[1].metric("Investable portfolio", money(portfolio_value))
networth_cols[2].metric("Currently invested", money(current_equity_value))
networth_cols[3].metric(
    "Current dry powder",
    money(current_dry_powder),
    pct(current_dry_powder / portfolio_value) if portfolio_value else "0.0%",
)
networth_cols[4].metric("Emergency fund", money(emergency_fund), "Excluded")
networth_cols[5].metric("Liabilities", money(liabilities))
st.caption(
    "Calculated net worth = investable portfolio + emergency fund + other assets − liabilities. "
    "Only current deployable dry powder is used by the live deployment engine."
)

end_download = date.today() + timedelta(days=1)
start_download = date.today() - timedelta(days=365 * lookback_years + 120)

mode = st.radio(
    "Benchmark selection mode",
    ["Choose manually", "Match using holdings CSV"],
    horizontal=True,
)
matched_name = None
if mode == "Match using holdings CSV":
    matched_name = render_matcher(start_download, end_download)

index_names = list(INDEX_CATALOG)
default_index = matched_name or "NIFTY 50"
selected_name = st.selectbox(
    "Benchmark for risk and backtest",
    index_names,
    index=index_names.index(default_index),
)
selected = INDEX_CATALOG[selected_name]
selected_symbol = custom_symbol.strip() or selected["symbol"]
st.caption(f"**{selected_name}** · {selected['kind']} · {selected['description']} · Data symbol: `{selected_symbol}`")

source_mode = st.radio(
    "Price-data source",
    ["Automatic — NSE official, then Yahoo fallback", "NSE official", "Yahoo/yfinance", "Upload benchmark CSV"],
    horizontal=True,
)
benchmark_csv = None
if source_mode == "Upload benchmark CSV":
    benchmark_csv = st.file_uploader("Upload Date/Close benchmark CSV", type=["csv"], key="benchmark_csv")

try:
    thresholds = parse_threshold_text(trigger_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

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
    st.info("Use the CSV option with Date and Close columns when a cloud host blocks NSE or Yahoo requests.")
    st.stop()

st.caption(f"Loaded {len(benchmark):,} observations from **{data_source_label}**; latest date: **{benchmark.index.max():%d %b %Y}**.")

risk_window_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)
risk_prices = benchmark.loc[benchmark.index >= risk_window_start]
if len(risk_prices) < 100:
    st.error("The selected benchmark has too little history for the chosen lookback. Reduce the lookback or choose another benchmark.")
    st.stop()

recommendation = recommend_dry_powder(
    prices=risk_prices,
    tolerance_drawdown=tolerance,
    policy_floor=POLICY_FLOORS[policy_name],
    custom_stress_drawdown=custom_stress,
)
recommended_amount = portfolio_value * recommendation.recommended_weight

st.header("2. Required dry-powder reserve")
reserve_gap = float(current_dry_powder) - float(recommended_amount)
reserve_coverage = (float(current_dry_powder) / float(recommended_amount)) if recommended_amount > 0 else 1.0
reserve_status = "Surplus" if reserve_gap >= 0 else "Shortfall"

cols = st.columns(7)
cols[0].metric("Recommended reserve", pct(recommendation.recommended_weight), money(recommended_amount))
cols[1].metric("Current dry powder", money(current_dry_powder), pct(current_dry_powder / portfolio_value))
cols[2].metric(f"Reserve {reserve_status.lower()}", money(abs(reserve_gap)), reserve_status)
cols[3].metric("Reserve coverage", pct(reserve_coverage))
cols[4].metric("Historical max drawdown", pct(recommendation.historical_max_drawdown))
cols[5].metric("Your tolerance", pct(recommendation.tolerance_drawdown))
cols[6].metric("Estimated stressed DD", pct(recommendation.estimated_stress_drawdown))

if reserve_gap < 0:
    st.warning(
        f"Your current dry powder is **{money(abs(reserve_gap))} below** the calculated starting reserve. "
        "Build this gap from future contributions or planned rebalancing rather than selling impulsively."
    )
else:
    st.success(
        f"Your current dry powder is **{money(reserve_gap)} above** the calculated starting reserve. "
        "The live engine will still deploy only the tranche amount justified by crossed triggers."
    )

st.write(
    f"The formula-only minimum is **{pct(recommendation.formula_minimum_weight)}**. "
    f"After applying your **{policy_name.split('—')[0].strip().lower()}** reserve floor and rounding upward, "
    f"the app suggests maintaining **{money(recommended_amount)}** as investable dry powder."
)
with st.expander("How the recommendation is calculated"):
    st.latex(r"\text{Minimum cash weight} = \max\left(0, 1 - \frac{\text{tolerable drawdown}}{\text{stress drawdown}}\right)")
    st.write(
        "This assumes the selected benchmark falls by the stress amount while cash remains broadly stable. "
        "It is an allocation estimate, not a guarantee. A concentrated portfolio may fall more than its benchmark."
    )

chart_df = pd.DataFrame({"Benchmark": risk_prices / risk_prices.iloc[0] * 100, "Drawdown": drawdown_series(risk_prices) * 100})
fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Benchmark"], name="Growth of 100"))
fig_risk.update_layout(title=f"{selected_name}: normalized history used for risk estimate", yaxis_title="Index level (start = 100)")
st.plotly_chart(fig_risk, use_container_width=True)

st.header("3. Current live dry-powder decision")
st.write(
    "This is a rules-based signal from the latest available closing price. It does not predict the market bottom. "
    "Each crossed drawdown trigger unlocks one equal tranche of the recommended reserve."
)

reference_mode = st.selectbox(
    "Drawdown reference peak",
    ["52-week peak", "Current calendar-year peak", "Risk-lookback peak"],
    index=0,
    help="The current drawdown is measured from the highest close in this reference window.",
)
latest_date = pd.Timestamp(benchmark.index.max())
if reference_mode == "52-week peak":
    reference_start = latest_date - pd.Timedelta(days=365)
elif reference_mode == "Current calendar-year peak":
    reference_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
else:
    reference_start = risk_window_start
reference_prices = benchmark.loc[(benchmark.index >= reference_start) & (benchmark.index <= latest_date)]

st.caption(
    f"Live inputs: **{money(current_dry_powder)} available now** and **{money(already_deployed)} already deployed** "
    "in the current decline. Edit both instantly from the sidebar."
)

live = recommend_live_deployment(
    prices=reference_prices,
    target_reserve=recommended_amount,
    current_dry_powder=current_dry_powder,
    already_deployed=already_deployed,
    thresholds=thresholds,
)

staleness_days = max((pd.Timestamp.today().normalize() - live.latest_date.normalize()).days, 0)
next_trigger_text = pct(live.next_trigger) if live.next_trigger is not None else "All crossed"
post_deployment_equity = current_equity_value + live.deploy_now
live_metrics = st.columns(8)
live_metrics[0].metric("Current drawdown", pct(live.current_drawdown))
live_metrics[1].metric("Triggers crossed", f"{len(live.triggers_crossed)} / {len(thresholds)}")
live_metrics[2].metric("Deploy now", money(live.deploy_now))
live_metrics[3].metric("Keep liquid", money(live.keep_as_dry_powder))
live_metrics[4].metric("Equity after deployment", money(post_deployment_equity))
live_metrics[5].metric("Target deployed", money(live.target_cumulative_deployment))
live_metrics[6].metric("Next trigger", next_trigger_text)
live_metrics[7].metric("Latest data", f"{live.latest_date:%d %b %Y}")

if staleness_days > 7:
    st.error(
        f"The latest price is {staleness_days} days old. Treat the amount below as indicative only and refresh the data "
        "before deploying capital."
    )

if live.deploy_now > 0:
    st.warning(
        f"**Rule signal: {live.status}.** The selected benchmark is {pct(live.current_drawdown)} below its "
        f"{reference_mode.lower()} of {live.peak_price:,.2f} on {live.peak_date:%d %b %Y}. "
        f"Your crossed triggers call for cumulative deployment of {money(live.target_cumulative_deployment)}. "
        f"After subtracting {money(already_deployed)} already deployed, invest **{money(live.deploy_now)} now** and "
        f"retain **{money(live.keep_as_dry_powder)}**."
    )
else:
    reason = (
        "the first trigger has not been reached"
        if not live.triggers_crossed
        else "the cumulative amount required by crossed triggers has already been deployed"
    )
    st.info(
        f"**Rule signal: {live.status}. Deploy ₹0 now** because {reason}. "
        f"Keep **{money(live.keep_as_dry_powder)}** available. "
        + (f"The next trigger is {pct(live.next_trigger)}." if live.next_trigger is not None else "All configured triggers are crossed.")
    )

trigger_rows = []
for position, trigger in enumerate(thresholds, start=1):
    cumulative_target = recommended_amount * position / len(thresholds)
    if live.current_drawdown >= trigger:
        state = "Crossed"
    elif trigger == live.next_trigger:
        state = "Next"
    else:
        state = "Waiting"
    trigger_rows.append(
        {
            "Drawdown trigger": pct(trigger),
            "Tranche": money(live.tranche_amount),
            "Cumulative target deployed": money(cumulative_target),
            "Status": state,
        }
    )
st.dataframe(pd.DataFrame(trigger_rows), use_container_width=True, hide_index=True)

live_drawdowns = drawdown_series(reference_prices) * 100
fig_live = go.Figure()
fig_live.add_trace(go.Scatter(x=live_drawdowns.index, y=live_drawdowns, name="Drawdown from running peak"))
for trigger in thresholds:
    fig_live.add_hline(y=-trigger * 100, line_dash="dot", annotation_text=f"{trigger * 100:g}% trigger")
fig_live.update_layout(
    title=f"{selected_name}: drawdown path for {reference_mode.lower()}",
    yaxis_title="Drawdown (%)",
)
st.plotly_chart(fig_live, use_container_width=True)

st.caption(
    "The amount is a mechanical allocation rule, not personalised investment advice. Use closing data, verify the benchmark "
    "still represents your holdings, and do not deploy emergency cash."
)

st.header("4. Completed-year dry-powder test")
years = available_completed_years(benchmark, count=15)
if not years:
    st.error(
        "No complete calendar year is available in the loaded data. "
        "Increase the risk lookback or upload a longer benchmark history."
    )
    st.stop()
selected_year = st.selectbox("Calendar year", years, index=0, format_func=year_label, key="calendar_year_selector_v2")

result, summary, events = run_dry_powder_backtest(
    prices=benchmark,
    start_date=selected_year[0],
    end_date=selected_year[1],
    initial_capital=portfolio_value,
    dry_powder_weight=recommendation.recommended_weight,
    thresholds=thresholds,
    annual_cash_yield=cash_yield,
)

m = st.columns(5)
m[0].metric("Fully invested end value", money(summary.fully_invested_end), pct(summary.fully_invested_return))
m[1].metric("Dry-powder strategy end", money(summary.strategy_end), pct(summary.strategy_return))
m[2].metric("Strategy vs fully invested", money(summary.opportunity_cost_or_gain))
m[3].metric("Max-DD reduction", f"{(summary.fully_invested_max_drawdown - summary.strategy_max_drawdown) * 100:.1f} pp")
m[4].metric("Cash deployed", money(summary.cash_deployed), f"{summary.triggers_hit} trigger(s)")

normalized = result[["Fully invested", "Dry powder deployed", "Dry powder kept idle"]] / portfolio_value * 100
fig = go.Figure()
for column in normalized.columns:
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[column], name=column))
fig.update_layout(title=f"Portfolio paths during {year_label(selected_year)}", yaxis_title="Portfolio value (start = 100)")
st.plotly_chart(fig, use_container_width=True)

if summary.triggers_hit == 0:
    st.warning(
        f"No deployment trigger was reached. The reserve reduced the strategy's maximum drawdown from "
        f"{pct(summary.fully_invested_max_drawdown)} to {pct(summary.strategy_max_drawdown)}, but keeping capital out "
        f"of equities changed the ending value by {money(summary.opportunity_cost_or_gain)} versus being fully invested."
    )
else:
    direction = "more" if summary.opportunity_cost_or_gain >= 0 else "less"
    st.success(
        f"The strategy deployed {money(summary.cash_deployed)} across {summary.triggers_hit} trigger(s). "
        f"Deployment added {money(summary.deployment_value_add)} versus leaving the reserve idle. "
        f"It ended with {money(abs(summary.opportunity_cost_or_gain))} {direction} than the fully invested portfolio, "
        f"while maximum drawdown changed from {pct(summary.fully_invested_max_drawdown)} to {pct(summary.strategy_max_drawdown)}."
    )

st.subheader("Deployment events")
if events.empty:
    st.caption("No event occurred in the selected year.")
else:
    event_display = events.copy()
    event_display["Trigger"] = (event_display["Trigger"] * 100).round(1).astype(str) + "%"
    event_display["Invested"] = event_display["Invested"].map(money)
    event_display["Index price"] = event_display["Index price"].round(2)
    event_display["Units bought"] = event_display["Units bought"].round(4)
    st.dataframe(event_display, use_container_width=True, hide_index=True)

export = result.reset_index().copy()
export.insert(1, "Benchmark", selected_name)
export.insert(2, "Symbol", selected_symbol)
st.download_button(
    "Download annual simulation CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"dry_powder_{selected_year[0].year}.csv",
    mime="text/csv",
)

st.header("5. Interpretation")
st.write(
    "A reserve is useful when it either prevents forced selling, reduces a drawdown to a level you can actually tolerate, "
    "or gives you capital to buy after preset declines. It is not automatically return-enhancing: in steadily rising years, "
    "cash usually creates an opportunity cost. The live recommendation should be followed as a pre-committed tranche rule, "
    "not as a prediction that the market has reached its bottom. Judge the policy across both difficult and strong years."
)
st.caption(
    "Educational tool only. Before acting, verify the selected benchmark, your portfolio's actual beta and concentration, "
    "the safety/liquidity of the reserve instrument, taxes, exit loads and transaction costs."
)
