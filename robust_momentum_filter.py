"""Point-in-time robust momentum prefilter for optimizer candidates.

The filter is intentionally independent of Streamlit and market-data providers.
Callers supply adjusted INR price histories that contain no observations beyond
the decision timestamp.  The optimizer can therefore test this module directly
and retain complete control over data provenance.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


REPORT_COLUMNS = [
    "Ticker",
    "Owned",
    "Stable Momentum Score",
    "Current Momentum Score",
    "Rank Instability",
    "3-1 INR Return",
    "6-1 INR Return",
    "12-1 INR Return",
    "Positive Horizons",
    "Below 200-Session Trend",
    "Eligible for Exclusion",
    "Excluded",
    "Reason",
]


def _clean_series(frame: pd.DataFrame, ticker: str) -> pd.Series:
    series = pd.to_numeric(frame[ticker], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return series.astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index().astype(float)
    return series[series > 0]


def _checkpoint_signal(
    series: pd.Series,
    *,
    checkpoint_sessions: int,
    lookback_sessions: int,
    skip_recent_sessions: int,
    annualization_sessions: int,
    volatility_floor_annual: float,
) -> tuple[float, float] | None:
    """Return (raw log return, volatility-adjusted return) at one checkpoint."""
    end_position = len(series) - 1 - int(checkpoint_sessions)
    start_position = end_position - int(lookback_sessions)
    signal_end_position = end_position - int(skip_recent_sessions)
    if start_position < 0 or signal_end_position <= start_position:
        return None

    start_price = float(series.iloc[start_position])
    signal_end_price = float(series.iloc[signal_end_position])
    if not (math.isfinite(start_price) and math.isfinite(signal_end_price)) or start_price <= 0 or signal_end_price <= 0:
        return None

    raw_log_return = float(math.log(signal_end_price / start_price))
    window = series.iloc[start_position : signal_end_position + 1]
    daily_returns = np.log(window / window.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily_returns) < 20:
        return None

    annualized_volatility = float(daily_returns.std(ddof=1) * math.sqrt(annualization_sessions))
    denominator = max(annualized_volatility, float(volatility_floor_annual))
    return raw_log_return, raw_log_return / denominator


def apply_robust_momentum_filter(
    inr_prices: pd.DataFrame,
    *,
    owned_tickers: Iterable[str] = (),
    config: Mapping[str, object],
) -> tuple[list[str], pd.DataFrame]:
    """Return retained tickers and an auditable robust-momentum report.

    Only zero-quantity candidates can be excluded.  The exclusion count is
    bounded by ``floor(candidate_count * maximum_exclusion_fraction)`` and a
    candidate must also have weak absolute momentum and a broken long trend.
    """
    if not isinstance(inr_prices, pd.DataFrame):
        raise TypeError("inr_prices must be a pandas DataFrame")

    tickers = [str(column) for column in inr_prices.columns]
    if not tickers:
        return [], pd.DataFrame(columns=REPORT_COLUMNS)

    owned = {str(ticker).strip().upper() for ticker in owned_tickers if str(ticker).strip()}
    if not bool(config.get("enabled", True)):
        report = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Owned": ticker.upper() in owned,
                    "Excluded": False,
                    "Reason": "FILTER_DISABLED",
                }
                for ticker in tickers
            ],
            columns=REPORT_COLUMNS,
        )
        return tickers, report

    lookbacks = tuple(int(value) for value in config["lookback_sessions"])
    checkpoints = tuple(int(value) for value in config["stability_checkpoint_sessions"])
    skip = int(config["skip_recent_sessions"])
    trend_lookback = int(config["trend_lookback_sessions"])
    annualization = int(config["annualization_sessions"])
    volatility_floor = float(config["volatility_floor_annual"])
    minimum_negative = int(config["minimum_negative_horizons"])
    maximum_fraction = float(config["maximum_exclusion_fraction"])

    adjusted_by_checkpoint: dict[int, pd.DataFrame] = {}
    raw_current: dict[str, dict[int, float]] = {ticker: {} for ticker in tickers}

    for checkpoint in checkpoints:
        rows = {}
        for ticker in tickers:
            series = _clean_series(inr_prices, ticker)
            values = {}
            for lookback in lookbacks:
                signal = _checkpoint_signal(
                    series,
                    checkpoint_sessions=checkpoint,
                    lookback_sessions=lookback,
                    skip_recent_sessions=skip,
                    annualization_sessions=annualization,
                    volatility_floor_annual=volatility_floor,
                )
                if signal is None:
                    continue
                raw_return, adjusted_return = signal
                values[lookback] = adjusted_return
                if checkpoint == 0:
                    raw_current[ticker][lookback] = raw_return
            rows[ticker] = values
        adjusted_by_checkpoint[checkpoint] = pd.DataFrame.from_dict(rows, orient="index").reindex(tickers)

    checkpoint_scores = pd.DataFrame(index=tickers, columns=checkpoints, dtype=float)
    for checkpoint, adjusted in adjusted_by_checkpoint.items():
        if adjusted.empty:
            continue
        ranks = adjusted.rank(axis=0, method="average", pct=True, na_option="keep")
        # A median prevents one abnormal horizon from dominating the decision.
        checkpoint_scores[checkpoint] = ranks.median(axis=1, skipna=False)

    stable_scores = checkpoint_scores.median(axis=1, skipna=False)
    current_scores = checkpoint_scores[checkpoints[0]]
    rank_instability = checkpoint_scores.max(axis=1, skipna=False) - checkpoint_scores.min(axis=1, skipna=False)

    candidate_tickers = [ticker for ticker in tickers if ticker.upper() not in owned]
    maximum_exclusions = int(math.floor(len(candidate_tickers) * maximum_fraction))
    scored_candidates = [ticker for ticker in candidate_tickers if pd.notna(stable_scores.get(ticker, np.nan))]
    bottom_candidate_set = set(
        sorted(scored_candidates, key=lambda ticker: (float(stable_scores[ticker]), ticker))[:maximum_exclusions]
    )

    rows = []
    eligible = []
    for ticker in tickers:
        series = _clean_series(inr_prices, ticker)
        raw = raw_current[ticker]
        negative_horizons = sum(float(raw.get(lookback, np.nan)) < 0 for lookback in lookbacks)
        positive_horizons = sum(float(raw.get(lookback, np.nan)) > 0 for lookback in lookbacks)

        below_trend = False
        if len(series) >= trend_lookback:
            trailing_average = float(series.iloc[-trend_lookback:].mean())
            below_trend = bool(float(series.iloc[-1]) < trailing_average)

        score = stable_scores.get(ticker, np.nan)
        score_available = bool(pd.notna(score))
        is_owned = ticker.upper() in owned
        can_exclude = bool(
            score_available
            and not is_owned
            and ticker in bottom_candidate_set
            and negative_horizons >= minimum_negative
            and below_trend
        )
        if can_exclude:
            eligible.append(ticker)

        reason = "ELIGIBLE_WEAK_MOMENTUM" if can_exclude else "MOMENTUM_RETAINED"
        if is_owned:
            reason = "OWNED_HOLDING_PROTECTED"
        elif not score_available:
            reason = "INSUFFICIENT_STABLE_MOMENTUM_HISTORY"
        elif negative_horizons < minimum_negative:
            reason = "ABSOLUTE_MOMENTUM_NOT_WEAK"
        elif not below_trend:
            reason = "LONG_TREND_NOT_BROKEN"

        rows.append(
            {
                "Ticker": ticker,
                "Owned": is_owned,
                "Stable Momentum Score": float(score) if score_available else np.nan,
                "Current Momentum Score": float(current_scores.get(ticker, np.nan)),
                "Rank Instability": float(rank_instability.get(ticker, np.nan)),
                "3-1 INR Return": float(raw.get(lookbacks[0], np.nan)),
                "6-1 INR Return": float(raw.get(lookbacks[1], np.nan)),
                "12-1 INR Return": float(raw.get(lookbacks[2], np.nan)),
                "Positive Horizons": int(positive_horizons),
                "Below 200-Session Trend": below_trend,
                "Eligible for Exclusion": can_exclude,
                "Excluded": False,
                "Reason": reason,
            }
        )

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if maximum_exclusions > 0 and eligible:
        selected_set = set(eligible)
        report.loc[report["Ticker"].isin(selected_set), "Excluded"] = True
        report.loc[report["Ticker"].isin(selected_set), "Reason"] = "BOTTOM_MOMENTUM_CONFIRMED"

    excluded = set(report.loc[report["Excluded"], "Ticker"].astype(str))
    retained = [ticker for ticker in tickers if ticker not in excluded]
    return retained, report
