"""Fast, cluster-aware preselection before full-history portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_scalable_candidates(
    close_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    ticker_clusters: dict[str, str],
    maximum_candidates: int = 400,
    minimum_per_cluster: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """Rank a broad universe using recent momentum, liquidity and data continuity.

    Ranking is performed within listing clusters before the global selection, so
    a large U.S. universe cannot erase smaller investable markets. This is only a
    computational preselection; the retained names still pass the complete INR,
    history, redundancy, risk and constrained-optimization funnel.
    """
    if close_history is None or close_history.empty:
        return [], pd.DataFrame()
    cap = max(int(maximum_candidates), 1)
    floor = max(int(minimum_per_cluster), 0)
    close = close_history.apply(pd.to_numeric, errors="coerce").sort_index()
    volume = (
        volume_history.apply(pd.to_numeric, errors="coerce").reindex_like(close)
        if volume_history is not None and not volume_history.empty
        else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    )
    rows = []
    for ticker in close.columns:
        prices = close[ticker].dropna()
        if len(prices) < 84 or not np.isfinite(prices.iloc[-1]) or prices.iloc[-1] <= 0:
            continue
        skip = min(21, max(len(prices) // 10, 1))
        endpoint = len(prices) - 1 - skip
        horizon_returns = []
        for horizon in (63, 126, 252):
            start = endpoint - horizon
            if start >= 0 and prices.iloc[start] > 0:
                horizon_returns.append(float(prices.iloc[endpoint] / prices.iloc[start] - 1.0))
        if not horizon_returns:
            continue
        daily = np.log(prices / prices.shift(1)).dropna().tail(252)
        annual_vol = float(daily.std(ddof=1) * np.sqrt(250)) if len(daily) > 1 else np.nan
        risk_scale = max(annual_vol, 0.05) if np.isfinite(annual_vol) else 0.50
        recent_volume = volume[ticker].dropna().tail(63) if ticker in volume else pd.Series(dtype=float)
        turnover = (
            float((close[ticker].reindex(recent_volume.index) * recent_volume).median())
            if not recent_volume.empty
            else 0.0
        )
        rows.append(
            {
                "Ticker": str(ticker).upper(),
                "Cluster": ticker_clusters.get(str(ticker).upper(), "Unknown"),
                "Momentum": float(np.median(horizon_returns) / risk_scale),
                "Turnover": max(turnover, 0.0),
                "Continuity": float(prices.count() / max(len(close.index), 1)),
                "Latest Price": float(prices.iloc[-1]),
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return [], report
    grouped = report.groupby("Cluster", dropna=False)
    report["Momentum rank"] = grouped["Momentum"].rank(pct=True, method="average")
    report["Liquidity rank"] = grouped["Turnover"].rank(pct=True, method="average")
    report["Continuity rank"] = grouped["Continuity"].rank(pct=True, method="average")
    report["Preselection score"] = (
        0.60 * report["Momentum rank"]
        + 0.30 * report["Liquidity rank"]
        + 0.10 * report["Continuity rank"]
    )
    report = report.sort_values(
        ["Preselection score", "Momentum", "Turnover", "Ticker"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(report) <= cap:
        report["Selected"] = True
        return report["Ticker"].tolist(), report

    selected: list[str] = []
    if floor:
        for _, cluster_rows in report.groupby("Cluster", sort=True, dropna=False):
            for ticker in cluster_rows.head(floor)["Ticker"]:
                if ticker not in selected and len(selected) < cap:
                    selected.append(ticker)
    for ticker in report["Ticker"]:
        if ticker not in selected and len(selected) < cap:
            selected.append(ticker)
    selected_set = set(selected)
    report["Selected"] = report["Ticker"].isin(selected_set)
    return selected, report
