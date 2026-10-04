"""Fast, cluster-aware preselection before full-history portfolio optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd


RISK_APPETITE_PAIRS = (
    ("IWM", "SPY", "Small caps / large caps"),
    ("XLY", "XLP", "Cyclicals / defensives"),
    ("HYG", "LQD", "High yield / investment grade"),
)


def calculate_risk_appetite_regime(
    benchmark_closes: pd.DataFrame,
    horizons: tuple[int, ...] = (63, 126),
) -> dict:
    """Return a bounded regime tilt from three relative-strength pairs."""
    neutral = {
        "status": "unavailable",
        "regime": "neutral",
        "score": 0.0,
        "tilt": 0.0,
        "positive_pairs": 0,
        "available_pairs": 0,
        "components": [],
    }
    if benchmark_closes is None or benchmark_closes.empty:
        return neutral

    closes = benchmark_closes.copy()
    closes.columns = [str(column).strip().upper() for column in closes.columns]
    components = []
    for numerator, denominator, label in RISK_APPETITE_PAIRS:
        if numerator not in closes or denominator not in closes:
            continue
        aligned = pd.concat(
            [
                pd.to_numeric(closes[numerator], errors="coerce"),
                pd.to_numeric(closes[denominator], errors="coerce"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        aligned = aligned[(aligned > 0).all(axis=1)]
        if len(aligned) <= max(horizons):
            continue
        log_ratio = np.log(aligned.iloc[:, 0] / aligned.iloc[:, 1])
        daily_volatility = float(log_ratio.diff().dropna().tail(252).std(ddof=1))
        horizon_scores = []
        raw_changes = []
        for horizon in horizons:
            if len(log_ratio) <= int(horizon):
                continue
            change = float(log_ratio.iloc[-1] - log_ratio.iloc[-1 - int(horizon)])
            raw_changes.append(change)
            scale = daily_volatility * np.sqrt(float(horizon))
            if np.isfinite(scale) and scale > 1e-8:
                horizon_scores.append(change / scale)
        if not horizon_scores:
            continue
        direction = float(np.mean(raw_changes)) if raw_changes else 0.0
        components.append(
            {
                "pair": f"{numerator}/{denominator}",
                "label": label,
                "score": float(np.clip(np.mean(horizon_scores), -3.0, 3.0)),
                "direction": (
                    "positive" if direction > 0 else "negative" if direction < 0 else "flat"
                ),
            }
        )

    if len(components) < 2:
        return {**neutral, "components": components, "available_pairs": len(components)}
    composite = float(np.mean([item["score"] for item in components]))
    positive_pairs = sum(item["direction"] == "positive" for item in components)
    negative_pairs = sum(item["direction"] == "negative" for item in components)
    if positive_pairs >= 2 and composite > 0:
        regime = "risk-on"
        tilt = float(np.tanh(composite / 1.5))
    elif negative_pairs >= 2 and composite < 0:
        regime = "risk-off"
        tilt = float(np.tanh(composite / 1.5))
    else:
        regime = "neutral"
        tilt = 0.0
    return {
        "status": "available",
        "regime": regime,
        "score": composite,
        "tilt": float(np.clip(tilt, -1.0, 1.0)),
        "positive_pairs": int(positive_pairs),
        "available_pairs": int(len(components)),
        "components": components,
    }


def filter_candidates_by_market_cap(
    candidates: pd.DataFrame,
    exclusion_fraction: float = 0.20,
    minimum_known_per_cluster: int = 5,
    minimum_retained_per_cluster: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Remove at most the bottom market-cap fraction within listing clusters.

    Market-cap units and currencies need not be comparable across exchanges because
    ranks are calculated independently within ``Exchange · Currency`` clusters.
    Missing market caps are retained rather than guessed, and small clusters remain
    intact so the early runtime gate cannot erase a market entirely.
    """
    if candidates is None or candidates.empty:
        empty = pd.DataFrame() if candidates is None else candidates.copy()
        return empty, {
            "before": 0,
            "after": 0,
            "excluded": 0,
            "known": 0,
            "unknown_retained": 0,
            "fraction": min(max(float(exclusion_fraction), 0.0), 0.20),
            "clusters": [],
        }

    frame = candidates.copy()
    cap_column = "Market Cap Millions"
    fraction = min(max(float(exclusion_fraction), 0.0), 0.20)
    if cap_column not in frame.columns:
        return frame, {
            "before": int(len(frame)),
            "after": int(len(frame)),
            "excluded": 0,
            "known": 0,
            "unknown_retained": int(len(frame)),
            "fraction": fraction,
            "clusters": [],
        }

    frame[cap_column] = pd.to_numeric(frame[cap_column], errors="coerce")
    frame.loc[frame[cap_column] <= 0, cap_column] = np.nan
    exchange = frame.get("Exchange", pd.Series("Unknown", index=frame.index))
    currency = frame.get("Currency", pd.Series("Unknown", index=frame.index))
    frame["_market_cap_cluster"] = (
        exchange.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        + " · "
        + currency.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    )

    remove_indices: list[object] = []
    cluster_report: list[dict] = []
    ticker_column = "Yahoo Ticker" if "Yahoo Ticker" in frame.columns else "Symbol"
    for cluster, rows in frame.groupby("_market_cap_cluster", sort=True, dropna=False):
        known = rows.loc[rows[cap_column].notna()].copy()
        removable = max(len(known) - max(int(minimum_retained_per_cluster), 1), 0)
        remove_count = min(int(np.floor(len(known) * fraction)), removable)
        if len(known) < max(int(minimum_known_per_cluster), 1):
            remove_count = 0
        if remove_count:
            ordered = known.sort_values(
                [cap_column, ticker_column],
                ascending=[True, True],
                kind="mergesort",
            )
            remove_indices.extend(ordered.head(remove_count).index.tolist())
        cluster_report.append(
            {
                "cluster": str(cluster),
                "candidates": int(len(rows)),
                "known_market_caps": int(len(known)),
                "excluded": int(remove_count),
                "unknown_retained": int(rows[cap_column].isna().sum()),
            }
        )

    filtered = frame.drop(index=remove_indices).drop(columns="_market_cap_cluster")
    filtered = filtered.reset_index(drop=True)
    return filtered, {
        "before": int(len(frame)),
        "after": int(len(filtered)),
        "excluded": int(len(remove_indices)),
        "known": int(frame[cap_column].notna().sum()),
        "unknown_retained": int(frame[cap_column].isna().sum()),
        "fraction": fraction,
        "clusters": cluster_report,
    }


def convert_candidate_history_to_inr(
    close_history: pd.DataFrame,
    ticker_currencies: dict[str, str],
    fx_history_to_inr: dict[str, pd.Series],
) -> tuple[pd.DataFrame, list[str]]:
    """Convert recent candidate histories to INR, omitting unconvertible currencies."""
    if close_history is None or close_history.empty:
        return pd.DataFrame(), []

    converted = pd.DataFrame(index=close_history.index)
    omitted_currencies: set[str] = set()
    for ticker in close_history.columns:
        currency = str(ticker_currencies.get(str(ticker).upper(), "INR")).strip().upper()
        prices = pd.to_numeric(close_history[ticker], errors="coerce")
        if currency == "INR":
            converted[ticker] = prices
            continue
        fx = fx_history_to_inr.get(currency)
        if fx is None or fx.empty:
            omitted_currencies.add(currency)
            continue
        aligned_fx = pd.to_numeric(fx, errors="coerce").reindex(close_history.index).ffill().bfill()
        if aligned_fx.isna().all():
            omitted_currencies.add(currency)
            continue
        converted[ticker] = prices * aligned_fx

    return converted, sorted(omitted_currencies)


def rank_scalable_candidates(
    close_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    ticker_clusters: dict[str, str],
    maximum_candidates: int = 400,
    minimum_per_cluster: int = 5,
    risk_appetite_regime: dict | None = None,
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
        trailing_returns = {}
        for horizon in (21, 63, 126, 252):
            trailing_returns[horizon] = (
                float(prices.iloc[-1] / prices.iloc[-1 - horizon] - 1.0)
                if len(prices) > horizon and prices.iloc[-1 - horizon] > 0
                else np.nan
            )
        daily = np.log(prices / prices.shift(1)).dropna().tail(252)
        annual_vol = float(daily.std(ddof=1) * np.sqrt(250)) if len(daily) > 1 else np.nan
        risk_scale = max(annual_vol, 0.05) if np.isfinite(annual_vol) else 0.50
        recent_volume = volume[ticker].dropna().tail(63) if ticker in volume else pd.Series(dtype=float)
        turnover = (
            float((close[ticker].reindex(recent_volume.index) * recent_volume).median())
            if not recent_volume.empty
            else 0.0
        )
        prior_volume = (
            volume[ticker].dropna().iloc[-84:-21]
            if ticker in volume and len(volume[ticker].dropna()) >= 84
            else pd.Series(dtype=float)
        )
        latest_volume = (
            volume[ticker].dropna().tail(21)
            if ticker in volume
            else pd.Series(dtype=float)
        )
        prior_median = float(prior_volume.median()) if not prior_volume.empty else 0.0
        volume_surge = (
            float(latest_volume.median() / prior_median - 1.0)
            if prior_median > 0 and not latest_volume.empty
            else 0.0
        )
        trailing_high = float(prices.tail(252).max())
        breakout_proximity = float(prices.iloc[-1] / trailing_high) if trailing_high > 0 else 0.0
        acceleration = float(
            np.nan_to_num(trailing_returns[21], nan=0.0)
            - np.nan_to_num(trailing_returns[63], nan=0.0) / 3.0
        )
        rows.append(
            {
                "Ticker": str(ticker).upper(),
                "Cluster": ticker_clusters.get(str(ticker).upper(), "Unknown"),
                "Momentum": float(np.median(horizon_returns) / risk_scale),
                "Turnover": max(turnover, 0.0),
                "Continuity": float(prices.count() / max(len(close.index), 1)),
                "Latest Price": float(prices.iloc[-1]),
                "21-session return": trailing_returns[21],
                "63-session return": trailing_returns[63],
                "Acceleration": acceleration,
                "Breakout proximity": breakout_proximity,
                "Volume surge": volume_surge,
            }
        )
    report = pd.DataFrame(rows)
    if report.empty:
        return [], report
    grouped = report.groupby("Cluster", dropna=False)
    report["Momentum rank"] = grouped["Momentum"].rank(pct=True, method="average")
    report["Liquidity rank"] = grouped["Turnover"].rank(pct=True, method="average")
    report["Continuity rank"] = grouped["Continuity"].rank(pct=True, method="average")
    report["Acceleration rank"] = grouped["Acceleration"].rank(pct=True, method="average")
    report["Breakout rank"] = grouped["Breakout proximity"].rank(pct=True, method="average")
    report["Volume-surge rank"] = grouped["Volume surge"].rank(pct=True, method="average")
    regime = dict(risk_appetite_regime or {})
    tilt = float(np.clip(regime.get("tilt", 0.0), -1.0, 1.0))
    report["Preselection score"] = (
        (0.60 + 0.10 * tilt) * report["Momentum rank"]
        + (0.30 - 0.05 * tilt) * report["Liquidity rank"]
        + (0.10 - 0.05 * tilt) * report["Continuity rank"]
    )
    report["Emerging-winner score"] = (
        (0.40 + 0.05 * tilt) * report["Acceleration rank"]
        + (0.25 + 0.025 * tilt) * report["Breakout rank"]
        + 0.20 * report["Volume-surge rank"]
        + (0.15 - 0.075 * tilt) * report["Liquidity rank"]
    )
    report["Risk-appetite regime"] = str(regime.get("regime", "neutral"))
    report["Risk-appetite tilt"] = tilt
    report["Balanced score"] = np.maximum(
        report["Preselection score"], report["Emerging-winner score"]
    )
    report = report.sort_values(
        ["Preselection score", "Momentum", "Turnover", "Ticker"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(report) <= cap:
        report["Selected"] = True
        report["Selection sleeve"] = "All eligible"
        return report["Ticker"].tolist(), report

    selected: list[str] = []
    sleeves: dict[str, str] = {}
    if floor:
        diversified = report.sort_values(
            ["Balanced score", "Ticker"], ascending=[False, True], kind="mergesort"
        )
        for _, cluster_rows in diversified.groupby("Cluster", sort=True, dropna=False):
            for ticker in cluster_rows.head(floor)["Ticker"]:
                if ticker not in selected and len(selected) < cap:
                    selected.append(ticker)
                    sleeves[ticker] = "Cluster reserve"

    remaining_capacity = max(cap - len(selected), 0)
    core_slots = int(round(remaining_capacity * (0.70 - 0.025 * tilt)))
    emerging_slots = int(round(remaining_capacity * (0.20 + 0.05 * tilt)))

    core_order = report.sort_values(
        ["Preselection score", "Momentum", "Ticker"],
        ascending=[False, False, True], kind="mergesort",
    )["Ticker"]
    core_added = 0
    for ticker in core_order:
        if ticker not in selected and len(selected) < cap and core_added < core_slots:
            selected.append(ticker)
            sleeves[ticker] = "Stable momentum"
            core_added += 1

    emerging_order = report.sort_values(
        ["Emerging-winner score", "Acceleration", "Ticker"],
        ascending=[False, False, True], kind="mergesort",
    )["Ticker"]
    emerging_added = 0
    for ticker in emerging_order:
        if ticker not in selected and len(selected) < cap and emerging_added < emerging_slots:
            selected.append(ticker)
            sleeves[ticker] = "Emerging winner"
            emerging_added += 1

    balanced_order = report.sort_values(
        ["Balanced score", "Turnover", "Ticker"],
        ascending=[False, False, True], kind="mergesort",
    )["Ticker"]
    for ticker in balanced_order:
        if ticker not in selected and len(selected) < cap:
            selected.append(ticker)
            sleeves[ticker] = "Balanced reserve"
    selected_set = set(selected)
    report["Selected"] = report["Ticker"].isin(selected_set)
    report["Selection sleeve"] = report["Ticker"].map(sleeves).fillna("Not selected")
    return selected, report
