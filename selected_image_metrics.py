"""Recalculate image-view evidence without a solver, downloads or payload mutation."""
import numpy as np
import pandas as pd

from allocation_image_metrics import finite_number, saved_image_metrics, image_metric_config
from portfolio_risk_metrics import (
    bootstrap_tail_summary, empirical_expected_shortfall,
    moving_block_bootstrap_return_distribution,
)


def _stats(series, annualization, risk_free, horizon):
    annual_return = float(series.mean() * annualization)
    volatility = float(series.std(ddof=1) * np.sqrt(annualization))
    scenarios = moving_block_bootstrap_return_distribution(series, horizon_sessions=horizon)
    tail = bootstrap_tail_summary(scenarios)
    return {
        "Annual Return": annual_return,
        "Annual Volatility": volatility,
        "Sharpe Ratio": (annual_return-risk_free)/volatility if volatility else None,
        "Historical ES 95% (1 Session)": empirical_expected_shortfall(np.expm1(series)),
        f"Block-Bootstrap ES 95% ({horizon} Sessions)": tail["expected_shortfall"],
        f"Block-Bootstrap Median Return ({horizon} Sessions)": tail["median_return"],
        f"Block-Bootstrap Tail Loss SD 95% ({horizon} Sessions)": tail["tail_loss_sd"],
    }


def selected_image_statistics(payload, selected_rows, history):
    """Compare the same selected listings, normalizing each mix independently.

    The image's allocation column retains its original base. Returns use dated
    INR-valued prices, including FX; missing open-day evidence is never filled.
    """
    config = image_metric_config(payload)
    horizon = saved_image_metrics(payload)[1] or 30
    # Keep the horizon explicit even when evidence is unavailable.
    empty = {f"Block-Bootstrap ES 95% ({horizon} Sessions)": None}
    result = {"current_stats": {}, "optimal_stats": empty,
              "portfolio_metrics_config": config}
    details = {"history_label": "Unavailable", "observations": 0, "error": None}
    try:
        annualization = finite_number(config.get("trading_days_per_year"), nonnegative=True)
        risk_free = finite_number(config.get("risk_free_rate_annual"))
        if not annualization or risk_free is None:
            raise ValueError("Recorded annualization or risk-free input unavailable")
        if not history or history.get("schema_version") != 2 or history.get("basis") != "INR":
            raise ValueError("Dated security-level INR prices and FX unavailable")
        columns = history.get("columns") or []
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate price-history listings")
        tickers = [r[0] for r in selected_rows]
        if set(tickers)-set(columns):
            raise ValueError("Missing selected security history: " + ", ".join(sorted(set(tickers)-set(columns))))
        prices = pd.DataFrame(history["values"], columns=columns,
                              index=pd.to_datetime(history["dates"])).loc[:, tickers].astype(float)
        if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
            raise ValueError("Price dates must be unique and increasing")
        valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        if not len(valid) or not valid.iloc[-1]:
            raise ValueError("Latest selected prices incomplete")
        if not valid.all():
            prices = prices.loc[prices.index > valid.index[~valid][-1]]
        returns = np.log(prices/prices.shift(1)).iloc[1:]
        if len(returns) < max(120, horizon):
            raise ValueError("Fewer than 120 contiguous common selected returns")
        target = np.array([r[2] for r in selected_rows], dtype=float)
        result["optimal_stats"] = _stats((returns @ (target/target.sum())).to_numpy(), annualization, risk_free, horizon)
        details.update(history_label=returns.index[-1].strftime("%d %b %Y"), observations=len(returns))
        current = dict.fromkeys(tickers, 0.)
        for row in payload.get("current_allocation") or []:
            ticker = str(row.get("Yahoo Ticker") or "").strip().upper()
            if ticker not in current:
                continue
            weight = finite_number(row.get("Weight"), nonnegative=True)
            if weight is None:
                raise ValueError("Current selected weights unavailable")
            current[ticker] += weight
        weights = np.array([current[t] for t in tickers])
        if weights.sum() > 0:
            result["current_stats"] = _stats((returns @ (weights/weights.sum())).to_numpy(), annualization, risk_free, horizon)
        else:
            details["error"] = "No current allocation in this selection; current statistics unavailable"
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        details["error"] = str(exc)
    metrics, _ = saved_image_metrics(result)
    return metrics, horizon, details
