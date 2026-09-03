from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGY_VERSION = "portfolio-rebalancer-v1"
INPUT_PATH_ENV = "PUBLIC_BASKET_INPUT_PATH"

DEFAULT_SETTINGS = {
    "days_to_flip": 10,
    "max_drawdown_input_pct": 20.0,
    "drop_bottom_fraction": 0.20,
    "history_buffer_days": 30,
    "redundancy_corr_threshold": 0.95,
    "use_target_volatility": False,
    "target_volatility": None,
}

REQUIRED_PORTFOLIO_COLUMNS = {
    "Symbol",
    "Yahoo Ticker",
    "Currency",
    "Quantity",
    "Average Price",
    "FX to INR",
    "Weight",
}


def _load_core_functions():
    """
    Import only the UI-free optimizer module.

    Never import portfolio_rebalancer_database.py here: that file contains
    top-level Streamlit page code and is unsafe inside a scheduled worker.
    """
    try:
        from portfolio_optimizer_core import (
            get_latest_price_map,
            rebalance_plan_multi,
            run_portfolio_analysis_multi,
        )
    except ImportError as exc:
        raise RuntimeError(
            "portfolio_optimizer_core.py is required. Move the existing "
            "run_portfolio_analysis_multi(), rebalance_plan_multi(), and "
            "get_latest_price_map() implementations into that UI-free module "
            "without changing their mathematics."
        ) from exc

    return (
        run_portfolio_analysis_multi,
        rebalance_plan_multi,
        get_latest_price_map,
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.to_dict(orient="records")]

    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return str(value)


def _read_frozen_input() -> dict[str, Any]:
    """
    Read an operator-prepared input snapshot.

    This is not an end-user upload. The scheduled deployment points to a controlled
    JSON file using PUBLIC_BASKET_INPUT_PATH. The complete input is copied into the
    immutable signal record by the scheduler.
    """
    configured_path = os.getenv(INPUT_PATH_ENV)
    if not configured_path:
        raise RuntimeError(
            f"{INPUT_PATH_ENV} is not configured. It must point to the controlled "
            "public-basket input JSON file."
        )

    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Public-basket input file was not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid public-basket input JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("The public-basket input must be a JSON object")

    return payload


def _validate_and_build_portfolio(payload: dict[str, Any]) -> pd.DataFrame:
    portfolio = payload.get("portfolio")
    if not isinstance(portfolio, list) or not portfolio:
        raise ValueError("Input field 'portfolio' must be a non-empty list")

    frame = pd.DataFrame(portfolio)
    missing = sorted(REQUIRED_PORTFOLIO_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError("Portfolio is missing columns: " + ", ".join(missing))

    frame["Yahoo Ticker"] = frame["Yahoo Ticker"].astype(str).str.strip()
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip()
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["Average Price"] = pd.to_numeric(frame["Average Price"], errors="coerce")
    frame["FX to INR"] = pd.to_numeric(frame["FX to INR"], errors="coerce")
    frame["Weight"] = pd.to_numeric(frame["Weight"], errors="coerce")

    invalid = frame[
        frame["Yahoo Ticker"].eq("")
        | frame["Symbol"].eq("")
        | frame["Quantity"].isna()
        | frame["Average Price"].isna()
        | frame["FX to INR"].isna()
        | frame["Weight"].isna()
        | frame["Quantity"].lt(0)
        | frame["Average Price"].le(0)
        | frame["FX to INR"].le(0)
        | frame["Weight"].lt(0)
    ]
    if not invalid.empty:
        raise ValueError(
            "Portfolio contains invalid symbol, ticker, quantity, price, FX, or weight rows: "
            + ", ".join(invalid.index.astype(str))
        )

    if frame["Yahoo Ticker"].duplicated().any():
        duplicates = sorted(
            frame.loc[frame["Yahoo Ticker"].duplicated(False), "Yahoo Ticker"].unique()
        )
        raise ValueError("Duplicate Yahoo tickers: " + ", ".join(duplicates))

    if float(frame["Weight"].sum()) <= 0:
        raise ValueError("Portfolio weights must have a positive total")

    return frame.reset_index(drop=True)


def _validated_settings(payload: dict[str, Any]) -> dict[str, Any]:
    supplied = payload.get("settings", {})
    if not isinstance(supplied, dict):
        raise ValueError("Input field 'settings' must be an object")

    unknown = sorted(set(supplied).difference(DEFAULT_SETTINGS))
    if unknown:
        raise ValueError("Unknown optimizer settings: " + ", ".join(unknown))

    settings = {**DEFAULT_SETTINGS, **supplied}
    settings["days_to_flip"] = int(settings["days_to_flip"])
    settings["max_drawdown_input_pct"] = float(
        settings["max_drawdown_input_pct"]
    )
    settings["drop_bottom_fraction"] = float(settings["drop_bottom_fraction"])
    settings["history_buffer_days"] = int(settings["history_buffer_days"])
    settings["redundancy_corr_threshold"] = float(
        settings["redundancy_corr_threshold"]
    )
    settings["use_target_volatility"] = bool(settings["use_target_volatility"])

    target = settings["target_volatility"]
    settings["target_volatility"] = float(target) if target is not None else None

    if not 0 <= settings["drop_bottom_fraction"] < 1:
        raise ValueError("drop_bottom_fraction must be at least 0 and below 1")
    if not 0 < settings["redundancy_corr_threshold"] <= 1:
        raise ValueError("redundancy_corr_threshold must be above 0 and at most 1")
    if settings["days_to_flip"] <= 0 or settings["history_buffer_days"] < 0:
        raise ValueError("days_to_flip must be positive and history_buffer_days cannot be negative")
    if settings["use_target_volatility"] and settings["target_volatility"] is None:
        raise ValueError("target_volatility is required when enabled")

    # Preserve the existing application's sign convention.
    settings["internal_max_dd"] = -abs(
        settings["max_drawdown_input_pct"] / 100.0
    )
    return settings


def _validate_effective_date(
    payload: dict[str, Any], scheduled_session_date: date
) -> None:
    declared = payload.get("scheduled_session_date")
    if declared is None:
        raise ValueError(
            "Input must declare scheduled_session_date so stale inputs cannot be reused"
        )

    declared_date = date.fromisoformat(str(declared))
    if declared_date != scheduled_session_date:
        raise ValueError(
            "Input scheduled_session_date does not match the scheduler gate: "
            f"{declared_date} != {scheduled_session_date}"
        )


def build_public_signal(*, scheduled_session_date: date) -> dict[str, Any]:
    """Run the unchanged optimizer and return the scheduler's immutable contract."""
    payload = _read_frozen_input()
    _validate_effective_date(payload, scheduled_session_date)
    portfolio_df = _validate_and_build_portfolio(payload)
    settings = _validated_settings(payload)

    (
        run_portfolio_analysis_multi,
        rebalance_plan_multi,
        get_latest_price_map,
    ) = _load_core_functions()

    tickers = portfolio_df["Yahoo Ticker"].tolist()
    target_volatility = (
        settings["target_volatility"]
        if settings["use_target_volatility"]
        else None
    )

    optimal_weights, log_returns, current_stats, optimal_stats, meta = (
        run_portfolio_analysis_multi(
            tickers,
            portfolio_df.copy(),
            max_dd=settings["internal_max_dd"],
            target_volatility=target_volatility,
            drop_bottom_pct=settings["drop_bottom_fraction"],
            buffer_days=settings["history_buffer_days"],
            redundancy_corr_threshold=settings["redundancy_corr_threshold"],
        )
    )

    if optimal_weights is None or log_returns is None or log_returns.empty:
        raise RuntimeError("Optimizer did not return a complete allocation and history")

    optimized_tickers = list(log_returns.columns)
    if len(optimized_tickers) != len(optimal_weights):
        raise RuntimeError("Optimizer returned different ticker and weight counts")

    price_map = get_latest_price_map(optimized_tickers)
    missing_price_tickers = sorted(
        ticker
        for ticker in optimized_tickers
        if ticker not in price_map
        or price_map[ticker] is None
        or not math.isfinite(float(price_map[ticker]))
        or float(price_map[ticker]) <= 0
    )
    if missing_price_tickers:
        raise RuntimeError(
            "Missing valid scheduled-session prices: "
            + ", ".join(missing_price_tickers)
        )

    rebalance_df, missing_prices, missing_alloc = rebalance_plan_multi(
        portfolio_df.copy(),
        optimal_weights,
        log_returns,
        price_map,
        settings["days_to_flip"],
    )
    if missing_prices or missing_alloc:
        raise RuntimeError(
            "Rebalance output is incomplete. Missing prices: "
            + ", ".join(sorted(missing_prices))
            + "; missing allocations: "
            + ", ".join(sorted(missing_alloc))
        )

    allocation = [
        {"yahoo_ticker": ticker, "target_weight": float(weight)}
        for ticker, weight in zip(optimized_tickers, optimal_weights)
    ]
    signal_rows = _json_safe(rebalance_df)
    decision_status = "NO_CHANGE" if rebalance_df.empty else "REBALANCED"

    frozen_settings = {
        **settings,
        "scheduled_session_date": scheduled_session_date.isoformat(),
        "market_data_cutoff": payload.get("market_data_cutoff"),
        "input_created_at": payload.get("input_created_at"),
        "adapter_generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return _json_safe(
        {
            "strategy_version": STRATEGY_VERSION,
            "settings": frozen_settings,
            "portfolio_before": portfolio_df,
            "optimizer_output": {
                "target_allocation": allocation,
                "current_stats": current_stats or {},
                "optimal_stats": optimal_stats or {},
                "history_start": str(meta.get("valid_start")),
                "history_end": str(meta.get("valid_end")),
                "history_rows": int(log_returns.shape[0]),
                "history_assets": int(log_returns.shape[1]),
                "dropped_tickers": meta.get("dropped_df", pd.DataFrame()),
                "merged_redundant_assets": meta.get(
                    "redundant_df", pd.DataFrame()
                ),
                "latest_prices": price_map,
            },
            "signal_output": signal_rows,
            "decision_status": decision_status,
        }
    )
