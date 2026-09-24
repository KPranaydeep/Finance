"""Validated, versioned defaults shared by interactive and scheduled optimizers."""
from __future__ import annotations

import json
import math
import os
import re
from datetime import date
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("portfolio_optimizer_config.json")
REQUIRED_KEYS = {
    "config_version",
    "risk_free_rate_annual",
    "trading_days_per_year",
    "max_weight_per_asset",
    "history_start_date",
    "momentum_filter",
}

MOMENTUM_FILTER_KEYS = {
    "enabled",
    "method_version",
    "maximum_exclusion_fraction",
    "lookback_sessions",
    "skip_recent_sessions",
    "stability_checkpoint_sessions",
    "trend_lookback_sessions",
    "minimum_negative_horizons",
    "annualization_sessions",
    "volatility_floor_annual",
    "apply_exclusion_to_owned_holdings",
}


def load_optimizer_config(path=None):
    configured = path or os.getenv("PORTFOLIO_OPTIMIZER_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    config_path = Path(configured).expanduser()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("OPTIMIZER_CONFIG_UNAVAILABLE") from exc
    if not isinstance(payload, dict) or set(payload) != REQUIRED_KEYS:
        raise ValueError("OPTIMIZER_CONFIG_KEYS_INVALID")

    version = payload["config_version"]
    if not isinstance(version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", version):
        raise ValueError("OPTIMIZER_CONFIG_VERSION_INVALID")

    risk_free = payload["risk_free_rate_annual"]
    max_weight = payload["max_weight_per_asset"]
    trading_days = payload["trading_days_per_year"]
    if (isinstance(risk_free, bool) or not isinstance(risk_free, (int, float)) or
            not math.isfinite(risk_free) or not 0 <= risk_free < 1):
        raise ValueError("OPTIMIZER_RISK_FREE_RATE_INVALID")
    if (isinstance(max_weight, bool) or not isinstance(max_weight, (int, float)) or
            not math.isfinite(max_weight) or not 0 < max_weight <= 1):
        raise ValueError("OPTIMIZER_MAX_WEIGHT_INVALID")
    if isinstance(trading_days, bool) or not isinstance(trading_days, int) or not 200 <= trading_days <= 366:
        raise ValueError("OPTIMIZER_TRADING_DAYS_INVALID")
    try:
        start = date.fromisoformat(payload["history_start_date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("OPTIMIZER_HISTORY_START_INVALID") from exc
    if start >= date.today():
        raise ValueError("OPTIMIZER_HISTORY_START_INVALID")

    momentum = payload["momentum_filter"]
    if not isinstance(momentum, dict) or set(momentum) != MOMENTUM_FILTER_KEYS:
        raise ValueError("OPTIMIZER_MOMENTUM_FILTER_KEYS_INVALID")
    if not isinstance(momentum["enabled"], bool):
        raise ValueError("OPTIMIZER_MOMENTUM_ENABLED_INVALID")
    method_version = momentum["method_version"]
    if not isinstance(method_version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", method_version):
        raise ValueError("OPTIMIZER_MOMENTUM_VERSION_INVALID")
    maximum_exclusion = momentum["maximum_exclusion_fraction"]
    if (isinstance(maximum_exclusion, bool) or not isinstance(maximum_exclusion, (int, float)) or
            not math.isfinite(maximum_exclusion) or not 0 <= maximum_exclusion <= 0.20):
        raise ValueError("OPTIMIZER_MOMENTUM_MAXIMUM_EXCLUSION_INVALID")
    lookbacks = momentum["lookback_sessions"]
    if (not isinstance(lookbacks, list) or len(lookbacks) != 3 or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 40 for value in lookbacks) or
            lookbacks != sorted(set(lookbacks))):
        raise ValueError("OPTIMIZER_MOMENTUM_LOOKBACKS_INVALID")
    skip_recent = momentum["skip_recent_sessions"]
    if isinstance(skip_recent, bool) or not isinstance(skip_recent, int) or not 0 < skip_recent < min(lookbacks):
        raise ValueError("OPTIMIZER_MOMENTUM_SKIP_INVALID")
    checkpoints = momentum["stability_checkpoint_sessions"]
    if (not isinstance(checkpoints, list) or len(checkpoints) < 1 or checkpoints[0] != 0 or
            any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in checkpoints) or
            checkpoints != sorted(set(checkpoints))):
        raise ValueError("OPTIMIZER_MOMENTUM_CHECKPOINTS_INVALID")
    trend_lookback = momentum["trend_lookback_sessions"]
    if isinstance(trend_lookback, bool) or not isinstance(trend_lookback, int) or trend_lookback < 100:
        raise ValueError("OPTIMIZER_MOMENTUM_TREND_INVALID")
    minimum_negative = momentum["minimum_negative_horizons"]
    if isinstance(minimum_negative, bool) or not isinstance(minimum_negative, int) or not 1 <= minimum_negative <= len(lookbacks):
        raise ValueError("OPTIMIZER_MOMENTUM_NEGATIVE_HORIZONS_INVALID")
    momentum_annualization = momentum["annualization_sessions"]
    if (isinstance(momentum_annualization, bool) or not isinstance(momentum_annualization, int) or
            not 200 <= momentum_annualization <= 366):
        raise ValueError("OPTIMIZER_MOMENTUM_ANNUALIZATION_INVALID")
    volatility_floor = momentum["volatility_floor_annual"]
    if (isinstance(volatility_floor, bool) or not isinstance(volatility_floor, (int, float)) or
            not math.isfinite(volatility_floor) or not 0 < volatility_floor < 1):
        raise ValueError("OPTIMIZER_MOMENTUM_VOLATILITY_FLOOR_INVALID")
    if momentum["apply_exclusion_to_owned_holdings"] is not False:
        raise ValueError("OPTIMIZER_MOMENTUM_OWNED_PROTECTION_REQUIRED")

    return {
        "config_version": version,
        "risk_free_rate_annual": float(risk_free),
        "trading_days_per_year": trading_days,
        "max_weight_per_asset": float(max_weight),
        "history_start_date": start.isoformat(),
        "momentum_filter": {
            "enabled": momentum["enabled"],
            "method_version": method_version,
            "maximum_exclusion_fraction": float(maximum_exclusion),
            "lookback_sessions": list(lookbacks),
            "skip_recent_sessions": skip_recent,
            "stability_checkpoint_sessions": list(checkpoints),
            "trend_lookback_sessions": trend_lookback,
            "minimum_negative_horizons": minimum_negative,
            "annualization_sessions": momentum_annualization,
            "volatility_floor_annual": float(volatility_floor),
            "apply_exclusion_to_owned_holdings": False,
        },
    }


OPTIMIZER_CONFIG = load_optimizer_config()
OPTIMIZER_CONFIG_VERSION = OPTIMIZER_CONFIG["config_version"]
RISK_FREE_RATE_ANNUAL = OPTIMIZER_CONFIG["risk_free_rate_annual"]
TRADING_DAYS_PER_YEAR = OPTIMIZER_CONFIG["trading_days_per_year"]
MAX_WEIGHT_PER_ASSET = OPTIMIZER_CONFIG["max_weight_per_asset"]
DEFAULT_HISTORY_START_DATE = OPTIMIZER_CONFIG["history_start_date"]
MOMENTUM_FILTER_CONFIG = OPTIMIZER_CONFIG["momentum_filter"]
