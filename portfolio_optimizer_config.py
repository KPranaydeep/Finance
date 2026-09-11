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

    return {
        "config_version": version,
        "risk_free_rate_annual": float(risk_free),
        "trading_days_per_year": trading_days,
        "max_weight_per_asset": float(max_weight),
        "history_start_date": start.isoformat(),
    }


OPTIMIZER_CONFIG = load_optimizer_config()
OPTIMIZER_CONFIG_VERSION = OPTIMIZER_CONFIG["config_version"]
RISK_FREE_RATE_ANNUAL = OPTIMIZER_CONFIG["risk_free_rate_annual"]
TRADING_DAYS_PER_YEAR = OPTIMIZER_CONFIG["trading_days_per_year"]
MAX_WEIGHT_PER_ASSET = OPTIMIZER_CONFIG["max_weight_per_asset"]
DEFAULT_HISTORY_START_DATE = OPTIMIZER_CONFIG["history_start_date"]
