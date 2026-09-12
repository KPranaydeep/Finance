import json
import math
import os
from datetime import date
from pathlib import Path
from .costs import KINDS


def load_policy(today=None):
    path = Path(os.environ.get("PUBLIC_REVIEW_POLICY_PATH",
                               str(Path(__file__).resolve().parents[1] / "public_review_policy.json")))
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("policy_approved") is not True:
        raise ValueError("POLICY_APPROVAL_REQUIRED")
    if policy.get("tax_profile") != "resident_individual_regular_delivery":
        raise ValueError("UNSUPPORTED_TAX_OR_ACCOUNT_PROFILE")
    today = today or date.today()
    age = (today - date.fromisoformat(policy["tariff_verified_on"])).days
    if age < 0 or age > policy["tariff_max_age_days"]:
        raise ValueError("TARIFF_REVIEW_REQUIRED")
    if today > date.fromisoformat(policy["calendar_verified_through"]):
        raise ValueError("CALENDAR_REVIEW_REQUIRED")
    ranges = {"slab_rate": (0, .30), "surcharge_rate": (0, .37), "slippage_bps": (0, 500),
              "target_xirr": (.01, 10), "drawdown_limit": (.01, .9), "concentration_limit": (.01, 1),
              "drift_limit": (.001, 1), "min_annual_improvement": (.06, 1),
              "crossing_probability": (.01, .5),
              "minimum_forecast_review_sessions": (1, 60), "max_review_sessions": (1, 60),
              "simulation_paths": (100, 5000), "block_length": (1, 30),
              "validation_train": (126, 756), "validation_horizon": (5, 60),
              "validation_min_folds": (20, 100), "validation_max_brier": (.01, .25),
              "history_years": (1, 10), "tariff_max_age_days": (1, 180),
              "entry_wait_after_open_minutes": (0, 240),
              "entry_max_quote_delay_minutes": (1, 720),
              "fx_quote_max_age_minutes": (1, 1440),
              "assessment_wait_after_close_minutes": (0, 180)}
    for key, (low, high) in ranges.items():
        value = policy[key]
        if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("INVALID_POLICY_" + key.upper())
    for key in ("minimum_forecast_review_sessions", "max_review_sessions", "simulation_paths", "block_length", "seed", "validation_train",
                "validation_horizon", "validation_min_folds", "history_years", "tariff_max_age_days",
                "entry_wait_after_open_minutes", "entry_max_quote_delay_minutes",
                "fx_quote_max_age_minutes",
                "assessment_wait_after_close_minutes"):
        if not isinstance(policy[key], int):
            raise ValueError("INTEGER_POLICY_REQUIRED")
    if policy["minimum_forecast_review_sessions"] > min(
            policy["max_review_sessions"], policy["validation_horizon"]):
        raise ValueError("INVALID_POLICY_MINIMUM_FORECAST_REVIEW_SESSIONS")
    if policy.get("entry_quote_interval") != "1m":
        raise ValueError("INVALID_POLICY_ENTRY_QUOTE_INTERVAL")
    if policy.get("profit_review_rule") != "ROUND_TRIP_BREAK_EVEN_AND_TARGET_XIRR":
        raise ValueError("INVALID_POLICY_PROFIT_REVIEW_RULE")
    # ``instrument_kinds`` is a resolved runtime field, not owner policy.
    # Accept it temporarily for backward-compatible tests/old deployments, but
    # production policy files no longer need or maintain a ticker registry.
    if any(not ((t.endswith(".NS") and k in KINDS) or
                (not t.endswith(".NS") and k == "foreign_us_listing"))
           for t, k in policy.get("instrument_kinds", {}).items()):
        raise ValueError("NSE_CLASSIFICATION_REQUIRED")
    if policy["capital_inr"] is not None and (not math.isfinite(policy["capital_inr"]) or policy["capital_inr"] < 1):
        raise ValueError("INVALID_CAPITAL")
    return policy
