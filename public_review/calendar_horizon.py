"""Bound forecasting and its validation by dates already verified by exchanges."""

from datetime import date


def verified_forecast_policy(policy, future):
    """Return a temporary forecast policy and explicit calendar coverage.

    The approved maximum stays unchanged. Five sessions is the validation
    engine's supported minimum; the owner's observation minimum may raise it.
    No date is invented or counted twice to fill the configured maximum.
    """
    future = list(future)
    requested = int(policy["max_review_sessions"])
    minimum = max(5, int(policy.get("minimum_forecast_review_sessions", 1)))
    try:
        verified = date.fromisoformat(policy["calendar_verified_through"])
        dates = [date.fromisoformat(day) for day in future]
        valid = dates == sorted(set(dates)) and all(day <= verified for day in dates)
    except (TypeError, ValueError):
        valid = False
    effective = min(requested, len(future)) if valid else 0
    usable = effective >= minimum
    coverage = {
        "required_forecast_sessions": requested,
        "available_forecast_sessions": len(future),
        "effective_forecast_sessions": effective if usable else 0,
        "minimum_forecast_sessions": minimum,
        "verified_through": policy["calendar_verified_through"],
        "complete": valid and len(future) >= requested,
        "forecast_usable": usable,
        "forecast_through": future[effective - 1] if usable else None,
    }
    if not usable:
        return None, coverage
    effective_policy = dict(policy)
    effective_policy["max_review_sessions"] = effective
    effective_policy["validation_horizon"] = min(int(policy["validation_horizon"]), effective)
    return effective_policy, coverage


def calendar_coverage_message(coverage):
    """Keep reduced-horizon and unavailable forecasts visibly distinct."""
    if not coverage or coverage.get("complete") is not False:
        return None
    if coverage.get("forecast_usable"):
        return (
            f"Review forecast uses {coverage['effective_forecast_sessions']} verified sessions "
            f"through {coverage['forecast_through']} (configured maximum "
            f"{coverage['required_forecast_sessions']}). Validation is capped to this window. "
            "The forecast will extend when more exchange dates are verified."
        )
    return (
        "Observed performance and risk checks are current; the review-date forecast is unavailable. "
        f"{coverage['available_forecast_sessions']} of {coverage['required_forecast_sessions']} future sessions "
        f"are verified through {coverage['verified_through']}. "
        "Reverify the calendar to restore the full forecast horizon."
    )
