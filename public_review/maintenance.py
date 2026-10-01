"""Operational maintenance checks for the owner-approved review policy."""

from __future__ import annotations

from datetime import date, timedelta


def _state(days_remaining: int, warning_days: int) -> str:
    if days_remaining < 0:
        return "expired"
    if days_remaining <= warning_days:
        return "review_due"
    return "current"


def policy_maintenance_status(policy: dict, *, today: date | None = None,
                              warning_days: int = 30) -> dict:
    """Return deterministic tariff/calendar maintenance horizons.

    ``valid_through`` follows the same inclusive semantics as ``load_policy``:
    a tariff verified on day D with maximum age N remains valid through D + N.
    """
    if isinstance(warning_days, bool) or not isinstance(warning_days, int) or warning_days < 0:
        raise ValueError("INVALID_MAINTENANCE_WARNING_DAYS")

    today = today or date.today()
    tariff_verified = date.fromisoformat(policy["tariff_verified_on"])
    tariff_valid_through = tariff_verified + timedelta(days=int(policy["tariff_max_age_days"]))
    calendar_valid_through = date.fromisoformat(policy["calendar_verified_through"])

    tariff_days = (tariff_valid_through - today).days
    calendar_days = (calendar_valid_through - today).days
    tariff_state = _state(tariff_days, warning_days)
    calendar_state = _state(calendar_days, warning_days)
    states = (tariff_state, calendar_state)
    overall = "expired" if "expired" in states else (
        "attention" if "review_due" in states else "current"
    )

    return {
        "status": overall,
        "as_of": today.isoformat(),
        "warning_days": warning_days,
        "tariff": {
            "verified_on": tariff_verified.isoformat(),
            "valid_through": tariff_valid_through.isoformat(),
            "days_remaining": tariff_days,
            "state": tariff_state,
        },
        "calendar": {
            "valid_through": calendar_valid_through.isoformat(),
            "days_remaining": calendar_days,
            "state": calendar_state,
        },
    }
