"""Small, durable decision helpers for reusing a recent X/Y search leader."""

from __future__ import annotations

from datetime import datetime, timezone


def skipper_decision(
    context,
    *,
    now_utc=None,
    fallback_discovery_at=None,
    maximum_age_hours=24,
):
    """Return a validated search context while its full scan is under 24h old."""
    if not isinstance(context, dict):
        return None
    try:
        discovery_value = (
            context.get("last_discovery_at_utc") or fallback_discovery_at
        )
        if not discovery_value:
            return None
        discovered = _as_datetime_utc(discovery_value)
        now = _as_datetime_utc(now_utc or datetime.now(timezone.utc))
        age_hours = (now - discovered).total_seconds() / 3600.0
        x = int(context["best_x"])
        y = int(context["best_y"])
        previous_return = float(context["last_checked_return"])
        if (
            x <= 0
            or y <= 0
            or not (previous_return == previous_return)
            or age_hours < 0
        ):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if age_hours >= max(float(maximum_age_hours), 0.0):
        return None
    return {**context, "best_x": x, "best_y": y,
            "last_checked_return": previous_return,
            "last_discovery_at_utc": discovered.isoformat(),
            "age_hours": age_hours}


def _as_datetime_utc(value) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
