"""Explain the next timing update using the deployed workflow's actual schedule."""

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from review_schedule_guard import should_run


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/public_portfolio_review.yml"


def next_scheduled_update(after, workflow_path=WORKFLOW):
    """Read supported UTC cron entries directly; never invent a refresh time."""
    if after.tzinfo is None:
        raise ValueError("AWARE_TIME_REQUIRED")
    schedules = re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', Path(workflow_path).read_text(encoding="utf-8"), re.M)
    if not schedules:
        raise ValueError("REVIEW_SCHEDULE_UNAVAILABLE")
    candidates = []
    day = after.astimezone(timezone.utc).date()
    for schedule in schedules:
        fields = schedule.split()
        if len(fields) != 5 or fields[2:4] != ["*", "*"]:
            raise ValueError("UNSUPPORTED_REVIEW_SCHEDULE")
        minute, hour = int(fields[0]), int(fields[1])
        if not 0 <= minute < 60 or not 0 <= hour < 24:
            raise ValueError("INVALID_REVIEW_SCHEDULE")
        days = set()
        for part in fields[4].split(","):
            if part == "*":
                days.update(range(7))
            elif "-" in part:
                first, last = map(int, part.split("-"))
                days.update(range(first, last + 1))
            else:
                days.add(int(part))
        if not days or not days <= set(range(7)):
            raise ValueError("INVALID_REVIEW_SCHEDULE")
        for offset in range(8):
            candidate = datetime.combine(day + timedelta(days=offset), time(hour, minute), timezone.utc)
            if (candidate >= after and (candidate.weekday() + 1) % 7 in days
                    and should_run(schedule, candidate)):
                candidates.append(candidate)
    if not candidates:
        raise ValueError("REVIEW_SCHEDULE_UNAVAILABLE")
    return min(candidates)


def timing_check_message(payload, now):
    if not payload.get("forecast_observation_pending") or not payload.get("observation_ready_at"):
        return None
    try:
        ready = datetime.fromisoformat(payload["observation_ready_at"])
        if ready.tzinfo is None or now.tzinfo is None:
            return None
        local_ready = ready.astimezone(ZoneInfo("Asia/Kolkata"))
        count = int((payload.get("policy") or {}).get("minimum_forecast_review_sessions", 1))
        update = next_scheduled_update(max(ready, now)).astimezone(ZoneInfo("Asia/Kolkata"))
        reason = (
            f"The required {count} post-entry sessions complete {local_ready:%d %b, %H:%M IST}. "
            if ready > now else
            f"The {count}-session observation window has elapsed; the saved timing assessment needs updating. "
        )
        decision = payload.get("decision", {})
        active = decision.get("review_required", bool(decision.get("reasons")))
        ending = ("The active review flag still applies now." if active else
                  "Validation is still required before a review date is confirmed.")
        return (f"**Next timing check: after {update:%a %d %b %Y, %H:%M IST}.** "
                + reason + "Allow the scheduled update to finish. " + ending)
    except (OSError, ValueError, TypeError):
        return "The observation window is pending; a reliable scheduled check-in time is unavailable."
