"""Market-neutral owner review-window planning.

The statistical model chooses *which session date* deserves reassessment.
This module separately chooses a practical local-time window in which a human
can complete that reassessment and still reach the represented markets.  Every
exchange contributes one calendar and one next-execution timestamp; no market
receives a special weight.
"""
from datetime import date, datetime, time, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

import pandas as pd

from . import market

REVIEW_WINDOW_MODEL = "equal-market-calendar-score-v6"


def _clock(value, key):
    try:
        return time.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise ValueError("INVALID_POLICY_" + key.upper()) from None


def _ceil_minutes(value, step):
    stamp = pd.Timestamp(value)
    discarded = stamp.minute % step
    if discarded or stamp.second or stamp.microsecond or stamp.nanosecond:
        stamp += pd.Timedelta(minutes=step - discarded)
    return stamp.replace(second=0, microsecond=0, nanosecond=0)


def _local_stamp(day, clock, timezone):
    return pd.Timestamp(datetime.combine(day, clock, tzinfo=timezone))


def _market_schedules(markets, start, end, policy):
    return {
        name: market.calendar(start, end, policy, name)
        for name in sorted(markets)
    }


def _next_execution(schedule, after, opening_buffer):
    for _, row in schedule.iterrows():
        opened = pd.Timestamp(row.market_open) + opening_buffer
        closed = pd.Timestamp(row.market_close)
        if opened <= after < closed:
            return after
        if opened > after:
            return opened
    raise ValueError("INCOMPLETE_SESSION_CALENDAR")


def estimate_review_window(review_date, policy, instrument_kinds):
    """Return the best complete-data, one-hour owner review window.

    Candidate starts are generated symmetrically around every represented
    exchange's pre-open, live and post-close states, plus the configured daily
    owner start.  Candidates before the last required close are rejected.  The
    winner minimises the latest then average next execution time across all
    represented exchanges, followed by information staleness.
    """
    if not review_date or not instrument_kinds:
        return None
    timezone_name = policy.get("review_timezone", "Asia/Kolkata")
    try:
        local_tz = ZoneInfo(timezone_name)
    except Exception:
        raise ValueError("INVALID_POLICY_REVIEW_TIMEZONE") from None
    duration = int(policy.get("owner_review_duration_minutes", 60))
    step = int(policy.get("review_window_step_minutes", 15))
    opening_buffer = pd.Timedelta(
        minutes=int(policy.get("execution_wait_after_open_minutes", 15)))
    close_buffer = pd.Timedelta(
        minutes=int(policy["assessment_wait_after_close_minutes"]))
    preferred_start = _clock(
        policy.get("owner_review_day_start", "08:00"),
        "owner_review_day_start",
    )
    latest_end = _clock(
        policy.get("owner_review_day_end", "22:00"),
        "owner_review_day_end",
    )
    target = date.fromisoformat(str(review_date)[:10])
    markets = {
        market.KIND_CALENDARS.get(kind) for kind in instrument_kinds.values()
    }
    if None in markets:
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    search_end = min(
        target + timedelta(days=7),
        date.fromisoformat(policy["calendar_verified_through"]),
    )
    schedules = _market_schedules(
        markets, target - timedelta(days=1), search_end, policy
    )
    target_rows = {}
    for name, schedule in schedules.items():
        matching = schedule[pd.Index(schedule.index.date) == target]
        if matching.empty:
            raise ValueError("INCOMPLETE_SESSION_CALENDAR")
        target_rows[name] = matching.iloc[0]
    data_ready = max(
        pd.Timestamp(row.market_close) + close_buffer
        for row in target_rows.values()
    )
    data_ready = _ceil_minutes(data_ready, step)

    candidates = set()
    first_local_day = data_ready.tz_convert(local_tz).date()
    for offset in range(4):
        local_day = first_local_day + timedelta(days=offset)
        candidates.add(_local_stamp(local_day, preferred_start, local_tz))
    window_duration = pd.Timedelta(minutes=duration)
    for schedule in schedules.values():
        for _, row in schedule.iterrows():
            opened, closed = pd.Timestamp(row.market_open), pd.Timestamp(row.market_close)
            candidates.update({
                opened - opening_buffer - window_duration,
                opened + opening_buffer,
                closed + close_buffer,
            })

    ranked = []
    for raw_start in candidates:
        start = _ceil_minutes(raw_start, step)
        end = start + window_duration
        local_start, local_end = start.tz_convert(local_tz), end.tz_convert(local_tz)
        allowed_start = _local_stamp(local_start.date(), preferred_start, local_tz)
        allowed_end = _local_stamp(local_start.date(), latest_end, local_tz)
        if start < data_ready or local_start < allowed_start or local_end > allowed_end:
            continue
        try:
            executions = {
                name: _next_execution(schedule, end, opening_buffer)
                for name, schedule in schedules.items()
            }
        except ValueError as exc:
            if str(exc) == "INCOMPLETE_SESSION_CALENDAR":
                continue
            raise
        execution_values = [value.value for value in executions.values()]
        # Lexicographic priorities make every represented market count equally:
        # minimise the last market reached, then the mean market reach time,
        # then prefer fresher information and an earlier finished review.
        score = (
            max(execution_values),
            mean(execution_values),
            (start - data_ready).value,
            end.value,
        )
        ranked.append((score, start, end, executions))
    if not ranked:
        raise ValueError("NO_PRACTICAL_REVIEW_WINDOW")
    _, start, end, executions = min(ranked, key=lambda item: item[0])

    latest_close = max(
        pd.Timestamp(row.market_close) for row in target_rows.values()
    )
    completed_markets = sorted(
        name for name, row in target_rows.items()
        if pd.Timestamp(row.market_close) == latest_close
    )
    earliest_execution = min(executions.values())
    next_markets = sorted(
        name for name, timestamp in executions.items()
        if timestamp == earliest_execution
    )
    completed_label = "/".join(completed_markets)
    next_label = "/".join(next_markets)
    context = f"Post {completed_label} | Pre {next_label}"
    if set(completed_markets) == set(next_markets) and len(markets) == 1:
        context = f"Post {completed_label} | Before next session"

    start_local, end_local = start.tz_convert(local_tz), end.tz_convert(local_tz)
    return {
        "model_review_date": target.isoformat(),
        "timezone": timezone_name,
        "start_at": start_local.isoformat(),
        "end_at": end_local.isoformat(),
        "date_label": start_local.strftime("%d %b").upper(),
        "time_label": (
            f"{start_local:%H:%M}-{end_local:%H:%M} "
            + ("IST" if timezone_name == "Asia/Kolkata" else timezone_name)
        ),
        "market_context": context,
        "data_ready_at": data_ready.tz_convert(local_tz).isoformat(),
        "execution_windows": {
            name: timestamp.tz_convert(local_tz).isoformat()
            for name, timestamp in executions.items()
        },
        "selection_method": REVIEW_WINDOW_MODEL,
    }
