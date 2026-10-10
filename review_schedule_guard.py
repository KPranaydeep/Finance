"""Skip the redundant half of paired US-market UTC schedules."""

from __future__ import annotations

import os
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")
PAIRED_US_SCHEDULE_HOURS = {
    "0 13 * * 1-5": 13,
    "0 14 * * 1-5": 14,
    "45 13 * * 1-5": 13,
    "45 14 * * 1-5": 14,
}


def should_run(schedule: str, now: datetime | None = None) -> bool:
    """Run only the cron matching 09:30 New York in the current DST regime."""
    expected_hour = PAIRED_US_SCHEDULE_HOURS.get(schedule.strip())
    if expected_hour is None:
        return True
    now = now or datetime.now(timezone.utc)
    ny_date = now.astimezone(NEW_YORK).date()
    market_open_utc = datetime.combine(
        ny_date, time(9, 30), tzinfo=NEW_YORK
    ).astimezone(timezone.utc)
    return expected_hour == market_open_utc.hour


def main() -> int:
    schedule = os.getenv("PUBLIC_REVIEW_CRON", "")
    run = should_run(schedule)
    output = os.getenv("GITHUB_OUTPUT")
    line = f"run={'true' if run else 'false'}\n"
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(line)
    print(
        "review schedule accepted"
        if run else
        "redundant DST/standard-time review schedule skipped before database access"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
