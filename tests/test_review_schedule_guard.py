from datetime import datetime, timezone

from review_schedule_guard import should_run


def test_daylight_time_uses_13_utc_pair_only():
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert should_run("0 13 * * 1-5", now)
    assert not should_run("0 14 * * 1-5", now)
    assert should_run("45 13 * * 1-5", now)
    assert not should_run("45 14 * * 1-5", now)


def test_standard_time_uses_14_utc_pair_only():
    now = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert not should_run("0 13 * * 1-5", now)
    assert should_run("0 14 * * 1-5", now)
    assert not should_run("45 13 * * 1-5", now)
    assert should_run("45 14 * * 1-5", now)


def test_manual_and_non_us_windows_always_run():
    assert should_run("")
    assert should_run("15 2 * * 1-5")
