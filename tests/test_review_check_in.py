from datetime import datetime, timezone
import pytest
from streamlit.testing.v1 import AppTest

from public_review.check_in import next_scheduled_update, timing_check_message
from public_review.calendar_horizon import calendar_coverage_message


def test_weekend_check_follows_the_workflow_instead_of_assuming_an_eight_am_run():
    now = datetime(2026, 11, 4, 10, tzinfo=timezone.utc)
    payload = {"forecast_observation_pending": True,
               "observation_ready_at": "2026-11-06T21:30:00+00:00",
               "policy": {"minimum_forecast_review_sessions": 4},
               "decision": {"reasons": ["SECURITY_TARGET_REVIEW"]}}
    message = timing_check_message(payload, now)
    assert "Sat 07 Nov 2026, 18:15 IST" in message
    assert "4 post-entry sessions complete 07 Nov, 03:00 IST" in message
    assert "active review flag still applies now" in message


def test_schedule_honors_the_existing_us_dst_guard():
    assert next_scheduled_update(datetime(2026, 11, 4, 12, 30, tzinfo=timezone.utc)) == datetime(2026, 11, 4, 14, tzinfo=timezone.utc)
    assert next_scheduled_update(datetime(2026, 7, 8, 12, 30, tzinfo=timezone.utc)) == datetime(2026, 7, 8, 13, tzinfo=timezone.utc)


def test_elapsed_observation_window_uses_a_future_update_and_does_not_claim_validation():
    payload = {"forecast_observation_pending": True,
               "observation_ready_at": "2026-11-06T21:30:00+00:00", "policy": {}}
    message = timing_check_message(payload, datetime(2026, 11, 7, 13, tzinfo=timezone.utc))
    assert "Sun 08 Nov 2026, 18:15 IST" in message
    assert "elapsed" in message and "Validation is still required" in message


def test_unknown_schedule_or_naive_timestamp_never_invents_a_time(tmp_path):
    with pytest.raises(OSError):
        next_scheduled_update(datetime(2026, 11, 4, tzinfo=timezone.utc), tmp_path / "missing")
    unsupported = tmp_path / "schedule.yml"
    unsupported.write_text('- cron: "*/15 * * * *"')
    with pytest.raises(ValueError):
        next_scheduled_update(datetime(2026, 11, 4, tzinfo=timezone.utc), unsupported)
    assert timing_check_message({"forecast_observation_pending": True, "observation_ready_at": "2026-11-07T03:00:00"}, datetime.now(timezone.utc)) is None


def test_calendar_gap_is_distinguished_from_missing_historical_prices():
    message = calendar_coverage_message({"complete": False, "forecast_usable": True,
        "effective_forecast_sessions": 37, "required_forecast_sessions": 40,
        "forecast_through": "2026-12-31"})
    assert "remaining 3" in message and "not missing observed prices" in message


def test_review_panel_distinguishes_current_trigger_from_future_timing_check():
    def app():
        from datetime import datetime, timezone
        from public_review.ui import render_events
        from public_review.core import evaluate, decision
        from review_fixtures import baseline, policy
        b, p = baseline(), policy()
        p["minimum_forecast_review_sessions"] = 4
        now = datetime(2026, 11, 4, 10, tzinfo=timezone.utc)
        m = evaluate(b, {"A.NS": 100., "B.NS": 50.}, "2026-11-03", p)
        d = decision(m, b, b["weights"], p, 10000.)
        d.update(reasons=["SECURITY_TARGET_REVIEW"], review_required=True, target_crossed_securities=["A.NS"])
        payload = {"checked_at": now.isoformat(), "as_of": "2026-11-03", "metrics": m,
            "decision": d, "forecast": {"status": "AWAITING_MINIMUM_OBSERVATION_SESSIONS"},
            "policy": p, "validation": {"passed": False}, "comparisons": [], "mmi": {},
            "forecast_observation_pending": True, "observation_ready_at": "2026-11-06T21:30:00+00:00",
            "dividend_assumption": "Synthetic fixture"}
        render_events([{"kind": "BASELINE", "seq": 1, "baseline_id": b["baseline_id"], "payload": b},
            {"kind": "ASSESSMENT", "seq": 2, "baseline_id": b["baseline_id"], "payload": payload, "event_hash": "synthetic"}], now=now)
    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception
    assert at.metric[0].label == "Review status" and at.metric[0].value == "Now"
    assert any("18:15 IST" in item.value for item in at.info)
    assert any("individual profit-review thresholds crossed: A.NS" in item.value for item in at.warning)
    assert not any("provisional until" in item.value for item in at.caption)
