from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from public_review import market
from public_review.calendar_horizon import calendar_coverage_message, verified_forecast_policy
from public_review.forecast import METHOD, estimate
from public_review.core import digest
from public_review.service import build_assessment
from review_fixtures import baseline, policy
import test_review_calendar_coverage as coverage_fixture


def future_dates(n):
    return [str(day.date()) for day in pd.bdate_range("2026-11-05", periods=n)]


def test_37_verified_dates_support_a_37_session_forecast_without_changing_approved_policy():
    p = policy()
    p.update(max_review_sessions=40, validation_horizon=40)
    original = deepcopy(p)
    effective, coverage = verified_forecast_policy(p, future_dates(37))
    assert effective["max_review_sessions"] == effective["validation_horizon"] == 37
    assert p == original
    assert not coverage["complete"] and coverage["forecast_usable"]
    assert "37 verified sessions" in calendar_coverage_message(coverage)
    assert "unavailable" not in calendar_coverage_message(coverage)


def test_full_horizon_keeps_optional_followup_dates_and_original_policy_hash():
    p = policy()
    effective, coverage = verified_forecast_policy(p, future_dates(25))
    assert effective == p
    assert coverage["effective_forecast_sessions"] == 20
    assert coverage["available_forecast_sessions"] == 25
    assert calendar_coverage_message(coverage) is None


@pytest.mark.parametrize("days", [[], future_dates(3), future_dates(5) + [future_dates(5)[-1]], ["2027-01-04"] * 5, ["bad"] * 5])
def test_too_short_duplicated_or_unverified_dates_do_not_enable_forecasting(days):
    effective, coverage = verified_forecast_policy(policy(), days)
    assert effective is None
    assert not coverage["forecast_usable"]
    assert "forecast is unavailable" in calendar_coverage_message(coverage)


def test_scheduled_assessment_validates_and_forecasts_the_same_37_sessions():
    p, b = policy(), baseline()
    p.update(max_review_sessions=40, validation_horizon=40)
    future = future_dates(37)
    with patch("public_review.service.validate", return_value={"passed": False}) as validate, \
         patch("public_review.service.estimate", return_value={"status": "RESEARCH_ONLY", "next_review": None}) as forecast, \
         patch("public_market_mood.fetch_mmi", return_value={"status": "UNAVAILABLE"}):
        result = build_assessment(b, coverage_fixture.ReviewCalendarCoverageTests().histories(p), "2026-09-09", future,
                                  p, [], b["weights"], datetime(2026, 11, 4, 13, tzinfo=timezone.utc), comparisons=False)
    assert validate.call_args.args[4]["validation_horizon"] == 37
    assert forecast.call_args.args[4]["max_review_sessions"] == 37
    assert result["policy"]["max_review_sessions"] == 40
    assert result["forecast_policy"]["max_review_sessions"] == 37
    assert result["forecast"]["status"] == "RESEARCH_ONLY"
    assert result["metrics"]["net_total_return"] is not None


def test_real_forecast_keeps_validation_binding_to_the_reduced_policy():
    p, b = policy(), baseline()
    p.update(max_review_sessions=40, validation_horizon=40)
    future = future_dates(37)
    effective, _ = verified_forecast_policy(p, future)
    returns = pd.DataFrame({"A.NS": [.002, -.001] * 70, "B.NS": [.001, -.0005] * 70})
    validation = {"passed": True, "policy_hash": digest(effective), "method": METHOD,
                  "tickers": [lot["ticker"] for lot in b["lots"]]}
    result = estimate(b, {"A.NS": 100., "B.NS": 50.}, returns, future, effective,
                      b["capital"], validation, count=100)
    assert len(result["curve"]) == result["horizon_sessions"] == 37
    assert result["next_review"] in future
    validation["policy_hash"] = digest(p)
    assert estimate(b, {"A.NS": 100., "B.NS": 50.}, returns, future, effective,
                    b["capital"], validation, count=100)["next_review"] is None


def test_actual_november_calendar_stays_inside_the_verified_year():
    p = policy()
    p.update(max_review_sessions=40, validation_horizon=40)
    _, _, future = market.sessions(datetime(2026, 11, 4, 13, tzinfo=timezone.utc),
                                   "2026-09-30T03:43:00+00:00", p,
                                   {"A.NS": "equity", "B": "foreign_us_listing"})
    effective, coverage = verified_forecast_policy(p, future)
    assert effective is not None
    assert 5 <= effective["max_review_sessions"] < 40
    assert coverage["forecast_through"] <= "2026-12-31"
