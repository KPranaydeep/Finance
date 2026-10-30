from datetime import date
import numpy as np
import pandas as pd
from rebalance_review import METRICS, estimate_review_window, review_market_date


def test_identical_portfolios_expire_at_model_horizon_and_check_all_metrics():
    lr = pd.DataFrame({"A": np.linspace(-.01,.012,300)}, index=pd.bdate_range("2025-01-01",periods=300))
    result = estimate_review_window(lr, {"A":100}, {"A":1})
    assert result["sessions"] == 20
    assert set(result["metrics"]) == set(METRICS)
    assert len(result["timeline"]) == 20


def test_missing_inputs_require_review_now():
    assert estimate_review_window(pd.DataFrame(), {"A":100}, {"A":1})["sessions"] == 0


def test_weekend_and_expired_window_do_not_reset_evidence_clock():
    scheduled = review_market_date("2026-10-30", 1, date(2026,10,30))
    assert scheduled["date"] == "2026-11-02"
    assert scheduled["sessions_remaining"] == 1
    expired = review_market_date("2026-10-22", 1, date(2026,10,30))
    assert expired["due_now"] and expired["sessions_remaining"] == 0


def test_unverified_calendar_year_is_explicit():
    assert review_market_date("2026-12-31", 20, date(2026,12,31))["date"] is None
