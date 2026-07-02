import pandas as pd
import pytest

from analytics import (
    max_drawdown,
    previous_completed_quarter,
    recommend_dry_powder,
    run_dry_powder_backtest,
)


def test_max_drawdown():
    prices = pd.Series([100, 110, 88, 99])
    assert max_drawdown(prices) == pytest.approx(0.20)


def test_formula_minimum():
    prices = pd.Series([100, 110, 71.5], index=pd.date_range("2024-01-01", periods=3))
    rec = recommend_dry_powder(
        prices=prices,
        tolerance_drawdown=0.25,
        policy_floor=0.0,
        custom_stress_drawdown=0.35,
    )
    assert rec.formula_minimum_weight == pytest.approx(1 - 0.25 / 0.35)
    assert rec.recommended_weight == pytest.approx(0.30)


def test_previous_completed_quarter():
    start, end = previous_completed_quarter(pd.Timestamp("2026-07-02"))
    assert start == pd.Timestamp("2026-04-01")
    assert end == pd.Timestamp("2026-06-30")


def test_backtest_deploys_on_drawdown():
    dates = pd.date_range("2026-04-01", periods=6, freq="D")
    prices = pd.Series([100, 105, 99, 94, 90, 98], index=dates)
    result, summary, events = run_dry_powder_backtest(
        prices=prices,
        start_date=dates[0],
        end_date=dates[-1],
        initial_capital=2_000_000,
        dry_powder_weight=0.20,
        thresholds=[0.05, 0.10],
        annual_cash_yield=0.0,
    )
    assert len(result) == 6
    assert summary.triggers_hit == 2
    assert summary.cash_deployed == pytest.approx(400_000)
    assert len(events) == 2


def test_nse_payload_parser():
    from market_data import _nse_payload_to_close

    payload = {
        "data": {
            "indexCloseOnlineRecords": [
                {"EOD_TIMESTAMP": "01-04-2026", "EOD_CLOSE_INDEX_VAL": "23,100.50"},
                {"EOD_TIMESTAMP": "02-04-2026", "EOD_CLOSE_INDEX_VAL": "23,250.00"},
            ]
        }
    }
    series = _nse_payload_to_close(payload, "NIFTY 50")
    assert len(series) == 2
    assert series.iloc[-1] == pytest.approx(23250.0)
