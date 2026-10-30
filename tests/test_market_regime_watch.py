import numpy as np
import pandas as pd
import pytest
pytest.importorskip("river")
from market_regime_watch import detect_market_changes


def prices(returns):
    return pd.Series(100*np.exp(np.r_[0., np.cumsum(returns)]), index=pd.bdate_range("2022-01-01", periods=len(returns)+1))


def test_constant_prices_do_not_create_change_flags():
    result = detect_market_changes(prices(np.zeros(500)))
    assert result["status"] == "no_recent_change"
    assert not result["events"]


def test_sustained_volatility_shift_is_detected():
    result = detect_market_changes(prices(np.r_[np.tile([.005,-.005],250),np.tile([.05,-.05],150)]))
    assert any(event["Change"]=="Volatility pressure increased" for event in result["events"])


def test_appending_future_prices_does_not_rewrite_old_detections():
    frame = prices(np.r_[np.tile([.005,-.005],250),np.tile([.05,-.05],150)])
    before = detect_market_changes(frame.iloc[:650])
    after = detect_market_changes(frame)
    assert before["events"] == [event for event in after["events"] if event["Observation"] < 649]


def test_insufficient_history_is_not_called_neutral():
    assert detect_market_changes(prices(np.zeros(30)))["status"] == "insufficient_history"
