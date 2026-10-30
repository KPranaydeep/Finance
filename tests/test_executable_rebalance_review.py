import numpy as np
import pandas as pd
from rebalance_review import assess_executable_rebalance


def history(a=0., b=0.):
    return pd.DataFrame({"A": np.full(360, a), "B": np.full(360, b)}, index=pd.bdate_range("2025-01-01", periods=360))


def orders(amount=100.):
    return [{"Yahoo Ticker": "A", "Action": "Sell", "Executable Value": amount},
            {"Yahoo Ticker": "B", "Action": "Buy", "Executable Value": amount * .99}]


def test_equal_returns_with_costs_means_hold():
    result = assess_executable_rebalance(history(), {"A": 1000.}, orders())
    assert not result["selected_orders"]


def test_unfunded_buys_do_not_pass_even_with_excellent_history():
    result = assess_executable_rebalance(history(0, .01), {"A": 1000.}, [orders()[1]])
    assert not result["selected_orders"]
    assert result["candidates"][0]["Status"].startswith("Not funded")


def test_missing_owned_history_and_oversell_cannot_pass():
    assert not assess_executable_rebalance(history(), {"MISSING": 1000.}, orders())["selected_orders"]
    assert not assess_executable_rebalance(history(0, .01), {"A": 10.}, orders())["selected_orders"]


def test_consistent_funded_improvement_passes_and_is_reproducible():
    first = assess_executable_rebalance(history(0, .002), {"A": 1000.}, orders())
    second = assess_executable_rebalance(history(0, .002), {"A": 1000.}, orders())
    assert first == second
    assert first["selected_orders"]
    assert first["candidates"][0]["Worst segment lower bound (log wealth)"] > 0


def test_segment_reversal_blocks_full_history_average_story():
    lr = history(0, .01)
    lr.loc[lr.index[-120:], "B"] = -.02
    result = assess_executable_rebalance(lr, {"A": 1000.}, orders())
    assert not result["selected_orders"]
