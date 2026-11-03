import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer_core as core
from trade_quantities import format_quantity, plan_quantity, quantity_step


def test_fractional_precision_and_whole_unit_fallback():
    assert quantity_step(2.8659, "USD") == 0.0001
    assert quantity_step(0.00123456, "USD") == 0.00000001
    assert quantity_step(2, "USD") == 1
    assert quantity_step(2.8659, "INR") == 1
    assert quantity_step(2.8659, "") == 1
    assert plan_quantity(0.123456, 2.8659, "USD", "Buy") == 0.1234
    assert plan_quantity(0.5, 2, "USD", "Buy") == 0
    assert plan_quantity(1.2345, 2.8659, "USD", "Buy", 3) == 0.4115
    assert format_quantity(0.00123456) == "0.00123456"
    assert format_quantity(475) == "475"


@pytest.mark.parametrize("held,currency,expected", [(0.37, "USD", 0.37), (6, "INR", 6), (0.37, "INR", 0)])
def test_exits_do_not_withhold_an_extra_share(held, currency, expected):
    assert plan_quantity(held, held, currency, "Sell") == expected
    assert plan_quantity(held + 10, held, currency, "Sell") <= held


@pytest.mark.parametrize("gap,held,days", [(float("nan"), 1, 1), (1, float("inf"), 1), (1, 1, 0), (-1, 1, 1)])
def test_invalid_sizing_fails_explicitly(gap, held, days):
    with pytest.raises(ValueError):
        plan_quantity(gap, held, "USD", "Buy", days)


def test_fractional_rebalance_and_summary_preserve_quantity_and_value():
    allocation = pd.DataFrame([
        {"Symbol": "A", "Yahoo Ticker": "A", "Currency": "USD", "Quantity": 0.37, "FX to INR": 90, "Weight": 0.37, "Latest Price INR": 900},
        {"Symbol": "B", "Yahoo Ticker": "B", "Currency": "USD", "Quantity": 0.63, "FX to INR": 90, "Weight": 0.63, "Latest Price INR": 900},
    ])
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.01], "B": [0.02, 0.01, -0.01]})
    plan, _, _ = core.rebalance_plan_multi(allocation, np.array([0, 1]), returns, {"A": 10, "B": 10}, 1)
    indexed = plan.set_index("Yahoo Ticker")
    assert indexed.loc["A", "Executable Quantity"] == pytest.approx(0.37)
    assert indexed.loc["B", "Executable Quantity"] == pytest.approx(0.37)
    assert indexed.loc["A", "Executable Value"] == pytest.approx(333)
    summary = core.build_holdings_action_summary(allocation, plan, {"A": 0, "B": 1}, {"A": 10, "B": 10}, {})
    assert summary.set_index("Symbol").loc["A", "Quantity"] == pytest.approx(0.37)
    assert "0.37" in core.style_holdings_action_summary(summary).to_html()
    assert "0.37" in core.style_rebalance_df(plan).to_html()


def test_summary_distinguishes_already_at_target_from_no_saved_order():
    allocation = pd.DataFrame([
        {"Symbol": "A", "Yahoo Ticker": "A", "Currency": "USD", "Quantity": 0.37, "FX to INR": 90, "Weight": 0.37, "Latest Price INR": 900},
    ])
    summary = core.build_holdings_action_summary(allocation, pd.DataFrame(), {"A": 0.37}, {"A": 10}, {})
    assert summary.iloc[0]["Status"] == "Already at the optimal allocation"
    summary = core.build_holdings_action_summary(allocation, pd.DataFrame(), {"A": 0.5}, {"A": 10}, {})
    assert "planning quantity step 0.01" in summary.iloc[0]["Status"]
    assert "whole share" not in summary.iloc[0]["Status"]
