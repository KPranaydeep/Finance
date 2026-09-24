import numpy as np
import pandas as pd

import portfolio_optimizer_core as core


def _allocation():
    return pd.DataFrame(
        [
            {
                "Symbol": "A",
                "Yahoo Ticker": "A.NS",
                "Currency": "INR",
                "Quantity": 80,
                "FX to INR": 1.0,
                "Weight": 0.80,
                "Latest Price INR": 1.0,
            },
            {
                "Symbol": "B",
                "Yahoo Ticker": "B.NS",
                "Currency": "INR",
                "Quantity": 10,
                "FX to INR": 1.0,
                "Weight": 0.10,
                "Latest Price INR": 1.0,
            },
            {
                "Symbol": "C",
                "Yahoo Ticker": "C.NS",
                "Currency": "INR",
                "Quantity": 10,
                "FX to INR": 1.0,
                "Weight": 0.10,
                "Latest Price INR": 1.0,
            },
        ]
    )


def _returns():
    return pd.DataFrame(
        {
            "A.NS": [0.001, 0.002, -0.001],
            "B.NS": [0.002, 0.003, 0.001],
            "C.NS": [-0.002, -0.001, -0.003],
        }
    )


def test_rebalance_plan_omits_partial_sells_but_keeps_buys_and_zero_target_exits():
    plan, missing_prices, missing_alloc = core.rebalance_plan_multi(
        _allocation(),
        np.array([0.50, 0.50, 0.0]),
        _returns(),
        {"A.NS": 1.0, "B.NS": 1.0, "C.NS": 1.0},
        days_to_flip=1,
    )

    assert missing_prices == []
    assert missing_alloc == []
    assert "A.NS" not in set(plan["Yahoo Ticker"])
    assert set(zip(plan["Yahoo Ticker"], plan["Action"])) == {
        ("B.NS", "Buy"),
        ("C.NS", "Sell"),
    }
    assert float(plan.set_index("Yahoo Ticker").loc["C.NS", "Optimal Weight"]) == 0.0


def test_holdings_summary_explains_suppressed_partial_sell():
    allocation = _allocation()
    summary = core.build_holdings_action_summary(
        allocation,
        pd.DataFrame(),
        {"A.NS": 0.50, "B.NS": 0.50, "C.NS": 0.0},
        {"A.NS": 1.0, "B.NS": 1.0, "C.NS": 1.0},
        {"redundant_df": pd.DataFrame(), "dropped_df": pd.DataFrame()},
    )

    row = summary.set_index("Symbol").loc["A"]
    assert row["Action"] == "Hold"
    assert "Partial sell suppressed" in row["Status"]
