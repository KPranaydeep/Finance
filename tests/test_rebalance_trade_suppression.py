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


def test_suppressed_trim_cannot_leave_an_overweight_plan_marked_executable():
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
    assert plan.empty
    assert not plan.attrs["execution_validation"]["passed"]
    assert any("weight cap" in v for v in plan.attrs["execution_validation"]["violations"])
    proposal = pd.DataFrame(plan.attrs["proposed_rebalancing_plan"]).set_index("Yahoo Ticker")
    assert proposal.loc["C.NS", "Optimal Weight"] == 0.
    assert proposal.loc["B.NS", "Executable Quantity"] <= 9


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
    assert "No executable trim generated" in row["Status"]
