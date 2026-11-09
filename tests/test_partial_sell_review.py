import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer_core as core
from portfolio_optimizer_config import RISK_FREE_RATE_ANNUAL
from partial_sell_review import partial_sell_caption, review_partial_sells


def fixture():
    rng = np.random.default_rng(472)
    noise = rng.normal(size=(300, 2))
    noise -= noise.mean(axis=0)
    history = pd.DataFrame(noise * [0.03, 0.004] + [0.0002, 0.001], columns=["A", "B"])
    allocation = pd.DataFrame([
        {"Symbol": "A", "Yahoo Ticker": "A", "Currency": "USD", "Quantity": 8.5, "FX to INR": 1, "Weight": 0.85, "Latest Price INR": 10},
        {"Symbol": "B", "Yahoo Ticker": "B", "Currency": "USD", "Quantity": 1.5, "FX to INR": 1, "Weight": 0.15, "Latest Price INR": 10},
    ])
    return allocation, history


def test_partial_sale_allowed_when_rounded_portfolio_net_sharpe_improves():
    allocation, history = fixture()
    plan, _, _ = core.rebalance_plan_multi(allocation, [0.5, 0.5], history, {"A": 10, "B": 10}, 1,
                                          maximum_weight=1.)
    check = plan.attrs["partial_sell_sharpe_check"]
    assert check["allowed"]
    assert check["proposed_net_sharpe"] > check["current_sharpe"]
    sale = plan.set_index("Yahoo Ticker").loc["A"]
    assert sale["Action"] == "Sell"
    assert sale["Optimal Weight"] == 0.5
    assert sale["Executable Quantity"] == pytest.approx(3.5)
    assert sale["Executable Value"] == pytest.approx(35)
    assert "Sharpe" in partial_sell_caption(check)


def test_costs_can_prevent_an_otherwise_improving_trim():
    allocation, history = fixture()
    plan, _, _ = core.rebalance_plan_multi(allocation, [0.5, 0.5], history, {"A": 10, "B": 10}, 1, transaction_cost_bps=5000)
    assert not plan.attrs["partial_sell_sharpe_check"]["allowed"]
    assert "A" not in set(plan["Yahoo Ticker"])


def test_partial_sale_with_worse_sharpe_is_withheld():
    allocation, history = fixture()
    history[["A", "B"]] = history[["B", "A"]].to_numpy()
    plan, _, _ = core.rebalance_plan_multi(allocation, [0.5, 0.5], history, {"A": 10, "B": 10}, 1)
    assert not plan.attrs["partial_sell_sharpe_check"]["allowed"]
    assert "A" not in set(plan["Yahoo Ticker"])


def test_missing_owned_history_does_not_qualify_a_trim_using_a_subset():
    allocation, history = fixture()
    allocation.loc[len(allocation)] = ["MISSING", "MISSING", "USD", 1, 1, 0.1, 10]
    plan, _, _ = core.rebalance_plan_multi(allocation, [0.5, 0.5], history, {"A": 10, "B": 10, "MISSING": 10}, 1)
    assert not plan.attrs["partial_sell_sharpe_check"]["allowed"]
    assert "Incomplete" in plan.attrs["partial_sell_sharpe_check"]["reason"]
    assert "A" not in set(plan["Yahoo Ticker"])


def test_paused_trades_are_removed_before_evaluating_sharpe():
    allocation, history = fixture()
    plan, _, _ = core.rebalance_plan_multi(allocation, [0.5, 0.5], history, {"A": 10, "B": 10}, 1, excluded_trade_tickers=["b"])
    assert "B" not in set(plan["Yahoo Ticker"])
    expected_orders = pd.DataFrame([{"Yahoo Ticker": "A", "Action": "Sell", "Optimal Weight": .5, "Executable Value": 35}])
    expected = review_partial_sells(history, {"A": 85, "B": 15}, expected_orders,
                                   cost_bps=22, risk_free_rate=RISK_FREE_RATE_ANNUAL)
    assert plan.attrs["partial_sell_sharpe_check"]["proposed_net_sharpe"] == expected["proposed_net_sharpe"]


def test_incremental_trim_must_also_improve_a_feasible_plan_without_it():
    _, history = fixture()
    # Selling the strong security would piggyback on a much more useful exit.
    history["C"] = -0.002 + np.linspace(-0.01, 0.01, len(history))
    orders = pd.DataFrame([
        {"Yahoo Ticker": "C", "Action": "Sell", "Optimal Weight": 0, "Executable Value": 60},
        {"Yahoo Ticker": "B", "Action": "Sell", "Optimal Weight": 0.1, "Executable Value": 15},
    ])
    check = review_partial_sells(history, {"A": 10, "B": 30, "C": 60}, orders,
                                 cost_bps=22, risk_free_rate=.08225)
    assert not check["allowed"]
    assert check["proposed_net_sharpe"] < check["without_trims_net_sharpe"]


@pytest.mark.parametrize("broken", ["short", "flat", "nonfinite", "cost"])
def test_inadequate_inputs_never_qualify_a_partial_sale(broken):
    _, history = fixture()
    if broken == "short":
        history = history.iloc[:3]
    elif broken == "flat":
        history[:] = 0
    elif broken == "nonfinite":
        history[:] = np.nan
    orders = pd.DataFrame([{"Yahoo Ticker": "A", "Action": "Sell", "Optimal Weight": .5, "Executable Value": 35}])
    check = review_partial_sells(history, {"A": 85, "B": 15}, orders,
                                 cost_bps=float("nan") if broken == "cost" else 22,
                                 risk_free_rate=.08225)
    assert not check["allowed"]
