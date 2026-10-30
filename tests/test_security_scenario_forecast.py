import numpy as np
import pandas as pd

from security_scenario_forecast import (
    METHOD,
    _stationary_bootstrap_terminal_log_returns,
    forecast_executable_buys,
    independent_security_log_returns,
    forecast_holdings_for_exit,
    normalize_saved_scenario_ranking,
)


def plan():
    return pd.DataFrame(
        [
            {"Symbol": "BUY_A", "Yahoo Ticker": "A", "Action": "Buy",
             "Executable Quantity": 4, "Optimal Weight": 0.20, "Current Weight": 0.05},
            {"Symbol": "SELL_B", "Yahoo Ticker": "B", "Action": "Sell",
             "Executable Quantity": 10, "Optimal Weight": 0.0},
            {"Symbol": "ZERO_C", "Yahoo Ticker": "C", "Action": "Buy",
             "Executable Quantity": 0, "Optimal Weight": 0.10},
        ]
    )


def test_scenario_rank_contains_executable_buys_only_and_is_reproducible():
    index = pd.bdate_range("2023-01-01", periods=500)
    returns = pd.DataFrame(
        {
            "A": np.linspace(-0.01, 0.015, len(index)),
            "B": np.linspace(0.02, -0.01, len(index)),
            "C": np.zeros(len(index)),
        },
        index=index,
    )
    first = forecast_executable_buys(
        plan(), returns, horizon_sessions=20, one_way_cost_bps=22, paths=500, seed=7
    )
    second = forecast_executable_buys(
        plan(), returns, horizon_sessions=20, one_way_cost_bps=22, paths=500, seed=7
    )

    assert first["Yahoo Ticker"].tolist() == ["A"]
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0]["Scenario method"] == METHOD
    assert first.iloc[0]["Optimized Weight (%)"] == 20.0
    assert 0.0 <= first.iloc[0]["Probability of positive net return (%)"] <= 100.0


def test_entry_and_exit_costs_reduce_scenario_score():
    index = pd.bdate_range("2023-01-01", periods=300)
    returns = pd.DataFrame({"A": np.full(len(index), 0.001)}, index=index)
    no_cost = forecast_executable_buys(
        plan(), returns, horizon_sessions=10, one_way_cost_bps=0, paths=500, seed=9
    )
    with_cost = forecast_executable_buys(
        plan(), returns, horizon_sessions=10, one_way_cost_bps=22, paths=500, seed=9
    )
    assert (
        with_cost.iloc[0]["Expected net scenario return (%)"]
        < no_cost.iloc[0]["Expected net scenario return (%)"]
    )


def test_single_buy_score_is_return_times_probability_not_portfolio_weight():
    index = pd.bdate_range("2023-01-01", periods=300)
    returns = pd.DataFrame(
        {
            "A": np.full(len(index), 0.001),
            "B": np.full(len(index), 0.0001),
        },
        index=index,
    )
    buys = pd.DataFrame(
        [
            {"Symbol": "A", "Yahoo Ticker": "A", "Action": "Buy",
             "Executable Quantity": 1, "Optimal Weight": 0.06, "Current Weight": 0.05},
            {"Symbol": "B", "Yahoo Ticker": "B", "Action": "Buy",
             "Executable Quantity": 1, "Optimal Weight": 0.95, "Current Weight": 0.05},
        ]
    )
    result = forecast_executable_buys(
        buys, returns, horizon_sessions=10, one_way_cost_bps=0, paths=300, seed=11
    )
    assert result.iloc[0]["Yahoo Ticker"] == "A"
    np.testing.assert_allclose(
        result["Return × positive-probability score (%)"],
        result["Expected net scenario return (%)"]
        * result["Probability of positive net return (%)"] / 100.0,
    )
    assert "Expected portfolio contribution at target weight (percentage points)" not in result


def test_no_executable_buy_returns_an_empty_typed_result():
    returns = pd.DataFrame({"A": [0.01, -0.01]})
    result = forecast_executable_buys(
        plan().query("Action == 'Sell'"), returns,
        horizon_sessions=2, paths=100,
    )
    assert result.empty
    assert "Return × positive-probability score (%)" in result.columns


def test_saved_legacy_rows_get_rescored_and_sorted_without_weight():
    records = [
        {"Security": "B", "Yahoo Ticker": "B", "Optimized Weight (%)": 90,
         "Expected net scenario return (%)": 4.0,
         "Probability of positive net return (%)": 60.0},
        {"Security": "A", "Yahoo Ticker": "A", "Optimized Weight (%)": 1,
         "Expected net scenario return (%)": 5.0,
         "Probability of positive net return (%)": 70.0},
    ]
    ranked = normalize_saved_scenario_ranking(records)
    assert ranked["Yahoo Ticker"].tolist() == ["A", "B"]
    assert ranked.iloc[0]["Return × positive-probability score (%)"] == 3.5


def test_stationary_bootstrap_is_finite_and_reproducible():
    values = np.array([0.01, -0.02, 0.03, 0.0])
    a = _stationary_bootstrap_terminal_log_returns(
        values, horizon=10, paths=100, rng=np.random.default_rng(3)
    )
    b = _stationary_bootstrap_terminal_log_returns(
        values, horizon=10, paths=100, rng=np.random.default_rng(3)
    )
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_scenario_history_keeps_latest_date_despite_stale_peer():
    dates = pd.bdate_range("2026-10-01", periods=20)
    prices = pd.DataFrame({"FRESH": np.arange(100., 120.), "STALE": np.arange(200., 220.)}, index=dates)
    prices.loc[dates[-5:], "STALE"] = np.nan
    returns = independent_security_log_returns(prices)
    assert returns["FRESH"].dropna().index.max() == dates[-1]
    assert returns["STALE"].dropna().index.max() == dates[-6]
    orders = pd.DataFrame({"Yahoo Ticker": ["FRESH", "STALE"], "Action": ["Buy", "Buy"], "Executable Quantity": [1, 1], "Optimal Weight": [0.5, 0.5]})
    ranked = forecast_executable_buys(orders, returns, horizon_sessions=20, paths=100)
    by_ticker = ranked.set_index("Yahoo Ticker")
    assert by_ticker.loc["FRESH", "Price data through"] == str(dates[-1].date())
    assert by_ticker.loc["STALE", "Price data through"] == str(dates[-6].date())


def test_holdings_exit_ranking_is_inverse_and_reports_current_weight():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    returns = pd.DataFrame({"GOOD": np.full(80, 0.01), "BAD": np.full(80, -0.01)}, index=dates)
    portfolio = pd.DataFrame({
        "Symbol": ["GOOD", "BAD"], "Yahoo Ticker": ["GOOD", "BAD"],
        "Quantity": [10, 10], "Current Weight": [0.6, 0.4],
    })
    result = forecast_holdings_for_exit(portfolio, returns, horizon_sessions=5, one_way_cost_bps=0, paths=300)
    assert result["Security"].tolist() == ["BAD", "GOOD"]
    assert result.iloc[0]["Current Weight (%)"] == 40.0
