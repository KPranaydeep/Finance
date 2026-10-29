import pytest

from portfolio_reserve import (
    calculate_leverage_stress_check,
    calculate_portfolio_reserve_target,
)


def test_reserve_uses_bootstrap_median_plus_downside_tail_sd():
    result = calculate_portfolio_reserve_target(
        {
            "Block-Bootstrap Median Return (20 Sessions)": 0.02,
            "Block-Bootstrap Tail Loss SD 95% (20 Sessions)": 0.03,
        },
        invested_base=100_000,
    )

    assert result["median_return"] == pytest.approx(0.02)
    assert result["tail_loss_sd"] == pytest.approx(0.03)
    assert result["reserve_fraction"] == pytest.approx(0.03)
    assert result["reserve_amount"] == pytest.approx(3_000)
    assert result["method"] == "moving-block bootstrap"


def test_negative_central_path_increases_reserve_reference():
    result = calculate_portfolio_reserve_target(
        {
            "Block-Bootstrap Median Return (20 Sessions)": -0.01,
            "Block-Bootstrap Tail Loss SD 95% (20 Sessions)": 0.03,
        },
        invested_base=100_000,
    )

    assert result["reserve_fraction"] == pytest.approx(0.04)


def test_older_saved_run_uses_labeled_volatility_based_fallback():
    result = calculate_portfolio_reserve_target(
        {"Annual Return": 0.20, "Annual Volatility": 0.15},
        invested_base=100_000,
    )

    assert result["reserve_fraction"] > 0
    assert result["method"] == "normal-tail volatility proxy (older saved run)"


def test_reserve_requires_saved_portfolio_distribution_or_fallback_stats():
    with pytest.raises(ValueError):
        calculate_portfolio_reserve_target({}, invested_base=100_000)


def test_leverage_ceiling_respects_user_loss_budget():
    result = calculate_leverage_stress_check(
        invested_base=100_000,
        expected_shortfall=0.05,
        portfolio_annual_return=0.20,
        borrowing_rate=0.08,
        loss_budget=0.10,
        maintenance_equity_ratio=0.20,
    )

    assert result["historical_return_minus_borrowing_rate"] == pytest.approx(0.12)
    assert result["modeled_horizon_loss_fraction"] == pytest.approx(0.10)
    assert result["debt_amount"] == pytest.approx(100_000 * (0.05 / (0.05 + 0.08 * 20 / 252)))


def test_leverage_ceiling_respects_maintenance_equity_requirement():
    result = calculate_leverage_stress_check(
        invested_base=100_000,
        expected_shortfall=0.05,
        portfolio_annual_return=0.20,
        borrowing_rate=0.03,
        loss_budget=0.50,
        maintenance_equity_ratio=0.50,
    )

    assert result["stressed_equity_ratio"] == pytest.approx(0.50)
    assert result["debt_ratio_to_base"] == pytest.approx(
        result["maintenance_debt_ratio_limit"]
    )


def test_no_borrowing_when_unlevered_tail_loss_exceeds_user_budget():
    result = calculate_leverage_stress_check(
        invested_base=100_000,
        expected_shortfall=0.05,
        portfolio_annual_return=0.20,
        borrowing_rate=0.03,
        loss_budget=0.04,
        maintenance_equity_ratio=0.50,
    )

    assert result["debt_amount"] == 0
    assert result["modeled_horizon_loss_fraction"] == pytest.approx(0.05)


def test_leverage_check_rejects_out_of_range_margin_assumption():
    with pytest.raises(ValueError):
        calculate_leverage_stress_check(100, 0.05, 0.2, 0.08, 0.2, 1.0)
