import math

import numpy as np
import pytest

from leverage_assessment import (
    LAMF_ADVERTISED_START_RATE,
    assess_leverage_scenarios,
    estimate_drawdown_conditioned_debt,
    estimate_growth_optimal_debt,
)


def test_financing_cost_is_applied_to_debt_only_and_reduces_return():
    result = assess_leverage_scenarios([0.001] * 252)
    unlevered, borrowed = result["rows"][:2]

    assert result["observations"] == 252
    assert unlevered["annualized_net_return"] == pytest.approx(math.expm1(0.001 * 252))
    assert borrowed["annual_interest_per_100k"] == pytest.approx(
        0.1 * LAMF_ADVERTISED_START_RATE * 100_000
    )
    assert borrowed["annualized_net_return"] > unlevered["annualized_net_return"]
    assert borrowed["positive_folds"] == 3
    assert borrowed["folds_beating_unlevered"] == 3


def test_financing_rate_changes_net_result_without_changing_unlevered_case():
    returns = [0.0008] * 180
    low_rate = assess_leverage_scenarios(returns, borrowing_rate=0.08)
    high_rate = assess_leverage_scenarios(returns, borrowing_rate=0.14)

    assert low_rate["rows"][0]["annualized_net_return"] == pytest.approx(
        high_rate["rows"][0]["annualized_net_return"]
    )
    assert high_rate["rows"][1]["annualized_net_return"] < low_rate["rows"][1]["annualized_net_return"]


def test_tail_loss_and_drawdown_increase_with_exposure_for_volatile_path():
    returns = np.tile([0.01, -0.012, 0.004, -0.02, 0.015], 100)
    result = assess_leverage_scenarios(returns)
    unlevered, doubled = result["rows"][0], result["rows"][-1]

    assert doubled["tail_expected_shortfall"] > unlevered["tail_expected_shortfall"]
    assert doubled["historical_max_drawdown"] > unlevered["historical_max_drawdown"]
    assert "not independent out-of-sample" in result["selection_bias_warning"]


def test_insufficient_or_invalid_history_is_rejected():
    with pytest.raises(ValueError, match="not enough"):
        assess_leverage_scenarios([0.001] * 40)
    with pytest.raises(ValueError, match="invalid"):
        assess_leverage_scenarios([0.001] * 100 + [float("nan")] + [0.001] * 100)


def test_growth_model_returns_zero_when_history_cannot_cover_financing_cost():
    result = estimate_growth_optimal_debt(
        [0.0001] * 240,
        bootstrap_paths=500,
        grid_step=0.1,
    )

    assert result["recommended_debt_ratio_of_portfolio_base"] == 0.0
    assert result["improvement_vs_unlevered"] == pytest.approx(0.0)


def test_growth_model_returns_explicit_model_amount_and_search_boundary():
    result = estimate_growth_optimal_debt(
        [0.001] * 240,
        borrowing_rate=0.05,
        bootstrap_paths=500,
        grid_step=0.1,
    )

    assert result["recommended_debt_ratio_of_portfolio_base"] == 1.0
    assert result["recommended_gross_exposure"] == 2.0
    assert result["at_search_boundary"] is True
    assert "not a personal risk limit" in result["warning"]


def test_drawdown_conditioned_model_uses_latest_saved_portfolio_drawdown():
    # A persistent decline leaves the latest saved portfolio 20% below its high.
    returns = np.r_[np.full(120, 0.001), np.full(120, -0.001)]
    result = estimate_drawdown_conditioned_debt(
        returns,
        borrowing_rate=0.20,
        horizon_sessions=10,
        drawdown_lookback_sessions=120,
        minimum_analogs=20,
        analog_fraction=0.2,
        grid_step=0.1,
    )

    assert result["current_drawdown"] < -0.1
    assert result["analog_count"] >= 20
    assert result["recommended_debt_ratio_of_portfolio_base"] == 0.0
    assert result["lower_tail_log_growth"] == pytest.approx(
        result["unlevered_lower_tail_log_growth"]
    )
    assert "not a safe borrowing limit" in result["warning"]


def test_leverage_assessment_includes_rate_specific_drawdown_scenarios():
    from leverage_assessment import build_portfolio_leverage_assessment

    result = build_portfolio_leverage_assessment(
        np.tile([0.002, -0.001], 175),
        invested_base=100_000,
    )
    base = result["scenarios"][0]["drawdown_conditioned"]

    assert base["estimated_debt_amount"] is None
    assert base["borrowing_amount_established"] is False
    assert base["current_drawdown"] <= 0
    assert base["analog_count"] > 0


def test_drawdown_forward_windows_exclude_state_day_return():
    logs = np.random.default_rng(91).normal(.001,.015,100)
    result = estimate_drawdown_conditioned_debt(logs, horizon_sessions=3,
        drawdown_lookback_sessions=5, minimum_analogs=4, analog_fraction=1,
        grid_step=.25, bootstrap_paths=100)
    states = np.arange(4, len(logs)-3)
    expected = np.quantile(logs[states[:,None]+1+np.arange(3)].sum(axis=1), .05)
    assert result["candidate_rows"][0]["lower_tail_log_growth"] == pytest.approx(expected)
    assert result["forward_windows_start_after_state"]
    indices = result["independent_state_indices"]
    assert all(b-a >= 3 for a,b in zip(indices,indices[1:]))


def test_overlap_and_boundary_do_not_establish_borrowing_capacity():
    result = estimate_drawdown_conditioned_debt([.001]*644, grid_step=.25)
    assert result["raw_historical_optimum_debt_ratio"] == 1
    assert result["at_search_boundary"]
    assert result["independent_window_count"] < result["analog_count"]
    assert result["evidence_status"] == "insufficient_independent_windows"
    assert result["validated_research_debt_ratio"] is None
    assert result["estimated_debt_amount"] is None


def test_boundary_is_blocked_even_with_sufficient_disjoint_history():
    result = estimate_drawdown_conditioned_debt([.002]*400, horizon_sessions=3,
        drawdown_lookback_sessions=5, minimum_analogs=4, analog_fraction=1,
        minimum_independent_windows=30, bootstrap_paths=100, grid_step=.25)
    assert result["uncertainty_checks"]
    assert result["evidence_status"] == "search_boundary_unresolved"
    assert result["validated_research_debt_ratio"] is None


def test_rate_stress_can_reject_an_apparent_base_rate_advantage():
    result = estimate_drawdown_conditioned_debt([.0005]*400, horizon_sessions=3,
        drawdown_lookback_sessions=5, minimum_analogs=4, analog_fraction=1,
        minimum_independent_windows=30, bootstrap_paths=100, grid_step=.25)
    assert result["raw_historical_optimum_debt_ratio"] > 0
    assert all(row["worst_adjusted_lower_bound"] < 0 for row in result["uncertainty_checks"])
    assert result["validated_research_debt_ratio"] is None


def test_model_can_select_positive_interior_debt_when_evidence_supports_it():
    # Controlled repeating cycles test the selection mechanics, not forecasting.
    logs = np.log1p(np.tile([.05, -.045, .002], 140))
    result = estimate_drawdown_conditioned_debt(logs, horizon_sessions=3,
        drawdown_lookback_sessions=6, minimum_analogs=4, analog_fraction=1,
        bootstrap_paths=100, grid_step=.25)
    assert result["independent_window_count"] >= result["tail_coverage_floor"]
    assert result["model_selected_debt_ratio"] == .25
    assert result["evidence_status"] == "historical_stability_supported"
    assert result["borrowing_amount_established"] is False


def test_sparse_tail_evidence_selects_explicit_zero_not_unknown():
    result = estimate_drawdown_conditioned_debt([.001]*644)
    assert result["tail_coverage_floor"] == 59
    assert result["model_selected_debt_ratio"] == 0
    assert result["tail_observation_coverage"] < .95
