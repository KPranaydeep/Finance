"""Portfolio-only leverage scenario math; results are diagnostics, not advice."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

LAMF_ADVERTISED_START_RATE = 0.0999
LAMF_RATE_STRESS_SHOCKS = (0.02, 0.04)


def estimate_drawdown_conditioned_debt(
    portfolio_log_returns: Sequence[float],
    *,
    borrowing_rate: float = LAMF_ADVERTISED_START_RATE,
    sessions_per_year: int = 252,
    horizon_sessions: int = 30,
    drawdown_lookback_sessions: int = 252,
    maximum_debt_ratio: float = 1.0,
    grid_step: float = 0.025,
    analog_fraction: float = 0.20,
    minimum_analogs: int = 30,
    minimum_independent_windows: int | None = None,
    bootstrap_paths: int = 5000,
    seed: int = 432,
) -> dict[str, object]:
    """Estimate a drawdown-state-conditioned debt scenario from the saved mix.

    Historical forward windows are matched to the portfolio's latest drawdown
    from its trailing 252-session high. The chosen debt ratio maximizes the empirical lower-tail (5%) net
    log growth over the configured horizon, falling back to zero debt unless it
    improves on the unlevered result. This is a conditional historical scenario,
    not a safety limit or lender-approved amount.
    """
    logs = np.asarray(portfolio_log_returns, dtype=float).reshape(-1)
    # At 95% confidence, seeing even one observation from the worst 5% needs
    # ceil(log(.05)/log(.95)) = 59 independent windows. This is a necessary
    # tail-coverage floor, not a claim of precise 5% quantile estimation.
    tail_coverage_floor = int(math.ceil(math.log(.05) / math.log(.95)))
    if minimum_independent_windows is None:
        minimum_independent_windows = tail_coverage_floor
    rate = float(borrowing_rate)
    required_history = (
        drawdown_lookback_sessions + horizon_sessions + minimum_analogs
    )
    if len(logs) < max(required_history, 60):
        raise ValueError("not enough saved returns for drawdown-conditioned analogs")
    if not np.isfinite(logs).all() or (logs > 50).any() or (logs < -50).any():
        raise ValueError("portfolio returns contain invalid values")
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("borrowing_rate must be finite and nonnegative")
    if sessions_per_year < 1 or horizon_sessions < 1 or drawdown_lookback_sessions < 1:
        raise ValueError("session counts must be positive")
    if not 0 < analog_fraction <= 1 or minimum_analogs < 1:
        raise ValueError("analog selection settings are invalid")
    if maximum_debt_ratio < 0 or grid_step <= 0:
        raise ValueError("debt search settings are invalid")
    if minimum_independent_windows < 3 or bootstrap_paths < 100:
        raise ValueError("evidence settings are invalid")

    wealth = np.exp(np.cumsum(logs))
    drawdowns = np.full(len(wealth), np.nan, dtype=float)
    for index in range(drawdown_lookback_sessions - 1, len(wealth)):
        trailing_peak = float(np.max(wealth[index - drawdown_lookback_sessions + 1 : index + 1]))
        drawdowns[index] = wealth[index] / trailing_peak - 1.0
    current_drawdown = float(drawdowns[-1])
    simple_returns = np.expm1(logs)
    starts = np.arange(
        drawdown_lookback_sessions - 1,
        len(logs) - horizon_sessions,
    )
    if len(starts) < minimum_analogs:
        raise ValueError("not enough forward windows for drawdown-conditioned analogs")
    distances = np.abs(drawdowns[starts] - current_drawdown)
    requested = max(minimum_analogs, int(math.ceil(len(starts) * analog_fraction)))
    analog_count = min(len(starts), requested)
    analog_starts = starts[np.argsort(distances, kind="stable")[:analog_count]]
    # The drawdown at t already includes return t. Forward outcomes start at
    # t+1, and the last matched observation must remain inside saved history.
    matched = simple_returns[analog_starts[:, None] + 1 + np.arange(horizon_sessions)]

    levels = np.arange(0.0, maximum_debt_ratio + grid_step / 2.0, grid_step)
    levels = np.unique(np.append(levels, maximum_debt_ratio))
    tail_growth: list[float] = []
    for debt_ratio in levels:
        gross = 1.0 + float(debt_ratio)
        net = gross * matched - float(debt_ratio) * rate / sessions_per_year
        valid = (net > -1.0).all(axis=1) & np.isfinite(net).all(axis=1)
        path_growth = np.full(len(net), -np.inf, dtype=float)
        path_growth[valid] = np.log1p(net[valid]).sum(axis=1)
        tail_growth.append(float(np.quantile(path_growth, 0.05)))

    candidate_rows = [
        {
            "debt_ratio": float(level),
            "gross_exposure": 1.0 + float(level),
            "lower_tail_log_growth": float(score),
            "lower_tail_30_session_return": float(np.expm1(score)),
            "change_vs_unlevered": float(score - tail_growth[0]),
        }
        for level, score in zip(levels, tail_growth)
    ]
    best_index = int(np.argmax(tail_growth))
    baseline = float(tail_growth[0])
    if tail_growth[best_index] <= baseline + 1e-10:
        best_index = 0
    raw_best_index = best_index
    raw_debt_ratio = float(levels[raw_best_index])
    at_boundary = bool(raw_best_index == len(levels) - 1 and raw_debt_ratio > 0)
    # Remove shared return observations before any uncertainty claim. Choose
    # closest states first, then sort them chronologically for stability tests.
    independent_starts = []
    for start in analog_starts:
        if all(abs(int(start) - prior) >= horizon_sessions for prior in independent_starts):
            independent_starts.append(int(start))
    independent_starts.sort()
    independent = simple_returns[np.array(independent_starts)[:, None] + 1 + np.arange(horizon_sessions)]
    evidence_status = "insufficient_independent_windows"
    validated_ratio = None
    bounds = []
    if len(independent_starts) >= minimum_independent_windows:
        rng = np.random.default_rng(seed)
        slices = np.array_split(np.arange(len(independent)), 3)
        resamples = []
        for segment in slices:
            count = len(segment)
            block = max(2, int(math.ceil(count ** (1/3))))
            starts_b = rng.integers(count, size=(bootstrap_paths, math.ceil(count / block)))
            resamples.append(((starts_b[..., None] + np.arange(block)) % count).reshape(bootstrap_paths, -1)[:, :count])
        alpha = .05 / max(1, (len(levels)-1) * 3 * 3)
        for index, ratio in enumerate(levels):
            if ratio <= 0:
                continue
            lower_bounds = []
            for funding_rate in (rate, rate + .02, rate + .04):
                net = (1 + ratio) * independent - ratio * funding_rate / sessions_per_year
                if np.any(net <= -1):
                    lower_bounds.append(float("-inf"))
                    continue
                outcomes = np.log1p(net).sum(axis=1)
                baseline_outcomes = np.log1p(independent).sum(axis=1)
                for segment, sample in zip(slices, resamples):
                    paired_gain = np.quantile(outcomes[segment][sample], .05, axis=1) - np.quantile(baseline_outcomes[segment][sample], .05, axis=1)
                    lower_bounds.append(float(np.quantile(paired_gain, alpha)))
            bounds.append({"debt_ratio": float(ratio), "worst_adjusted_lower_bound": min(lower_bounds)})
        eligible = [row for row in bounds if row["worst_adjusted_lower_bound"] > 1e-10 and row["debt_ratio"] < maximum_debt_ratio - 1e-10]
        if raw_debt_ratio == 0:
            evidence_status = "no_historical_advantage"
        elif at_boundary:
            evidence_status = "search_boundary_unresolved"
        elif eligible:
            # Optimize the stressed uncertainty-adjusted improvement, choosing
            # lower debt when objectives tie. Zero is always a competing option.
            validated_ratio = max(eligible, key=lambda row: (row["worst_adjusted_lower_bound"], -row["debt_ratio"]))["debt_ratio"]
            evidence_status = "historical_stability_supported"
        else:
            evidence_status = "uncertain_or_rate_stress_failed"
    # Compatibility field is a screened research ratio, never lender capacity.
    debt_ratio = float(validated_ratio or 0.0)
    best_index = int(np.argmin(abs(levels - debt_ratio)))
    return {
        "method": "nearest historical portfolio-drawdown states; 30-session 5% net log-growth maximum",
        "current_drawdown": current_drawdown,
        "analog_count": int(analog_count),
        "analog_fraction": float(analog_count / len(starts)),
        "horizon_sessions": int(horizon_sessions),
        "drawdown_lookback_sessions": int(drawdown_lookback_sessions),
        "borrowing_rate": rate,
        "recommended_debt_ratio_of_portfolio_base": debt_ratio,
        "recommended_gross_exposure": 1.0 + debt_ratio,
        "estimated_debt_amount": None,
        "schema_version": 2,
        "raw_historical_optimum_debt_ratio": raw_debt_ratio,
        "validated_research_debt_ratio": validated_ratio,
        "borrowing_amount_established": False,
        "evidence_status": evidence_status,
        "independent_window_count": len(independent_starts),
        "minimum_independent_windows": minimum_independent_windows,
        "tail_coverage_floor": tail_coverage_floor,
        "tail_observation_coverage": float(1 - .95 ** len(independent_starts)),
        "model_selected_debt_ratio": debt_ratio,
        "model_decision_reason": (
            "Positive debt maximizes the stressed, uncertainty-adjusted historical objective"
            if debt_ratio > 0 else
            "Zero additional debt selected: positive debt did not pass the tail-evidence, stability, funding-stress and interior-optimum requirements"
        ),
        "independent_state_indices": independent_starts,
        "forward_windows_start_after_state": True,
        "uncertainty_checks": bounds,
        "lower_tail_log_growth": float(tail_growth[best_index]),
        "unlevered_lower_tail_log_growth": baseline,
        "improvement_vs_unlevered": float(tail_growth[best_index] - baseline),
        "candidate_rows": candidate_rows,
        "at_search_boundary": at_boundary,
        "maximum_tested_debt_ratio": float(maximum_debt_ratio),
        "warning": (
            "Historical analogs are overlapping and use the saved full-history optimized mix; "
            "this is not a safe borrowing limit, forecast, or lender capacity."
        ),
    }


def assess_leverage_scenarios(
    portfolio_log_returns: Sequence[float],
    *,
    borrowing_rate: float = LAMF_ADVERTISED_START_RATE,
    sessions_per_year: int = 252,
    leverage_levels: Sequence[float] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0),
    fold_count: int = 3,
    tail_confidence: float = 0.95,
    stress_horizon_sessions: int = 30,
) -> dict[str, object]:
    """Calculate financing-adjusted fixed-mix scenarios across chronological folds.

    The daily mix is rebalanced to each gross-exposure level for arithmetic.
    Chronological folds are diagnostic only: saved portfolio weights were
    selected using the full historical sample, so these are not true out-of-
    sample optimizer tests and cannot establish a safe borrowing limit.
    """
    try:
        logs = np.asarray(portfolio_log_returns, dtype=float).reshape(-1)
        rate = float(borrowing_rate)
        levels = sorted({float(level) for level in leverage_levels})
    except (TypeError, ValueError) as exc:
        raise ValueError("returns, rate, and leverage levels must be numeric") from exc
    if logs.size < max(fold_count * stress_horizon_sessions, 60):
        raise ValueError("not enough daily portfolio observations for leverage scenarios")
    if not np.isfinite(logs).all() or (logs > 50).any() or (logs < -50).any():
        raise ValueError("portfolio returns contain invalid values")
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("borrowing_rate must be finite and nonnegative")
    if sessions_per_year <= 0 or fold_count < 2 or stress_horizon_sessions <= 0:
        raise ValueError("session, fold, and horizon counts must be positive")
    if not 0 < tail_confidence < 1:
        raise ValueError("tail_confidence must be between zero and one")
    if not levels or levels[0] < 1 or not all(math.isfinite(x) for x in levels):
        raise ValueError("leverage levels must be finite and at least 1x")

    asset_returns = np.expm1(logs)
    boundaries = np.linspace(0, len(asset_returns), fold_count + 1, dtype=int)
    fold_slices = [
        asset_returns[boundaries[i] : boundaries[i + 1]] for i in range(fold_count)
    ]
    if any(len(fold) < stress_horizon_sessions for fold in fold_slices):
        raise ValueError("each chronological segment must cover the stress horizon")

    daily_rows: list[dict[str, float | int | bool]] = []
    for multiple in levels:
        debt_ratio = multiple - 1.0
        daily_net = multiple * asset_returns - debt_ratio * rate / sessions_per_year
        solvent = bool(np.isfinite(daily_net).all() and (daily_net > -1.0).all())
        if not solvent:
            daily_rows.append(
                {
                    "leverage_multiple": multiple,
                    "debt_ratio": debt_ratio,
                    "solvent_in_history": False,
                }
            )
            continue

        log_net = np.log1p(daily_net)
        annualized_net = float(np.expm1(log_net.mean() * sessions_per_year))
        fold_returns = [
            float(
                np.expm1(
                    np.log1p(
                        multiple * fold - debt_ratio * rate / sessions_per_year
                    ).mean()
                    * sessions_per_year
                )
            )
            if np.all(1.0 + multiple * fold - debt_ratio * rate / sessions_per_year > 0)
            else -1.0
            for fold in fold_slices
        ]
        horizon_returns = np.convolve(
            np.log1p(daily_net),
            np.ones(stress_horizon_sessions),
            mode="valid",
        )
        horizon_simple = np.expm1(horizon_returns)
        tail_count = max(1, int(math.ceil((1.0 - tail_confidence) * len(horizon_simple))))
        tail_mean_return = float(np.partition(horizon_simple, tail_count - 1)[:tail_count].mean())
        wealth = np.exp(np.concatenate(([0.0], np.cumsum(log_net))))
        running_peak = np.maximum.accumulate(wealth)
        max_drawdown = float(np.max(1.0 - wealth / running_peak))
        daily_rows.append(
            {
                "leverage_multiple": multiple,
                "debt_ratio": debt_ratio,
                "solvent_in_history": True,
                "annualized_net_return": annualized_net,
                "fold_net_returns": fold_returns,
                "positive_folds": sum(value > 0 for value in fold_returns),
                "folds_beating_unlevered": 0,
                "tail_mean_return": tail_mean_return,
                "tail_expected_shortfall": max(0.0, -tail_mean_return),
                "historical_max_drawdown": max_drawdown,
                "annual_interest_per_100k": debt_ratio * rate * 100_000.0,
            }
        )

    unlevered = next((row for row in daily_rows if row["leverage_multiple"] == 1.0 and row["solvent_in_history"]), None)
    if unlevered:
        base_folds = unlevered["fold_net_returns"]
        for row in daily_rows:
            if row["solvent_in_history"]:
                row["folds_beating_unlevered"] = sum(
                    float(value) > float(base)
                    for value, base in zip(row["fold_net_returns"], base_folds)
                )

    return {
        "method": "fixed-mix chronological scenarios with daily financing drag",
        "observations": int(len(logs)),
        "fold_count": int(fold_count),
        "borrowing_rate": rate,
        "sessions_per_year": int(sessions_per_year),
        "stress_horizon_sessions": int(stress_horizon_sessions),
        "tail_confidence": float(tail_confidence),
        "rows": daily_rows,
        "selection_bias_warning": (
            "Chronological segments use the saved optimal weights selected on the full history; "
            "they are not independent out-of-sample optimizer validation."
        ),
    }


def estimate_growth_optimal_debt(
    portfolio_log_returns: Sequence[float],
    *,
    borrowing_rate: float = LAMF_ADVERTISED_START_RATE,
    sessions_per_year: int = 252,
    fold_count: int = 3,
    bootstrap_paths: int = 2000,
    block_sessions: int = 5,
    tail_confidence: float = 0.95,
    maximum_debt_ratio: float = 1.0,
    grid_step: float = 0.025,
    random_seed: int = 20261014,
) -> dict[str, object]:
    """Estimate a conservative growth-optimal debt ratio via blocked resampling.

    For each of three chronological segments, resample 252-session paths from
    five-session contiguous blocks, apply constant gross exposure and financing
    drag, then maximize the worst segment's lower-tail annual log growth. This
    supplies a portfolio-only model amount without a user loss-budget input.
    It is not a safe amount: saved weights were selected on the full history,
    bootstraps cannot model regime breaks, and lender collateral capacity is
    independent and unavailable here.
    """
    try:
        logs = np.asarray(portfolio_log_returns, dtype=float).reshape(-1)
        rate = float(borrowing_rate)
        max_debt = float(maximum_debt_ratio)
        step = float(grid_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("returns, rate, and debt-search settings must be numeric") from exc
    if not np.isfinite(logs).all() or (logs > 50).any() or (logs < -50).any():
        raise ValueError("portfolio returns contain invalid values")
    if len(logs) < max(180, fold_count * block_sessions * 8):
        raise ValueError("not enough daily returns for a stable leverage estimate")
    if rate < 0 or not math.isfinite(rate):
        raise ValueError("borrowing_rate must be finite and nonnegative")
    if sessions_per_year < 1 or fold_count < 2 or bootstrap_paths < 500:
        raise ValueError("simulation and session counts are invalid")
    if block_sessions < 1 or step <= 0 or max_debt < 0:
        raise ValueError("block, grid, and debt limits are invalid")
    if not 0 < tail_confidence < 1:
        raise ValueError("tail_confidence must be between zero and one")

    boundaries = np.linspace(0, len(logs), fold_count + 1, dtype=int)
    folds = [logs[boundaries[i] : boundaries[i + 1]] for i in range(fold_count)]
    if any(len(fold) < block_sessions * 8 for fold in folds):
        raise ValueError("each chronological segment needs at least eight sample blocks")

    horizon = int(sessions_per_year)
    blocks_per_path = int(math.ceil(horizon / block_sessions))
    offsets = np.arange(block_sessions, dtype=int)
    levels = np.arange(0.0, max_debt + step / 2.0, step, dtype=float)
    levels = np.unique(np.append(levels, max_debt))
    lower_tail_scores: list[list[float]] = [[] for _ in levels]

    for fold_index, fold in enumerate(folds):
        rng = np.random.default_rng(random_seed + fold_index)
        maximum_start = len(fold) - block_sessions
        starts = rng.integers(
            0,
            maximum_start + 1,
            size=(bootstrap_paths, blocks_per_path),
        )
        sampled_logs = fold[starts[..., None] + offsets]
        sampled_logs = sampled_logs.reshape(bootstrap_paths, -1)[:, :horizon]
        sampled_returns = np.expm1(sampled_logs)
        for level_index, debt_ratio in enumerate(levels):
            gross_exposure = 1.0 + float(debt_ratio)
            net_daily = (
                gross_exposure * sampled_returns
                - float(debt_ratio) * rate / sessions_per_year
            )
            solvent = (net_daily > -1.0).all(axis=1) & np.isfinite(net_daily).all(axis=1)
            annual_log_growth = np.full(bootstrap_paths, -np.inf, dtype=float)
            if solvent.any():
                annual_log_growth[solvent] = np.log1p(net_daily[solvent]).sum(axis=1)
            lower_tail_scores[level_index].append(
                float(np.quantile(annual_log_growth, 1.0 - tail_confidence))
            )

    robust_scores = np.asarray(
        [min(fold_scores) for fold_scores in lower_tail_scores],
        dtype=float,
    )
    best_index = int(np.argmax(robust_scores))
    baseline_index = int(np.argmin(np.abs(levels)))
    best_debt_ratio = float(levels[best_index])
    baseline_score = float(robust_scores[baseline_index])
    best_score = float(robust_scores[best_index])
    # If resampling uncertainty leaves no modeled growth improvement, don't
    # turn an unstable numerical tie into a positive borrowing suggestion.
    if best_score <= baseline_score + 1e-10:
        best_debt_ratio = 0.0
        best_index = baseline_index
        best_score = baseline_score

    fold_scores = [float(value) for value in lower_tail_scores[best_index]]
    return {
        "method": "blocked-bootstrap worst-segment 95% lower-tail log-growth maximum",
        "borrowing_rate": rate,
        "recommended_debt_ratio_of_portfolio_base": best_debt_ratio,
        "recommended_gross_exposure": 1.0 + best_debt_ratio,
        "robust_annual_log_growth_at_model_amount": best_score,
        "unlevered_robust_annual_log_growth": baseline_score,
        "improvement_vs_unlevered": best_score - baseline_score,
        "segment_lower_tail_log_growth": fold_scores,
        "segment_count": int(fold_count),
        "bootstrap_paths_per_segment": int(bootstrap_paths),
        "block_sessions": int(block_sessions),
        "maximum_tested_debt_ratio": max_debt,
        "at_search_boundary": bool(best_index == len(levels) - 1 and best_debt_ratio > 0),
        "tail_confidence": float(tail_confidence),
        "warning": (
            "Model-implied scenario only; not a personal risk limit, safe amount, or lender limit. "
            "The selected weights use full-history data, and bootstrap paths cannot guarantee future outcomes."
        ),
    }


def build_portfolio_leverage_assessment(
    portfolio_log_returns: Sequence[float],
    *,
    borrowing_rates: Sequence[float] | None = None,
    invested_base: float | None = None,
) -> dict[str, object]:
    """Build the saved sensitivity frontier and portfolio-only model amount."""
    rates = borrowing_rates or (
        LAMF_ADVERTISED_START_RATE,
        LAMF_ADVERTISED_START_RATE + LAMF_RATE_STRESS_SHOCKS[0],
        LAMF_ADVERTISED_START_RATE + LAMF_RATE_STRESS_SHOCKS[1],
    )
    scenarios = []
    for rate in rates:
        frontier = assess_leverage_scenarios(
            portfolio_log_returns,
            borrowing_rate=rate,
        )
        growth_optimum = estimate_growth_optimal_debt(
            portfolio_log_returns,
            borrowing_rate=rate,
        )
        debt_ratio = float(growth_optimum["recommended_debt_ratio_of_portfolio_base"])
        growth_optimum["estimated_debt_amount"] = (
            float(invested_base) * debt_ratio
            if invested_base is not None and math.isfinite(float(invested_base))
            else None
        )
        drawdown_scenario = estimate_drawdown_conditioned_debt(
            portfolio_log_returns,
            borrowing_rate=rate,
        )
        drawdown_debt_ratio = float(
            drawdown_scenario["recommended_debt_ratio_of_portfolio_base"]
        )
        # No mutual-fund collateral/lender capacity exists in this dataset.
        drawdown_scenario["estimated_debt_amount"] = None
        scenarios.append(
            {
                "rate": float(rate),
                "frontier": frontier["rows"],
                "growth_optimum": growth_optimum,
                "drawdown_conditioned": drawdown_scenario,
                "observations": frontier["observations"],
                "selection_bias_warning": frontier["selection_bias_warning"],
            }
        )
    return {
        "schema_version": 1,
        "method": "portfolio blocked-bootstrap growth scenarios v1",
        "scenarios": scenarios,
    }
