"""Historical block-resampling scenarios for executable rebalancing buys.

The output is a descriptive scenario distribution, not a validated predictive
model. Contiguous blocks preserve some local dependence in observed returns;
they do not make historical outcomes a promise about future returns.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


METHOD = "eb-shrunk-stationary-block-bootstrap-v1"


def _long_run_mean_variance(values: np.ndarray) -> float:
    """Newey-West estimate of sampling variance for a dependent-series mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("inf")
    centered = values - float(np.mean(values))
    max_lag = min(n - 1, max(1, int(math.ceil(n ** (1.0 / 3.0)))))
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        bartlett_weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    return max(long_run_variance, 0.0) / n


def _stationary_bootstrap_terminal_log_returns(
    values: np.ndarray,
    *,
    horizon: int,
    paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample terminal log returns using random-length contiguous blocks."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=float)

    # Cube-root block length is a simple, data-size-based starting point for
    # dependent-series resampling; disclose the method rather than tuning it
    # to maximize a particular security's result.
    mean_block_length = max(2, int(math.ceil(n ** (1.0 / 3.0))))
    indices = rng.integers(0, n, size=paths)
    totals = np.zeros(paths, dtype=float)

    for _ in range(horizon):
        totals += values[indices]
        restart = rng.random(paths) < (1.0 / mean_block_length)
        following = (indices + 1) % n
        indices = np.where(restart, rng.integers(0, n, size=paths), following)

    return totals


def forecast_executable_buys(
    rebalance_plan: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    horizon_sessions: int,
    one_way_cost_bps: float = 22.0,
    paths: int = 4000,
    seed: int = 20261014,
) -> pd.DataFrame:
    """Rank executable buys using historical block-bootstrap scenarios.

    Expected return is measured across the resampled paths. The ranking score
    is expected net return multiplied by positive-path frequency, as requested;
    optimized weight is display-only and does not affect single-buy ranking.
    Historical drift is empirically-Bayes shrunk toward the
    cross-section median; residual paths use stationary block bootstrap.
    Cost is applied on entry and exit as a configurable proportional estimate.
    """
    columns = [
        "Security",
        "Yahoo Ticker",
        "Optimized Weight (%)",
        "Expected net scenario return (%)",
        "Probability of positive net return (%)",
        "Return × positive-probability score (%)",
        "Median net return (%)",
        "10th–90th percentile net return (%)",
        "History sessions",
        "Scenario method",
    ]
    if rebalance_plan is None or rebalance_plan.empty or log_returns is None:
        return pd.DataFrame(columns=columns)
    if int(horizon_sessions) < 1:
        raise ValueError("Scenario horizon must be at least one trading session.")
    if int(paths) < 100:
        raise ValueError("At least 100 scenario paths are required.")
    if not math.isfinite(float(one_way_cost_bps)) or one_way_cost_bps < 0:
        raise ValueError("One-way trading cost must be a finite non-negative value.")

    buys = rebalance_plan.loc[
        rebalance_plan["Action"].astype(str).str.casefold().eq("buy")
        & pd.to_numeric(rebalance_plan["Executable Quantity"], errors="coerce").gt(0)
    ]
    if buys.empty:
        return pd.DataFrame(columns=columns)

    cross_section_means = pd.to_numeric(log_returns.mean(axis=0), errors="coerce")
    cross_section_means = cross_section_means[np.isfinite(cross_section_means)]
    prior_mean = float(cross_section_means.median()) if len(cross_section_means) else 0.0
    if len(cross_section_means) > 1:
        robust_scale = 1.4826 * float(np.median(np.abs(cross_section_means - prior_mean)))
        candidate_mean_variances = []
        for ticker in buys["Yahoo Ticker"].astype(str):
            if ticker in log_returns.columns:
                candidate_history = pd.to_numeric(
                    log_returns[ticker], errors="coerce"
                ).to_numpy(dtype=float)
                candidate_history = candidate_history[np.isfinite(candidate_history)]
                if len(candidate_history) >= 2:
                    candidate_mean_variances.append(
                        _long_run_mean_variance(candidate_history)
                    )
        typical_sampling_variance = (
            float(np.median(candidate_mean_variances))
            if candidate_mean_variances
            else 0.0
        )
        prior_variance = max(robust_scale**2 - typical_sampling_variance, 0.0)
    else:
        prior_variance = 0.0

    cost_per_side = float(one_way_cost_bps) / 10_000.0
    if cost_per_side >= 1.0:
        raise ValueError("One-way cost must be less than 10,000 basis points.")
    round_trip_multiplier = (1.0 - cost_per_side) ** 2
    rng = np.random.default_rng(int(seed))
    rows = []

    for _, row in buys.iterrows():
        ticker = str(row.get("Yahoo Ticker", "")).strip()
        if not ticker or ticker not in log_returns.columns:
            continue
        history = pd.to_numeric(log_returns[ticker], errors="coerce").to_numpy(dtype=float)
        history = history[np.isfinite(history)]
        if len(history) < 2:
            continue

        observed_mean = float(np.mean(history))
        observation_variance = _long_run_mean_variance(history)
        posterior_weight = (
            prior_variance / (prior_variance + observation_variance)
            if prior_variance > 0 and math.isfinite(observation_variance)
            else 0.0
        )
        shrunk_mean = prior_mean + posterior_weight * (observed_mean - prior_mean)
        residual_history = history - observed_mean

        sampled_log_returns = _stationary_bootstrap_terminal_log_returns(
            residual_history,
            horizon=int(horizon_sessions),
            paths=int(paths),
            rng=rng,
        )
        sampled_log_returns += shrunk_mean * int(horizon_sessions)
        # Clip only at the floating-point boundary; expm1 can otherwise
        # overflow on pathological historical input and poison the whole table.
        gross_log = np.clip(sampled_log_returns, -700.0, 700.0)
        net_returns = np.exp(gross_log) * round_trip_multiplier - 1.0
        net_returns = net_returns[np.isfinite(net_returns)]
        if not len(net_returns):
            continue

        q10, median, q90 = np.quantile(net_returns, [0.10, 0.50, 0.90])
        expected = float(np.mean(net_returns))
        positive_probability = float(np.mean(net_returns > 0.0))
        rows.append({
            "Security": str(row.get("Symbol", ticker)),
            "Yahoo Ticker": ticker,
            "Optimized Weight (%)": float(row.get("Optimal Weight", 0.0)) * 100.0,
            "Expected net scenario return (%)": expected * 100.0,
            "Probability of positive net return (%)": positive_probability * 100.0,
            "Return × positive-probability score (%)": (
                expected * 100.0 * positive_probability
            ),
            "Median net return (%)": float(median) * 100.0,
            "10th–90th percentile net return (%)": f"{q10 * 100.0:.1f}% to {q90 * 100.0:.1f}%",
            "History sessions": int(len(history)),
            "Scenario method": METHOD,
        })

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return score_scenario_ranking(result)


def score_scenario_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the requested return × positive-path probability ranking.

    Weight remains display-only. The product is a user-facing heuristic, not
    expected value: expected return already includes negative as well as
    positive paths, so multiplying by P(positive) penalizes probability twice.
    """
    result = frame.copy()
    if result.empty:
        return result
    expected = pd.to_numeric(
        result.get("Expected net scenario return (%)"), errors="coerce"
    )
    probability = pd.to_numeric(
        result.get("Probability of positive net return (%)"), errors="coerce"
    )
    result["Return × positive-probability score (%)"] = expected * probability / 100.0
    return result.sort_values(
        ["Return × positive-probability score (%)", "Expected net scenario return (%)",
         "Probability of positive net return (%)"],
        ascending=[False, False, False],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def normalize_saved_scenario_ranking(records) -> pd.DataFrame:
    """Restore and rank saved records, including records from older app versions."""
    frame = pd.DataFrame(records or [])
    if frame.empty:
        return frame
    return score_scenario_ranking(frame)
