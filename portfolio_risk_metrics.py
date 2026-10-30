"""Tail-risk measures used by the portfolio rebalancer."""

from __future__ import annotations

import math

import numpy as np


def empirical_expected_shortfall(
    simple_returns,
    confidence: float = 0.95,
) -> float:
    """Return the mean non-negative loss in the worst confidence tail."""
    values = np.asarray(simple_returns, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    losses = -values
    threshold = float(np.quantile(losses, confidence, method="higher"))
    tail_losses = losses[losses >= threshold]
    if tail_losses.size == 0:
        return 0.0
    return max(float(tail_losses.mean()), 0.0)


def moving_block_bootstrap_expected_shortfall(
    daily_log_returns,
    *,
    horizon_sessions: int = 30,
    confidence: float = 0.95,
    block_sessions: int = 5,
    simulation_paths: int = 10_000,
    random_seed: int = 20_261_004,
) -> float:
    """Estimate horizon expected shortfall with deterministic moving blocks.

    Sampling short contiguous blocks retains some observed serial dependence,
    while joining independently selected blocks creates enough horizon scenarios
    for a stable empirical tail estimate.
    """
    scenarios = moving_block_bootstrap_return_distribution(
        daily_log_returns,
        horizon_sessions=horizon_sessions,
        block_sessions=block_sessions,
        simulation_paths=simulation_paths,
        random_seed=random_seed,
    )
    if scenarios.size == 0:
        return float("nan")
    return empirical_expected_shortfall(scenarios, confidence=confidence)


def moving_block_bootstrap_return_distribution(
    daily_log_returns,
    *,
    horizon_sessions: int = 30,
    block_sessions: int = 5,
    simulation_paths: int = 10_000,
    random_seed: int = 20_261_004,
) -> np.ndarray:
    """Return deterministic simple-return paths from a moving-block bootstrap."""
    values = np.asarray(daily_log_returns, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    horizon = int(horizon_sessions)
    block = int(block_sessions)
    paths = int(simulation_paths)
    if horizon < 1 or block < 1 or paths < 100:
        raise ValueError("bootstrap horizon, block and path counts are invalid")
    if values.size < block:
        return np.asarray([], dtype=float)

    blocks_per_path = int(math.ceil(horizon / block))
    maximum_start = values.size - block
    rng = np.random.default_rng(int(random_seed))
    starts = rng.integers(
        0,
        maximum_start + 1,
        size=(paths, blocks_per_path),
    )
    offsets = np.arange(block, dtype=int)
    sampled = values[starts[..., None] + offsets]
    horizon_log_returns = sampled.reshape(paths, -1)[:, :horizon].sum(axis=1)
    return np.expm1(horizon_log_returns)


def bootstrap_tail_summary(
    simple_return_scenarios,
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Summarize the central return and dispersion of the worst loss tail."""
    values = np.asarray(simple_return_scenarios, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "median_return": float("nan"),
            "expected_shortfall": float("nan"),
            "tail_loss_sd": float("nan"),
        }
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    losses = -values
    threshold = float(np.quantile(losses, confidence, method="higher"))
    tail_losses = np.maximum(losses[losses >= threshold], 0.0)
    tail_sd = (
        float(np.std(tail_losses, ddof=1))
        if tail_losses.size > 1
        else 0.0
    )
    return {
        "median_return": float(np.median(values)),
        "expected_shortfall": max(float(tail_losses.mean()), 0.0),
        "tail_loss_sd": max(tail_sd, 0.0),
    }
