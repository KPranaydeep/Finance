"""Causal recent-risk and balanced trend/reversal feedback, with a slow anchor.

Forecast losses use each day's BEFORE-observation predictions. Full historical
returns remain untouched. Fixed numerical conventions are versioned here, rather
than searched against the final portfolio's fitted return.
"""
import numpy as np
import pandas as pd

METHOD = "recent13-anchored-feedback-v1"
RECENT = 13
WARMUP = 26
FLOOR = 1e-12


def adaptive_state(log_returns):
    frame = pd.DataFrame(log_returns)
    r = np.ascontiguousarray(frame.to_numpy(dtype=float))
    if (len(r) < WARMUP + 1 or not r.shape[1] or not np.isfinite(r).all()
            or frame.index.duplicated().any() or not frame.index.is_monotonic_increasing):
        raise ValueError("Complete chronological history required for adaptive feedback")
    # All paths start from the same deterministic prior, so restart/resume and
    # historical training prefixes cannot inherit a future or another user's state.
    weights = np.arange(1, RECENT + 1, dtype=float)
    weights /= weights.sum()
    effective_n = 1 / np.sum(weights**2)
    signal_shrinkage = effective_n / (effective_n + RECENT)
    decay = np.exp(-1 / RECENT)
    prefix = np.vstack([np.zeros(r.shape[1]), np.cumsum(r, axis=0)])
    squares = np.vstack([np.zeros(r.shape[1]), np.cumsum(r*r, axis=0)])
    mean_losses = np.zeros((3, r.shape[1]))
    variance_loss_gap = np.zeros(r.shape[1])
    released_factor = np.ones(r.shape[1])
    feedback_days = 0
    for t in range(WARMUP, len(r) + 1):
        slow_mean = prefix[t] / t
        slow_var = np.maximum(squares[t] / t - slow_mean**2, FLOOR)
        recent = r[t-RECENT:t]
        recent_mean = weights @ recent
        # Innovation risk relative to the slow anchor also detects a persistent
        # low-volatility decline/rally; a new mean must not make risk disappear.
        recent_var = np.maximum(weights @ (recent - slow_mean)**2, FLOOR)
        signal = (recent_mean - slow_mean) * signal_shrinkage
        forecasts = np.vstack([slow_mean, slow_mean + signal, slow_mean - signal])
        # Soft expert weights; retain at least half on the slow forecast. Trend
        # and reversal have equal priors and no hard-coded winning direction.
        evidence = -mean_losses
        evidence -= evidence.max(axis=0)
        expert = np.exp(evidence)
        expert /= expert.sum(axis=0)
        shares = .5 * expert
        shares[0] += .5
        mean_shift = (shares[1] - shares[2]) * signal
        fast_share = np.minimum(.5, np.exp(-np.logaddexp(0., variance_loss_gap)))
        variance = (1-fast_share) * slow_var + fast_share * recent_var
        raw_factor = np.maximum(1., np.sqrt(variance / slow_var))
        # Tighten immediately; release old stress gradually over the same
        # 13-session response scale. Never create an exposure cap above baseline.
        released_factor = np.maximum(raw_factor, 1 + decay*(released_factor-1))
        if t == len(r):
            break
        observed = r[t]
        # Scale-free log scoring softens a single extreme error rather than
        # permanently deleting an expert. Observed returns are NEVER clipped.
        normalized_loss = np.log1p((observed - forecasts)**2 / slow_var)
        mean_losses = decay*mean_losses + (1-decay)*normalized_loss
        residual2 = (observed - slow_mean)**2
        qlike_gap = np.log(recent_var/slow_var) + residual2*(1/recent_var - 1/slow_var)
        variance_loss_gap = decay*variance_loss_gap + (1-decay)*np.arcsinh(qlike_gap)
        feedback_days += 1
    if not np.isfinite(released_factor).all() or not np.isfinite(mean_shift).all():
        raise ValueError("Nonfinite adaptive feedback")
    return {"factors": released_factor, "mean_shift": mean_shift, "long_mean": slow_mean,
            "recent_covariance": (recent-recent_mean).T @ ((recent-recent_mean)*weights[:, None]),
            "covariance_blend": float(np.mean(fast_share)),
            "summary": {"method": METHOD, "recent_sessions": RECENT,
                "recent_effective_observations": round(float(effective_n), 2),
                "feedback_observations": feedback_days, "history_sessions": len(r),
                "history_through": str(frame.index[-1])[:10],
                "trend_share": float(np.mean(shares[1])),
                "reversal_share": float(np.mean(shares[2])),
                "anchor_share": float(np.mean(shares[0])),
                "stressed_assets": int(np.sum(released_factor > 1.000001)),
                "median_stress_factor": float(np.median(released_factor)),
                "maximum_stress_factor": float(np.max(released_factor))}}


def scaled_risk_scenarios(raw_scenarios, state, horizon):
    """Upward-only volatility scaling; the separate raw-history guard is retained."""
    raw = np.asarray(raw_scenarios, dtype=float)
    ruined = raw <= -1
    center = np.asarray(state["long_mean"], dtype=float) * horizon
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.expm1((np.log1p(np.where(ruined, 0., raw))-center)*state["factors"]+center)
    scaled[ruined] = -1.
    if not np.isfinite(scaled).all():
        raise ValueError("Adaptive risk scenarios overflowed; risk was not clipped")
    return scaled


def conditional_objective_means(raw_scenarios, state, horizon):
    # Use raw historical means plus the earned trend/reversal adjustment; never
    # reward volatility scaling for mechanically raising arithmetic means.
    raw_mean = np.mean(raw_scenarios, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.expm1(np.log1p(raw_mean) + horizon*state["mean_shift"])
    if not np.isfinite(means).all():
        raise ValueError("Nonfinite adaptive return forecast")
    return means
