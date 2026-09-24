import numpy as np
import pandas as pd
import pytest

from robust_momentum_filter import apply_robust_momentum_filter


def config(maximum=0.20):
    return {
        "enabled": True,
        "method_version": "robust-momentum-v1",
        "maximum_exclusion_fraction": maximum,
        "lookback_sessions": [63, 126, 252],
        "skip_recent_sessions": 21,
        "stability_checkpoint_sessions": [0, 21, 42],
        "trend_lookback_sessions": 200,
        "minimum_negative_horizons": 2,
        "annualization_sessions": 250,
        "volatility_floor_annual": 0.05,
        "apply_exclusion_to_owned_holdings": False,
    }


def histories():
    index = pd.bdate_range("2024-01-01", periods=340)
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "STRONG": 100.0 * np.exp(0.0020 * x),
            "GOOD": 100.0 * np.exp(0.0013 * x),
            "FLAT": 100.0 * np.exp(0.0001 * x),
            "WEAK1": 200.0 * np.exp(-0.0010 * x),
            "WEAK2": 220.0 * np.exp(-0.0015 * x),
            "OWNED_WEAK": 240.0 * np.exp(-0.0020 * x),
        },
        index=index,
    )


def test_excludes_no_more_than_twenty_percent_of_candidates_and_protects_owned():
    prices = histories()
    kept, report = apply_robust_momentum_filter(
        prices,
        owned_tickers=("OWNED_WEAK",),
        config=config(),
    )

    excluded = report.loc[report["Excluded"], "Ticker"].tolist()
    assert len(excluded) == 1  # floor(5 zero-quantity candidates * 20%)
    assert excluded[0] in {"WEAK1", "WEAK2"}
    assert "OWNED_WEAK" in kept
    owned_row = report.set_index("Ticker").loc["OWNED_WEAK"]
    assert not bool(owned_row["Excluded"])
    assert owned_row["Reason"] == "OWNED_HOLDING_PROTECTED"


def test_does_not_force_an_exclusion_when_absolute_momentum_is_not_weak():
    index = pd.bdate_range("2024-01-01", periods=340)
    x = np.arange(len(index), dtype=float)
    prices = pd.DataFrame(
        {f"UP{number}": 100.0 * np.exp((0.0005 + number * 0.0001) * x) for number in range(5)},
        index=index,
    )
    kept, report = apply_robust_momentum_filter(prices, config=config())
    assert kept == list(prices.columns)
    assert not report["Excluded"].any()


def test_small_universe_never_rounds_up_past_twenty_percent():
    prices = histories().iloc[:, :4]
    kept, report = apply_robust_momentum_filter(prices, config=config())
    assert kept == list(prices.columns)
    assert not report["Excluded"].any()


def test_does_not_backfill_past_bottom_quintile_when_weakest_fails_confirmation():
    prices = histories().drop(columns=["OWNED_WEAK"]).copy()
    # The lowest-ranked candidate ends above its long trend, so the filter must
    # remove nobody rather than reaching above the bottom-quintile boundary.
    prices.loc[prices.index[-1], "WEAK2"] = prices["WEAK2"].iloc[-200:].mean() * 1.10
    kept, report = apply_robust_momentum_filter(prices, config=config())
    assert kept == list(prices.columns)
    assert not report["Excluded"].any()


def test_optimizer_wiring_passes_only_positive_quantity_tickers_as_owned(monkeypatch):
    import portfolio_optimizer_core as core

    captured = {}

    def stop_after_capture(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("CAPTURED")

    monkeypatch.setitem(
        core.run_portfolio_analysis_multi.__globals__,
        "get_daily_log_returns",
        stop_after_capture,
    )
    allocation = pd.DataFrame(
        [
            {"Yahoo Ticker": "OWNED.NS", "Currency": "INR", "Quantity": 2},
            {"Yahoo Ticker": "CANDIDATE.NS", "Currency": "INR", "Quantity": 0},
        ]
    )
    with pytest.raises(RuntimeError, match="CAPTURED"):
        core.run_portfolio_analysis_multi(
            ["OWNED.NS", "CANDIDATE.NS"],
            allocation,
        )

    assert captured["owned_tickers"] == ("OWNED.NS",)
