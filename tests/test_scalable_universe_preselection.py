import numpy as np
import pandas as pd

from scalable_universe_preselection import (
    convert_candidate_history_to_inr,
    rank_scalable_candidates,
)


def histories():
    index = pd.bdate_range("2025-01-01", periods=320)
    close = pd.DataFrame(index=index)
    volume = pd.DataFrame(index=index)
    for ticker, growth, activity in (
        ("US_FAST", 1.0, 1_000_000),
        ("US_SLOW", 0.2, 800_000),
        ("US_WEAK", -0.2, 900_000),
        ("IN_FAST", 0.8, 500_000),
        ("IN_SLOW", 0.1, 400_000),
    ):
        close[ticker] = 100 * np.exp(np.linspace(0, growth, len(index)))
        volume[ticker] = activity
    return close, volume


def test_preselection_is_capped_and_preserves_cluster_representation():
    close, volume = histories()
    clusters = {
        "US_FAST": "NYQ · USD",
        "US_SLOW": "NYQ · USD",
        "US_WEAK": "NYQ · USD",
        "IN_FAST": "NSI · INR",
        "IN_SLOW": "NSI · INR",
    }

    selected, report = rank_scalable_candidates(
        close, volume, clusters, maximum_candidates=3, minimum_per_cluster=1
    )

    assert len(selected) == 3
    assert "US_FAST" in selected
    assert "IN_FAST" in selected
    assert set(report.loc[report["Selected"], "Cluster"]) == {"NYQ · USD", "NSI · INR"}


def test_preselection_skips_unusable_history_and_is_deterministic():
    close, volume = histories()
    close["BROKEN"] = np.nan
    clusters = {ticker: "NYQ · USD" for ticker in close.columns}

    first, report = rank_scalable_candidates(close, volume, clusters, maximum_candidates=2)
    second, _ = rank_scalable_candidates(close, volume, clusters, maximum_candidates=2)

    assert first == second
    assert "BROKEN" not in report["Ticker"].tolist()


def test_emerging_winner_sleeve_retains_recent_breakout_with_volume_confirmation():
    index = pd.bdate_range("2025-01-01", periods=320)
    close = pd.DataFrame(index=index)
    volume = pd.DataFrame(index=index)
    for number in range(9):
        ticker = f"CORE{number}"
        close[ticker] = 100 * np.exp(np.linspace(0, 0.15 + number * 0.02, len(index)))
        volume[ticker] = 1_000_000
    close["EMERGING"] = np.r_[np.full(285, 100.0), np.linspace(100, 165, 35)]
    volume["EMERGING"] = np.r_[np.full(299, 100_000.0), np.full(21, 2_000_000.0)]
    clusters = {ticker: "NYQ · USD" for ticker in close.columns}

    selected, report = rank_scalable_candidates(
        close, volume, clusters, maximum_candidates=4, minimum_per_cluster=0
    )

    assert "EMERGING" in selected
    sleeve = report.loc[report["Ticker"].eq("EMERGING"), "Selection sleeve"].iloc[0]
    assert sleeve == "Emerging winner"


def test_recent_candidate_histories_are_converted_to_inr_without_double_fallback():
    index = pd.bdate_range("2026-01-01", periods=3)
    closes = pd.DataFrame(
        {"INDIA.NS": [100.0, 101.0, 102.0], "US": [10.0, 11.0, 12.0], "UK": [5.0, 6.0, 7.0]},
        index=index,
    )
    converted, omitted = convert_candidate_history_to_inr(
        closes,
        {"INDIA.NS": "INR", "US": "USD", "UK": "GBP"},
        {"USD": pd.Series([80.0, 81.0, 82.0], index=index)},
    )

    assert converted["INDIA.NS"].tolist() == closes["INDIA.NS"].tolist()
    assert converted["US"].tolist() == [800.0, 891.0, 984.0]
    assert "UK" not in converted.columns
    assert omitted == ["GBP"]
