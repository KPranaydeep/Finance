import numpy as np
import pandas as pd

from scalable_universe_preselection import rank_scalable_candidates


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
