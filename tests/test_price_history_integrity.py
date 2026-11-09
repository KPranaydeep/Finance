import numpy as np
import pandas as pd

from price_history_integrity import (
    reconcile_flagged_history,
    reconcile_price_frame,
)


def _dhy_like_evidence():
    index = pd.to_datetime(
        ["2026-09-28", "2026-09-30", "2026-10-01", "2026-10-02"]
    )
    raw = pd.Series([1.61, np.nan, 16.05, 16.07], index=index)
    splits = pd.Series([0.0, 0.1, 0.0, 0.0], index=index)
    return {
        "raw_close": raw,
        "adjusted_close": raw.copy(),
        "dividends": pd.Series(0.0, index=index),
        "stock_splits": splits,
    }


def test_reverse_split_is_reconciled_instead_of_becoming_return():
    evidence = _dhy_like_evidence()
    repaired, report = reconcile_flagged_history(
        evidence["adjusted_close"], **evidence
    )

    observed = repaired.dropna().pct_change(fill_method=None).dropna()
    assert observed.abs().max() < 0.01
    assert report["status"] == "REPAIRED_CORPORATE_ACTION"
    assert report["split_events"] == [{"date": "2026-09-30", "ratio": 0.1}]


def test_unexplained_extreme_candidate_is_quarantined_not_clipped():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"JUMP": [10.0, 25.0, 26.0]}, index=index)
    evidence = {
        "JUMP": {
            "raw_close": prices["JUMP"],
            "adjusted_close": prices["JUMP"],
            "dividends": pd.Series(0.0, index=index),
            "stock_splits": pd.Series(0.0, index=index),
        }
    }

    corrected, report = reconcile_price_frame(prices, evidence)

    assert "JUMP" not in corrected.columns
    assert report.iloc[0]["Status"] == "QUARANTINED_UNEXPLAINED_JUMP"
    assert report.iloc[0]["Optimizer action"] == "EXCLUDE_NEW_CANDIDATE"


def test_unexplained_extreme_owned_holding_is_retained_for_weight_freeze():
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"OWNED": [10.0, 25.0, 26.0]}, index=index)
    evidence = {
        "OWNED": {
            "raw_close": prices["OWNED"],
            "adjusted_close": prices["OWNED"],
            "dividends": pd.Series(0.0, index=index),
            "stock_splits": pd.Series(0.0, index=index),
        }
    }

    corrected, report = reconcile_price_frame(
        prices, evidence, owned_tickers=("OWNED",)
    )

    assert "OWNED" in corrected.columns
    assert bool(report.iloc[0]["Owned"])
    assert report.iloc[0]["Optimizer action"] == "FREEZE_OWNED_WEIGHT"


def test_quarantined_owned_weight_remains_fixed_through_postprocessing():
    import portfolio_optimizer_core as core

    bounds = core.weight_bounds(
        3,
        ("OWNED", "NEW1", "NEW2"),
        {"OWNED": 0.20},
    )
    initial = core.feasible_initial_weights(bounds)
    cleaned = core.enforce_min_weight_postprocess(
        np.array([0.20, 0.795, 0.005]),
        min_weight=0.01,
        frozen_positions={0: 0.20},
    )

    assert bounds[0] == (0.20, 0.20)
    assert np.isclose(initial.sum(), 1.0)
    assert np.isclose(cleaned[0], 0.20)
    assert np.isclose(cleaned.sum(), 1.0)


def test_optimizer_weight_bounds_accepts_dataframe_column_index():
    import portfolio_optimizer_core as core

    tickers = pd.Index(["A", "B", "C"])
    bounds = core.weight_bounds(3, tickers)

    assert len(bounds) == 3
    assert all(lower == 0.0 for lower, _ in bounds)


def test_confirmed_crash_remains_a_loss_observation_and_owned_weight_is_not_frozen():
    index = pd.bdate_range("2026-01-01", periods=4)
    series = pd.Series([10., 10., 5., 5.2], index=index)
    evidence = {"CRASH": {"raw_close":series, "adjusted_close":series,
                            "confirmation_close": series * 2,
                            "confirmation_source":"Independent test exchange feed"}}
    corrected, report = reconcile_price_frame(pd.DataFrame({"CRASH":series}), evidence,
                                               owned_tickers=["CRASH"])
    assert report.iloc[0]["Status"] == "VERIFIED_MARKET_MOVE"
    assert report.iloc[0]["Optimizer action"] == "USE_RECONCILED_HISTORY"
    assert corrected["CRASH"].pct_change().min() == -.5


def test_missing_conflicting_or_same_provider_witness_cannot_certify_a_crash():
    index = pd.bdate_range("2026-01-01", periods=3)
    series = pd.Series([10.,5.,5.1],index=index)
    for witness, source in [(pd.Series([10.,9.,9.1],index=index),"Independent"),
                            (series.iloc[1:],"Independent"), (series,"Yahoo chart")]:
        _, report = reconcile_flagged_history(series, raw_close=series, adjusted_close=series,
                                               confirmation_close=witness, confirmation_source=source)
        assert report["status"] == "QUARANTINED_UNEXPLAINED_JUMP"


def test_pipeline_requests_second_source_only_for_unresolved_moves(monkeypatch):
    import portfolio_optimizer_core as core
    ns = core.apply_price_integrity_gate.__globals__
    dates = pd.bdate_range("2026-01-01", periods=3)
    series = pd.Series([10.,5.,5.1],index=dates)
    primary = {"CRASH": {"raw_close":series,"adjusted_close":series,
                          "dividends":pd.Series(0.,index=dates),"stock_splits":pd.Series(0.,index=dates)}}
    calls=[]
    monkeypatch.setitem(ns,"_download_price_integrity_evidence_batch",lambda *a,**k:primary)
    def confirmation(ticker, *args):
        calls.append(ticker)
        return {"confirmation_close":series,"confirmation_source":"Independent test exchange"}
    monkeypatch.setitem(ns,"_download_price_move_confirmation",confirmation)
    corrected, report = core.apply_price_integrity_gate(pd.DataFrame({"CRASH":series,"NORMAL":[10.,11.,12.]},index=dates),owned_tickers=["CRASH"])
    assert calls == ["CRASH"]
    assert report.iloc[0]["Status"] == "VERIFIED_MARKET_MOVE"
    assert corrected["CRASH"].pct_change().min() == -.5
