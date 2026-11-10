import threading
import time

import numpy as np
import pandas as pd

from price_history_integrity import extreme_return_columns, extreme_return_dates, reconcile_price_frame
from price_integrity_runtime import EvidenceCache, bounded_evidence, integrity_progress_label


def test_block_scan_matches_observed_prices_with_gaps_and_invalid_values():
    prices = pd.DataFrame({"A": [10, np.nan, 4, 5, np.inf],
                           "B": [0, 10, -4, 10, 10],
                           "C": [10, 11, 12, 13, 14]})
    expected = [x for x in prices if extreme_return_dates(prices[x])]
    assert extreme_return_columns(prices) == expected == ["A"]


def test_completed_evidence_survives_new_cache_instance_and_scope_changes(tmp_path):
    dates = pd.date_range("2026-01-01", periods=3)
    path = tmp_path / "evidence.db"
    cache = EvidenceCache(path, "v1", "2026-01-01", "2026-01-04")
    calls = []
    updates = []
    thread_id = threading.get_ident()

    def fetch(batch):
        calls.extend(batch)
        return {x: {"raw_close": pd.Series([10., 5., 6.], index=dates)} for x in batch}

    def progress(details):
        assert threading.get_ident() == thread_id
        updates.append(details)

    first = bounded_evidence(["A", "B"], fetch, cache=cache, kind="primary", on_progress=progress)
    fresh_instance = EvidenceCache(path, "v1", "2026-01-01", "2026-01-04")
    second = bounded_evidence(["A", "B"], fetch, cache=fresh_instance, kind="primary", on_progress=progress)
    assert calls == ["A", "B"]
    pd.testing.assert_series_equal(first["A"]["raw_close"], second["A"]["raw_close"], check_freq=False)
    assert updates[-1]["integrity_cached"] == 2
    assert "2/2" in integrity_progress_label(updates[-1])
    assert EvidenceCache(path, "v2", "2026-01-01", "2026-01-04").read("primary", ["A"]) == {}
    assert EvidenceCache(path, "v1", "2026-01-01", "2026-01-05").read("primary", ["A"]) == {}


def test_deadline_returns_without_waiting_for_provider_and_late_success_is_cached(tmp_path):
    release = threading.Event()
    entered = threading.Event()
    finished = threading.Event()
    cache = EvidenceCache(tmp_path / "evidence.db", "v1", "start", "end")

    def stalled(batch):
        entered.set()
        release.wait(2)
        finished.set()
        return {x: {"source": "independent"} for x in batch}

    started = time.monotonic()
    try:
        results = bounded_evidence(["A"], stalled, cache=cache, kind="secondary", budget_seconds=.05)
        assert entered.is_set()
        assert time.monotonic() - started < .5
        assert results == {}
    finally:
        release.set()
    assert finished.wait(1)
    deadline = time.monotonic() + 1
    while not cache.read("secondary", ["A"]) and time.monotonic() < deadline:
        time.sleep(.01)
    assert cache.read("secondary", ["A"]) == {"A": {"source": "independent"}}
    assert results == {}  # Background completion cannot mutate the published report.


def test_failure_cached_briefly_then_retried_and_no_retry_cascade(tmp_path, monkeypatch):
    cache = EvidenceCache(tmp_path / "evidence.db", "v1", "start", "end")
    calls = []

    def fail(batch):
        calls.append(batch)
        raise ValueError("provider unavailable")

    assert bounded_evidence(["A"], fail, cache=cache, kind="primary") == {"A": None}
    assert bounded_evidence(["A"], fail, cache=cache, kind="primary") == {"A": None}
    assert len(calls) == 1
    future_time = time.time() + 901
    monkeypatch.setattr("price_integrity_runtime.time.time", lambda: future_time)
    bounded_evidence(["A"], fail, cache=cache, kind="primary")
    assert len(calls) == 2


def test_no_evidence_quarantines_candidates_keeps_owned_and_preserves_other_history():
    prices = pd.DataFrame({"OWNED": [10., 5., 5.1], "NEW": [10., 25., 25.1],
                           "NORMAL": [10., 11., 12.]}, index=pd.bdate_range("2026-01-01", periods=3))
    corrected, report = reconcile_price_frame(prices, {}, owned_tickers=["OWNED"])
    assert list(corrected) == ["OWNED", "NORMAL"]
    pd.testing.assert_series_equal(corrected["NORMAL"], prices["NORMAL"])
    assert report["Status"].eq("QUARANTINED_VALIDATION_UNAVAILABLE").all()
    assert report.set_index("Ticker").loc["OWNED", "Optimizer action"] == "FREEZE_OWNED_WEIGHT"


def test_no_budget_launches_no_network_and_broken_progress_does_not_change_evidence(tmp_path):
    cache = EvidenceCache(tmp_path / "evidence.db", "v1", "start", "end")
    calls = []
    result = bounded_evidence(["A"], lambda batch: calls.append(batch), cache=cache,
                              kind="primary", budget_seconds=0)
    assert calls == [] and result == {}

    def broken_progress(details):
        raise RuntimeError("UI unavailable")

    assert bounded_evidence(["A"], lambda batch: {"A": {"source": "test"}}, cache=cache,
                            kind="primary", on_progress=broken_progress) == {"A": {"source": "test"}}


def test_provider_history_uses_isolated_ticker_and_preserves_local_session_dates(monkeypatch):
    import yfinance as yf
    import portfolio_optimizer_core as core
    dates = pd.date_range("2026-01-01", periods=3, tz="Asia/Kolkata")
    calls = []

    class Ticker:
        def __init__(self, symbol):
            assert symbol == "SYNTH.NS"

        def history(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"Close": [10., 5., 6.], "Adj Close": [10., 5., 6.],
                                 "Dividends": [0., 0., 0.], "Stock Splits": [0., 0., 0.]}, index=dates)

    monkeypatch.setattr(yf, "Ticker", Ticker)
    result = core._download_price_integrity_evidence("SYNTH.NS", "2026-01-01", "2026-01-04")
    assert calls[0]["timeout"] == 5 and calls[0]["raise_errors"] is True
    assert result["raw_close"].index.equals(dates.tz_localize(None))
    assert result["raw_close"].iloc[1] == 5.
