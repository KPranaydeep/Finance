from pathlib import Path

import pandas as pd

from market_history_cache import sync_market_history


def _frames(tickers, start, end):
    index = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    closes = pd.DataFrame(
        {ticker: range(100, 100 + len(index)) for ticker in tickers}, index=index
    )
    volumes = pd.DataFrame(
        {ticker: range(1_000, 1_000 + len(index)) for ticker in tickers}, index=index
    )
    return closes, volumes


def test_history_cache_reuses_and_incrementally_extends(tmp_path):
    cache_path = Path(tmp_path) / "history.duckdb"
    calls = []

    def fetcher(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        closes, volumes = _frames(tickers, start, end)
        return closes, volumes, {}

    closes, volumes, first = sync_market_history(
        cache_path,
        ["AAA", "BBB"],
        "2026-01-01",
        "2026-01-04",
        fetcher,
        batch_size=120,
    )

    assert calls == [(('AAA', 'BBB'), '2026-01-01', '2026-01-04')]
    assert closes.shape == (3, 2)
    assert volumes.shape == (3, 2)
    assert first["symbols_refreshed"] == 2
    assert first["rows_written"] == 6

    calls.clear()
    cached_closes, _, second = sync_market_history(
        cache_path,
        ["AAA", "BBB"],
        "2026-01-01",
        "2026-01-04",
        fetcher,
    )
    assert calls == []
    assert cached_closes.equals(closes)
    assert second["symbols_reused"] == 2
    assert second["network_requests"] == 0

    calls.clear()
    extended, _, third = sync_market_history(
        cache_path,
        ["AAA", "BBB"],
        "2026-01-01",
        "2026-01-05",
        fetcher,
    )
    assert calls == [(('AAA', 'BBB'), '2026-01-04', '2026-01-05')]
    assert extended.shape == (4, 2)
    assert third["rows_written"] == 2


def test_recent_no_data_attempt_is_not_retried_immediately(tmp_path):
    cache_path = Path(tmp_path) / "history.duckdb"
    calls = []

    def empty_fetcher(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        return pd.DataFrame(), pd.DataFrame(), {ticker: "missing" for ticker in tickers}

    _, _, first = sync_market_history(
        cache_path,
        ["MISSING"],
        "2026-01-01",
        "2026-01-04",
        empty_fetcher,
    )
    _, _, second = sync_market_history(
        cache_path,
        ["MISSING"],
        "2026-01-01",
        "2026-01-04",
        empty_fetcher,
    )

    assert len(calls) == 1
    assert first["symbols_unavailable"] == 1
    assert second["network_requests"] == 0


def test_progress_reports_planning_batches_and_cache_load(tmp_path):
    cache_path = Path(tmp_path) / "history.duckdb"
    progress = []

    def fetcher(tickers, start, end):
        closes, volumes = _frames(tickers, start, end)
        return closes, volumes, {}

    sync_market_history(
        cache_path,
        ["AAA", "BBB", "CCC"],
        "2026-01-01",
        "2026-01-03",
        fetcher,
        batch_size=2,
        on_progress=lambda stage, fraction, details: progress.append(
            (stage, fraction, details)
        ),
    )

    stages = [item[0] for item in progress]
    assert stages[0] == "Planning incremental history refresh"
    assert stages.count("Refreshing Yahoo history") == 2
    assert stages[-1] == "Loading cached history"


def test_known_pre_listing_gap_is_not_downloaded_again(tmp_path):
    cache_path = Path(tmp_path) / "history.duckdb"
    calls = []

    def listed_late_fetcher(tickers, start, end):
        calls.append((tuple(tickers), start, end))
        closes, volumes = _frames(tickers, "2026-01-03", end)
        return closes, volumes, {}

    sync_market_history(
        cache_path,
        ["IPO"],
        "2026-01-01",
        "2026-01-05",
        listed_late_fetcher,
    )
    sync_market_history(
        cache_path,
        ["IPO"],
        "2026-01-01",
        "2026-01-05",
        listed_late_fetcher,
        retry_after_hours=0,
    )

    assert calls == [(('IPO',), '2026-01-01', '2026-01-05')]
