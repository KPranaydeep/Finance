"""Persistent incremental cache for broad-universe daily market history."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter

import pandas as pd


_CACHE_LOCK = RLock()
SCHEMA_VERSION = 1


def _connect(path):
    import duckdb

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(cache_path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_history (
            ticker VARCHAR NOT NULL,
            session_date DATE NOT NULL,
            close DOUBLE,
            volume DOUBLE,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (ticker, session_date)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_history_sync (
            ticker VARCHAR PRIMARY KEY,
            requested_start DATE NOT NULL,
            requested_end DATE NOT NULL,
            attempted_at TIMESTAMPTZ NOT NULL,
            status VARCHAR NOT NULL
        )
        """
    )
    return connection


def _normalized_tickers(tickers):
    return tuple(
        str(ticker).strip().upper()
        for ticker in dict.fromkeys(tickers)
        if str(ticker).strip()
    )


def _coverage(connection, tickers):
    if not tickers:
        return {}
    rows = connection.execute(
        """
        SELECT ticker, MIN(session_date), MAX(session_date), COUNT(*)
        FROM market_history
        WHERE ticker IN (SELECT UNNEST(?))
        GROUP BY ticker
        """,
        [list(tickers)],
    ).fetchall()
    return {
        str(ticker): {
            "first": pd.Timestamp(first).normalize(),
            "last": pd.Timestamp(last).normalize(),
            "rows": int(rows_count),
        }
        for ticker, first, last, rows_count in rows
    }


def _attempts(connection, tickers, retry_after_hours):
    if not tickers:
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=float(retry_after_hours))
    rows = connection.execute(
        """
        SELECT ticker, requested_start, requested_end, attempted_at, status
        FROM market_history_sync
        WHERE ticker IN (SELECT UNNEST(?))
        """,
        [list(tickers)],
    ).fetchall()
    return {
        str(ticker): {
            "start": pd.Timestamp(start).normalize(),
            "end": pd.Timestamp(end).normalize(),
            "attempted_at": attempted_at,
            "status": str(status),
            "recent": attempted_at >= cutoff,
        }
        for ticker, start, end, attempted_at, status in rows
    }


def _requests_for_ticker(ticker, coverage, attempt, requested_start, requested_end):
    """Return uncovered half-open date ranges, suppressing recent retry storms."""
    ranges = []
    covered = coverage.get(ticker)
    if covered is None:
        if not (
            attempt is not None
            and attempt["recent"]
            and attempt["start"] <= requested_start
            and attempt["end"] >= requested_end
        ):
            ranges.append((requested_start, requested_end))
    else:
        first = covered["first"]
        last_exclusive = covered["last"] + pd.Timedelta(days=1)
        # Once Yahoo has been asked for the complete prefix, a later first date
        # is treated as the instrument's available-history boundary. Do not
        # redownload a pre-IPO/pre-listing void every day.
        prefix_already_attempted = (
            attempt is not None and attempt["start"] <= requested_start
        )
        if first > requested_start and not prefix_already_attempted:
            ranges.append((requested_start, min(first, requested_end)))
        recent_tail_attempt = (
            attempt is not None
            and attempt["recent"]
            and attempt["end"] >= requested_end
        )
        if last_exclusive < requested_end and not recent_tail_attempt:
            ranges.append((max(last_exclusive, requested_start), requested_end))

    return [(start, end) for start, end in ranges if start < end]


def _history_rows(closes, volumes):
    tickers = sorted(set(closes.columns) | set(volumes.columns))
    frames = []
    for ticker in tickers:
        close = (
            pd.to_numeric(closes[ticker], errors="coerce")
            if ticker in closes
            else pd.Series(dtype=float)
        )
        volume = (
            pd.to_numeric(volumes[ticker], errors="coerce")
            if ticker in volumes
            else pd.Series(dtype=float)
        )
        index = close.index.union(volume.index)
        if index.empty:
            continue
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "session_date": pd.to_datetime(index).tz_localize(None).normalize(),
                "close": close.reindex(index).to_numpy(dtype=float),
                "volume": volume.reindex(index).to_numpy(dtype=float),
            }
        )
        frame = frame.dropna(subset=["close", "volume"], how="all")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["ticker", "session_date", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


def _upsert_rows(connection, rows):
    if rows.empty:
        return 0
    incoming = rows.copy()
    incoming["updated_at"] = datetime.now(timezone.utc)
    connection.register("incoming_market_history", incoming)
    try:
        connection.execute(
            """
            INSERT INTO market_history
            SELECT ticker, session_date, close, volume, updated_at
            FROM incoming_market_history
            ON CONFLICT (ticker, session_date) DO UPDATE SET
                close = COALESCE(EXCLUDED.close, market_history.close),
                volume = COALESCE(EXCLUDED.volume, market_history.volume),
                updated_at = EXCLUDED.updated_at
            """
        )
    finally:
        connection.unregister("incoming_market_history")
    return int(len(incoming))


def _record_attempts(connection, tickers, start, end, recovered):
    if not tickers:
        return
    now = datetime.now(timezone.utc)
    rows = pd.DataFrame(
        {
            "ticker": list(tickers),
            "requested_start": pd.Timestamp(start).date(),
            "requested_end": pd.Timestamp(end).date(),
            "attempted_at": now,
            "status": ["recovered" if ticker in recovered else "no_data" for ticker in tickers],
        }
    )
    connection.register("incoming_market_sync", rows)
    try:
        connection.execute(
            """
            INSERT INTO market_history_sync
            SELECT ticker, requested_start, requested_end, attempted_at, status
            FROM incoming_market_sync
            ON CONFLICT (ticker) DO UPDATE SET
                requested_start = LEAST(market_history_sync.requested_start, EXCLUDED.requested_start),
                requested_end = GREATEST(market_history_sync.requested_end, EXCLUDED.requested_end),
                attempted_at = EXCLUDED.attempted_at,
                status = EXCLUDED.status
            """
        )
    finally:
        connection.unregister("incoming_market_sync")


def _load_frames(connection, tickers, start, end):
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    data = connection.execute(
        """
        SELECT ticker, session_date, close, volume
        FROM market_history
        WHERE ticker IN (SELECT UNNEST(?))
          AND session_date >= ? AND session_date < ?
        ORDER BY session_date, ticker
        """,
        [list(tickers), pd.Timestamp(start).date(), pd.Timestamp(end).date()],
    ).df()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    closes = data.pivot(index="session_date", columns="ticker", values="close")
    volumes = data.pivot(index="session_date", columns="ticker", values="volume")
    for frame in (closes, volumes):
        frame.index = pd.DatetimeIndex(frame.index)
        frame.columns.name = None
    closes = closes.dropna(axis=1, how="all").sort_index()
    volumes = volumes.dropna(axis=1, how="all").sort_index()
    return closes, volumes


def sync_market_history(
    path,
    tickers,
    start,
    end,
    fetcher,
    *,
    batch_size=120,
    retry_after_hours=20,
    on_progress=None,
):
    """Refresh missing ranges and return cached close/volume matrices plus diagnostics.

    ``fetcher`` receives ``(tickers, start_string, end_string)`` and returns
    ``(close_frame, volume_frame, failures)``. Dates use yfinance's half-open
    convention: start inclusive, end exclusive.
    """
    symbols = _normalized_tickers(tickers)
    requested_start = pd.Timestamp(start).tz_localize(None).normalize()
    requested_end = pd.Timestamp(end).tz_localize(None).normalize()
    if requested_start >= requested_end:
        raise ValueError("Market-history cache requires start before end.")

    started = perf_counter()
    timings = {
        "cache_planning": 0.0,
        "yahoo_download": 0.0,
        "cache_write": 0.0,
        "cache_read": 0.0,
    }
    with _CACHE_LOCK:
        connection = _connect(path)
        try:
            planning_started = perf_counter()
            before = _coverage(connection, symbols)
            attempts = _attempts(connection, symbols, retry_after_hours)
            grouped = defaultdict(list)
            for ticker in symbols:
                for range_start, range_end in _requests_for_ticker(
                    ticker, before, attempts.get(ticker), requested_start, requested_end
                ):
                    grouped[(range_start, range_end)].append(ticker)

            planned_batches = sum(
                (len(range_tickers) + max(int(batch_size), 1) - 1)
                // max(int(batch_size), 1)
                for range_tickers in grouped.values()
            )
            timings["cache_planning"] = perf_counter() - planning_started
            if on_progress is not None:
                on_progress(
                    "Planning incremental history refresh",
                    0.03,
                    {
                        "batches": int(planned_batches),
                        "symbols_reused": int(sum(ticker in before for ticker in symbols)),
                    },
                )

            refreshed = set()
            failed = set()
            written_rows = 0
            request_count = 0
            for (range_start, range_end), range_tickers in sorted(grouped.items()):
                for offset in range(0, len(range_tickers), max(int(batch_size), 1)):
                    batch = range_tickers[offset : offset + max(int(batch_size), 1)]
                    request_count += 1
                    if on_progress is not None:
                        on_progress(
                            "Refreshing Yahoo history",
                            0.05 + 0.67 * (request_count - 1) / max(planned_batches, 1),
                            {
                                "batch": int(request_count),
                                "batches": int(planned_batches),
                                "symbols": int(len(batch)),
                            },
                        )
                    fetch_started = perf_counter()
                    closes, volumes, failures = fetcher(
                        batch,
                        range_start.strftime("%Y-%m-%d"),
                        range_end.strftime("%Y-%m-%d"),
                    )
                    timings["yahoo_download"] += perf_counter() - fetch_started
                    closes = closes if isinstance(closes, pd.DataFrame) else pd.DataFrame()
                    volumes = volumes if isinstance(volumes, pd.DataFrame) else pd.DataFrame()
                    write_started = perf_counter()
                    rows = _history_rows(closes, volumes)
                    written_rows += _upsert_rows(connection, rows)
                    recovered = set(rows["ticker"].astype(str)) if not rows.empty else set()
                    refreshed.update(recovered)
                    failed.update(str(item).upper() for item in (failures or {}))
                    _record_attempts(
                        connection, batch, range_start, range_end, recovered
                    )

                    timings["cache_write"] += perf_counter() - write_started

            if on_progress is not None:
                on_progress(
                    "Loading cached history",
                    0.76,
                    {"available_symbols": int(len(before) + len(refreshed))},
                )
            read_started = perf_counter()
            closes, volumes = _load_frames(
                connection, symbols, requested_start, requested_end
            )
            after = _coverage(connection, symbols)
            timings["cache_read"] = perf_counter() - read_started
        finally:
            connection.close()

    available = set(closes.columns)
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "path": str(Path(path)),
        "requested_symbols": len(symbols),
        "symbols_reused": int(sum(ticker in before for ticker in symbols)),
        "symbols_refreshed": int(len(refreshed)),
        "symbols_available": int(len(available)),
        "symbols_unavailable": int(len(set(symbols) - available)),
        "rows_written": int(written_rows),
        "network_requests": int(request_count),
        "oldest_session": (
            min(item["first"] for item in after.values()).date().isoformat()
            if after
            else None
        ),
        "latest_session": (
            max(item["last"] for item in after.values()).date().isoformat()
            if after
            else None
        ),
        "failed_symbols_reported": int(len(failed)),
        "timings_seconds": {
            key: round(float(value), 3) for key, value in timings.items()
        },
        "total_seconds": round(float(perf_counter() - started), 3),
    }
    return closes, volumes, diagnostics
