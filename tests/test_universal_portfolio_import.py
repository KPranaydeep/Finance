import sqlite3

import pandas as pd
import pytest

from universal_portfolio_import import (
    apply_verified_import,
    apply_verified_overseas_replacement,
    ensure_import_schema,
    get_import_job,
    import_items_frame,
    parse_universal_source_csv,
    prepare_import_job,
    record_validation_batch,
    safe_report_csv,
)


MASTER_DDL = """
CREATE TABLE master_holdings (
    owner TEXT NOT NULL,
    symbol TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    yahoo_ticker TEXT,
    exchange TEXT,
    currency TEXT,
    quantity REAL NOT NULL,
    average_price REAL,
    market_cap_millions REAL,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, symbol)
)
"""


def connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(MASTER_DDL)
    ensure_import_schema(conn)
    conn.commit()
    return conn


def test_schema_migrates_existing_import_jobs_with_merge_mode():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE universal_import_jobs (
            job_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL, file_name TEXT NOT NULL,
            status TEXT NOT NULL, source_rows INTEGER NOT NULL,
            unique_symbols INTEGER NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, applied_at TEXT
        )
        """
    )
    ensure_import_schema(conn)

    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(universal_import_jobs)")
    }

    assert "mode" in columns


def source_csv(rows):
    header = "Rank,Name,Symbol,pe_ratio_ttm,price (USD),country\n"
    body = "\n".join(",".join(map(str, row)) for row in rows)
    return (header + body + "\n").encode("utf-8")


def test_parser_accepts_supplied_companiesmarketcap_shape_and_deduplicates():
    content = source_csv(
        [
            (1, "Alpha", "AAA", 5, 10, "United States"),
            (2, "Alpha duplicate", "aaa", 6, 11, "United States"),
            (3, "Berkshire", "BRK-B", 7, 12, "United States"),
        ]
    )

    frame, stats = parse_universal_source_csv(content)

    assert frame["source_symbol"].tolist() == ["AAA", "BRK-B"]
    assert frame["stock_name"].tolist() == ["Alpha", "Berkshire"]
    assert stats == {
        "source_rows": 3,
        "unique_symbols": 2,
        "duplicates_removed": 1,
        "blank_symbols_removed": 0,
    }


def test_parser_requires_a_symbol_column():
    with pytest.raises(ValueError, match="Symbol or Ticker"):
        parse_universal_source_csv(b"Rank,Name\n1,Alpha\n")


def test_parser_accepts_tickertape_us_screener_columns():
    content = (
        '"name","ticker","industry","marketCapitalizationMln","prevClosePrice"\n'
        '"NVIDIA Corporation","NVDA","Semiconductors","5514691.8707","233.95"\n'
        '"Apple Inc.","AAPL","Consumer Electronics","4860153.9543","330.32"\n'
    ).encode("utf-8")

    frame, stats = parse_universal_source_csv(content)

    assert frame["source_symbol"].tolist() == ["NVDA", "AAPL"]
    assert frame["stock_name"].tolist() == ["NVIDIA Corporation", "Apple Inc."]
    assert frame["market_cap_millions"].tolist() == [5514691.8707, 4860153.9543]
    assert stats["unique_symbols"] == 2


def test_market_cap_metadata_survives_staging_and_apply():
    conn = connection()
    content = (
        '"name","ticker","marketCapitalizationMln"\n'
        '"NVIDIA Corporation","NVDA","5514691.8707"\n'
    ).encode("utf-8")
    job_id, _, _ = prepare_import_job(conn, content, "tickertape.csv", "__universal__")
    record_validation_batch(
        conn,
        job_id,
        ["NVDA"],
        {"NVDA": {"yahoo_ticker": "NVDA", "exchange": "NMS", "currency": "USD"}},
    )

    apply_verified_import(conn, job_id, "__universal__")

    stored = conn.execute(
        "SELECT market_cap_millions FROM master_holdings WHERE symbol='NVDA'"
    ).fetchone()
    assert stored[0] == pytest.approx(5514691.8707)


def test_prepare_is_resumable_and_protects_exchange_collisions():
    conn = connection()
    conn.execute(
        """
        INSERT INTO master_holdings VALUES
        ('__universal__', 'VT', 'Vanguard Total World', 'VT', 'PCX', 'USD', 0, NULL, NULL, 'x', 'x'),
        ('__universal__', 'SBC', 'SBC Exports', 'SBC.NS', 'NSI', 'INR', 0, NULL, NULL, 'x', 'x')
        """
    )
    conn.commit()
    content = source_csv(
        [
            (1, "Vanguard", "VT", 5, 10, "United States"),
            (2, "US Medical", "SBC", 6, 11, "United States"),
            (3, "New company", "NEW", 7, 12, "United States"),
        ]
    )

    job_id, resumed, job = prepare_import_job(
        conn, content, "companies.csv", "__universal__"
    )
    same_job_id, resumed_again, same_job = prepare_import_job(
        conn, content, "companies.csv", "__universal__"
    )

    assert resumed is False
    assert resumed_again is True
    assert same_job_id == job_id
    assert job["counts"] == {"conflict": 1, "existing": 1, "pending": 1}
    assert same_job["unique_symbols"] == 3


def test_validation_retries_then_apply_is_atomic_and_idempotent():
    conn = connection()
    content = source_csv(
        [
            (1, "Active", "ACT", 5, 10, "United States"),
            (2, "Inactive", "OLD", 6, 11, "United States"),
        ]
    )
    job_id, _, _ = prepare_import_job(conn, content, "source.csv", "__universal__")

    metadata = {"ACT": {"yahoo_ticker": "ACT", "exchange": "NYQ", "currency": "USD"}}
    record_validation_batch(conn, job_id, ["ACT", "OLD"], metadata, max_attempts=3)
    record_validation_batch(conn, job_id, ["OLD"], {}, max_attempts=3)
    ready = record_validation_batch(conn, job_id, ["OLD"], {}, max_attempts=3)

    assert ready["status"] == "ready"
    assert ready["counts"] == {"unresolved": 1, "verified": 1}
    result = apply_verified_import(conn, job_id, "__universal__")
    assert result["added"] == 1
    stored = conn.execute(
        "SELECT symbol, yahoo_ticker, quantity FROM master_holdings"
    ).fetchall()
    assert [tuple(row) for row in stored] == [("ACT", "ACT", 0.0)]
    assert get_import_job(conn, job_id)["status"] == "applied"

    second = apply_verified_import(conn, job_id, "__universal__")
    assert second["added"] == 0
    assert conn.execute("SELECT COUNT(*) FROM master_holdings").fetchone()[0] == 1


def test_apply_refuses_an_incomplete_scan():
    conn = connection()
    content = source_csv([(1, "Pending", "PEND", 5, 10, "United States")])
    job_id, _, _ = prepare_import_job(conn, content, "source.csv", "__universal__")

    with pytest.raises(ValueError, match="Validation is incomplete"):
        apply_verified_import(conn, job_id, "__universal__")


def test_report_export_neutralizes_spreadsheet_formulas():
    frame = pd.DataFrame(
        [{"Symbol": "SAFE", "Name": "=HYPERLINK(\"bad\")", "Status": "rejected"}]
    )

    exported = safe_report_csv(frame).decode("utf-8-sig")

    assert "'=HYPERLINK" in exported


def test_overseas_replacement_preserves_india_and_is_atomic():
    conn = connection()
    conn.execute(
        """
        INSERT INTO master_holdings VALUES
        ('__universal__', 'INDIA', 'India', 'INDIA.NS', 'NSI', 'INR', 0, NULL, NULL, 'x', 'x'),
        ('__universal__', 'OLD', 'Old US', 'OLD', 'NYQ', 'USD', 0, NULL, NULL, 'x', 'x'),
        ('__universal__', 'KEEP', 'Keep US', 'KEEP', 'NMS', 'USD', 0, NULL, NULL, 'x', 'x'),
        ('alice', 'PERSONAL', 'Personal', 'PERSONAL', 'NYQ', 'USD', 5, 10, NULL, 'x', 'x')
        """
    )
    conn.commit()
    content = source_csv(
        [
            (1, "Keep US", "KEEP", 5, 10, "United States"),
            (2, "New US", "NEW", 6, 11, "United States"),
            (3, "India collision", "INDIA", 7, 12, "United States"),
        ]
    )
    job_id, _, job = prepare_import_job(
        conn,
        content,
        "tickertape.csv",
        "__universal__",
        mode="replace_overseas",
    )

    assert job["mode"] == "replace_overseas"
    assert job["counts"] == {"conflict": 1, "pending": 2}
    metadata = {
        ticker: {"yahoo_ticker": ticker, "exchange": "NYQ", "currency": "USD"}
        for ticker in ("KEEP", "NEW")
    }
    record_validation_batch(conn, job_id, ["KEEP", "NEW"], metadata)
    result = apply_verified_overseas_replacement(
        conn, job_id, "__universal__", min_verified_ratio=0.5
    )

    assert result["removed_overseas"] == 2
    assert result["added_overseas"] == 2
    assert result["retained_india"] == 1
    universal = conn.execute(
        "SELECT symbol FROM master_holdings WHERE owner='__universal__' ORDER BY symbol"
    ).fetchall()
    assert [row["symbol"] for row in universal] == ["INDIA", "KEEP", "NEW"]
    assert conn.execute(
        "SELECT quantity FROM master_holdings WHERE owner='alice' AND symbol='PERSONAL'"
    ).fetchone()[0] == 5


def test_overseas_replacement_refuses_low_validation_coverage_without_deleting():
    conn = connection()
    conn.execute(
        """
        INSERT INTO master_holdings VALUES
        ('__universal__', 'OLD', 'Old US', 'OLD', 'NYQ', 'USD', 0, NULL, NULL, 'x', 'x')
        """
    )
    conn.commit()
    content = source_csv(
        [(index, f"Company {index}", f"S{index}", 5, 10, "United States") for index in range(10)]
    )
    job_id, _, _ = prepare_import_job(
        conn, content, "tickertape.csv", "__universal__", mode="replace_overseas"
    )
    metadata = {"S0": {"yahoo_ticker": "S0", "exchange": "NYQ", "currency": "USD"}}
    for _ in range(3):
        record_validation_batch(
            conn, job_id, [f"S{index}" for index in range(10)], metadata, max_attempts=3
        )

    with pytest.raises(ValueError, match="were verified"):
        apply_verified_overseas_replacement(conn, job_id, "__universal__")

    assert conn.execute(
        "SELECT COUNT(*) FROM master_holdings WHERE owner='__universal__' AND symbol='OLD'"
    ).fetchone()[0] == 1
