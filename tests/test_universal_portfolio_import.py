import sqlite3

import pandas as pd
import pytest

from universal_portfolio_import import (
    apply_verified_import,
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


def test_prepare_is_resumable_and_protects_exchange_collisions():
    conn = connection()
    conn.execute(
        """
        INSERT INTO master_holdings VALUES
        ('__universal__', 'VT', 'Vanguard Total World', 'VT', 'PCX', 'USD', 0, NULL, 'x', 'x'),
        ('__universal__', 'SBC', 'SBC Exports', 'SBC.NS', 'NSI', 'INR', 0, NULL, 'x', 'x')
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
