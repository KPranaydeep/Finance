import sqlite3

import pandas as pd
import pytest

from market_cap_snapshot import (
    apply_market_cap_snapshot,
    ensure_market_cap_metadata_schema,
    prepare_market_cap_snapshot,
    summarize_market_cap_coverage,
)


def holdings_connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE master_holdings (
            owner TEXT NOT NULL, symbol TEXT NOT NULL, stock_name TEXT NOT NULL,
            yahoo_ticker TEXT, exchange TEXT, currency TEXT, quantity REAL NOT NULL,
            average_price REAL, market_cap_millions REAL, added_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY(owner, symbol))"""
    )
    conn.executemany(
        """INSERT INTO master_holdings
           (owner,symbol,stock_name,yahoo_ticker,exchange,currency,quantity,
            average_price,market_cap_millions,added_at,updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [
            ("__universal__", "AAA", "Alpha", "AAA", "NMS", "USD", 0, None, None, "x", "x"),
            ("__universal__", "BBB", "Beta India", "BBB.NS", "NSI", "INR", 0, None, None, "x", "x"),
        ],
    )
    conn.commit()
    return conn


def test_snapshot_matches_exact_yahoo_ticker_and_applies_only_market_cap():
    conn = holdings_connection()
    holdings = pd.DataFrame(
        [
            {"Symbol": "AAA", "Yahoo Ticker": "AAA"},
            {"Symbol": "BBB", "Yahoo Ticker": "BBB.NS"},
        ]
    )
    content = (
        '"ticker","marketCapitalizationMln"\n'
        '"AAA","12500.5"\n'
        '"BBB","999999"\n'
        '"NO_MATCH","42"\n'
    ).encode()

    preview, counts = prepare_market_cap_snapshot(content, holdings)
    ensure_market_cap_metadata_schema(conn)
    result = apply_market_cap_snapshot(
        conn, preview, owner="__universal__", source="Tickertape CSV", as_of="2026-10-12"
    )

    stored = conn.execute(
        "SELECT market_cap_millions, market_cap_source, market_cap_as_of, quantity "
        "FROM master_holdings WHERE symbol='AAA'"
    ).fetchone()
    missed = conn.execute(
        "SELECT market_cap_millions FROM master_holdings WHERE symbol='BBB'"
    ).fetchone()
    assert counts["ready"] == 1
    assert counts["unmatched"] == 2
    assert result["updated"] == 1
    assert stored[0] == pytest.approx(12500.5)
    assert stored[1:3] == ("Tickertape CSV", "2026-10-12")
    assert stored[3] == 0
    assert missed[0] is None


def test_snapshot_requires_explicit_millions_column_and_source_date_is_not_inferred():
    with pytest.raises(ValueError, match="explicitly expressed in millions"):
        prepare_market_cap_snapshot(
            b'ticker,market cap\nAAA,12500\n',
            pd.DataFrame([{"Symbol": "AAA", "Yahoo Ticker": "AAA"}]),
        )


def test_coverage_marks_undated_caps_stale_and_uses_the_user_supplied_as_of_date():
    rows = pd.DataFrame(
        {
            "Market Cap Millions": [100.0, 200.0, None],
            "Market Cap As Of": ["2026-10-01", None, None],
            "Market Cap Source": ["provider", "old-import", None],
        }
    )
    summary = summarize_market_cap_coverage(rows, as_of="2026-10-14", max_age_days=90)
    assert summary["current"] == 1
    assert summary["stale"] == 1
    assert summary["missing"] == 1
    assert summary["sources"] == ["provider"]
