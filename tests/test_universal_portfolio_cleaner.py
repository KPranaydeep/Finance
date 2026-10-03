import sqlite3

import numpy as np
import pandas as pd
import pytest

from universal_portfolio_cleaner import (
    apply_cleaner_exclusions,
    clear_optimizer_exclusions,
    cluster_members_frame,
    delete_cluster_snapshot,
    ensure_cleaner_schema,
    finalize_cleaner_job,
    filter_optimizer_candidates,
    get_cleaner_job,
    insider_sale_signal,
    optimizer_exclusions_frame,
    prepare_cleaner_job,
    representative_cluster_sample,
    record_history_batch,
    score_price_history,
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
    ensure_cleaner_schema(conn)
    conn.commit()
    return conn


def add_row(conn, owner, symbol, ticker, exchange="NYQ", currency="USD", quantity=0):
    conn.execute(
        "INSERT INTO master_holdings VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'x', 'x')",
        (owner, symbol, symbol, ticker, exchange, currency, quantity),
    )


def falling_history(start=300, stop=50):
    return pd.Series(np.linspace(start, stop, 240))


def rising_history(start=50, stop=300):
    return pd.Series(np.linspace(start, stop, 240))


def test_price_screen_uses_strict_bearish_dma_stack():
    bearish = score_price_history(falling_history())
    healthy = score_price_history(rising_history())
    short = score_price_history(pd.Series(range(50)))

    assert bearish["bearish_stack"] == 1
    assert bearish["latest_price"] < bearish["dma25"] < bearish["dma50"] < bearish["dma200"]
    assert healthy["bearish_stack"] == 0
    assert short["history_status"] == "insufficient_history"


def test_stale_history_is_flagged_for_manual_listing_review():
    stale = pd.Series(
        np.linspace(100, 90, 240),
        index=pd.date_range("2025-01-01", periods=240, freq="D"),
    )

    result = score_price_history(stale, as_of="2026-10-01T00:00:00Z")

    assert result["history_status"] == "no_price"
    assert "stale" in result["reason"].lower()


def test_management_sales_are_secondary_and_missing_data_is_neutral():
    transactions = pd.DataFrame(
        [
            {
                "Text": "Sale at price 100.00 per share.",
                "Position": "Chief Financial Officer",
                "Start Date": "2026-09-01",
                "Value": 1000,
            },
            {
                "Text": "Purchase at price 90.00 per share.",
                "Position": "Director",
                "Start Date": "2026-09-02",
                "Value": 250,
            },
            {
                "Text": "Sale at price 80.00 per share.",
                "Position": "10% Owner",
                "Start Date": "2026-09-03",
                "Value": 9999,
            },
        ]
    )

    result = insider_sale_signal(transactions, as_of="2026-10-01T00:00:00Z")

    assert result["status"] == "available"
    assert result["sale_value"] == 1000
    assert result["net_sale_ratio"] == 0.6
    assert insider_sale_signal(pd.DataFrame())["status"] == "unavailable"


def test_cluster_scope_and_owned_holdings_are_protected():
    conn = connection()
    add_row(conn, "__universal__", "AAA", "AAA", "NYQ", "USD")
    add_row(conn, "__universal__", "BBB.NS", "BBB.NS", "NSI", "INR")
    add_row(conn, "alice", "AAA", "AAA", "NYQ", "USD", quantity=4)
    conn.commit()

    job = prepare_cleaner_job(conn, "__universal__", ["NYQ · USD"], 10, False)

    assert job["total_symbols"] == 1
    assert job["counts"]["protected"] == 1


def test_cleaner_accepts_full_exclusion_cap_and_rejects_above_100():
    conn = connection()
    add_row(conn, "__universal__", "AAA", "AAA")
    conn.commit()

    job = prepare_cleaner_job(conn, "__universal__", [], 100, False)

    assert job["removal_percentile"] == 100
    with pytest.raises(ValueError, match="between 1% and 100%"):
        prepare_cleaner_job(conn, "__universal__", [], 101, False)


def test_cluster_sample_is_stable_and_cluster_delete_uses_reviewed_snapshot_only():
    conn = connection()
    for index in range(8):
        add_row(conn, "__universal__", f"US{index}", f"US{index}", "NYQ", "USD")
    add_row(conn, "__universal__", "INDIA.NS", "INDIA.NS", "NSI", "INR")
    conn.commit()
    members = cluster_members_frame(conn, "__universal__", "NYQ · USD")
    sample_a = representative_cluster_sample(members, "NYQ · USD", sample_size=4)
    sample_b = representative_cluster_sample(
        members.iloc[::-1], "NYQ · USD", sample_size=4
    )

    assert len(members) == 8
    assert sample_a["Symbol"].tolist() == sample_b["Symbol"].tolist()

    reviewed = members["Symbol"].head(2).tolist()
    add_row(conn, "alice", reviewed[0], reviewed[0], quantity=3)
    conn.commit()
    result = delete_cluster_snapshot(
        conn, "__universal__", "NYQ · USD", reviewed
    )

    assert result["protected"] == [reviewed[0]]
    assert result["removed"] == [reviewed[1]]
    assert len(result["not_in_reviewed_snapshot"]) == 6
    assert conn.execute(
        "SELECT COUNT(*) FROM master_holdings WHERE owner='__universal__'"
    ).fetchone()[0] == 8


def test_final_proposal_is_ranked_and_hard_capped_then_rechecks_ownership():
    conn = connection()
    for index in range(10):
        add_row(conn, "__universal__", f"S{index}", f"S{index}")
    conn.commit()
    job = prepare_cleaner_job(conn, "__universal__", [], 20, False)
    histories = {
        f"S{index}": score_price_history(
            falling_history(400 - index * 5, 30 + index) if index < 4 else rising_history()
        )
        for index in range(10)
    }
    record_history_batch(conn, job["job_id"], histories)

    ready = finalize_cleaner_job(conn, job["job_id"])
    proposed = conn.execute(
        "SELECT symbol FROM universal_cleaner_items WHERE job_id=? AND proposed_remove=1",
        (job["job_id"],),
    ).fetchall()

    assert ready["status"] == "ready"
    assert len(proposed) == 2
    newly_owned = proposed[0]["symbol"]
    add_row(conn, "alice", newly_owned, newly_owned, quantity=1)
    conn.commit()

    result = apply_cleaner_exclusions(conn, job["job_id"], "__universal__")

    assert result["protected_at_apply"] == [newly_owned]
    assert len(result["excluded"]) == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM master_holdings WHERE owner='__universal__'"
    ).fetchone()[0] == 10
    assert optimizer_exclusions_frame(conn)["Symbol"].tolist() == result["excluded"]


def test_scoped_cleaner_exclusions_accumulate_and_are_reversible():
    conn = connection()
    for symbol, exchange, currency in (("US", "NYQ", "USD"), ("IN.NS", "NSI", "INR")):
        add_row(conn, "__universal__", symbol, symbol, exchange, currency)
    conn.commit()

    for cluster in ("NYQ · USD", "NSI · INR"):
        job = prepare_cleaner_job(conn, "__universal__", [cluster], 20, False)
        symbol = "US" if cluster.startswith("NYQ") else "IN.NS"
        record_history_batch(conn, job["job_id"], {symbol: score_price_history(falling_history())})
        ready = finalize_cleaner_job(conn, job["job_id"])
        # A one-row scope is conservatively capped to one eligible exclusion.
        conn.execute(
            "UPDATE universal_cleaner_items SET proposed_remove=1 WHERE job_id=?",
            (ready["job_id"],),
        )
        conn.execute(
            "UPDATE universal_cleaner_jobs SET status='ready' WHERE job_id=?",
            (ready["job_id"],),
        )
        conn.commit()
        apply_cleaner_exclusions(conn, ready["job_id"], "__universal__")

    assert set(optimizer_exclusions_frame(conn)["Symbol"]) == {"US", "IN.NS"}
    assert clear_optimizer_exclusions(conn) == 2
    assert optimizer_exclusions_frame(conn).empty


def test_optimizer_filter_matches_symbol_or_ticker_without_mutating_input():
    candidates = pd.DataFrame(
        [
            {"Symbol": "AAA", "Yahoo Ticker": "AAA"},
            {"Symbol": "BRK", "Yahoo Ticker": "BRK-B"},
            {"Symbol": "KEEP", "Yahoo Ticker": "KEEP"},
        ]
    )
    exclusions = pd.DataFrame(
        [
            {"Symbol": "AAA", "Yahoo ticker": "AAA"},
            {"Symbol": "OLD-BRK", "Yahoo ticker": "BRK-B"},
        ]
    )

    filtered = filter_optimizer_candidates(candidates, exclusions)

    assert filtered["Symbol"].tolist() == ["KEEP"]
    assert len(candidates) == 3


def test_broad_market_data_failure_blocks_all_removals():
    conn = connection()
    for index in range(10):
        add_row(conn, "__universal__", f"S{index}", f"S{index}")
    conn.commit()
    job = prepare_cleaner_job(conn, "__universal__", [], 20, False)
    results = {
        f"S{index}": (
            score_price_history(rising_history())
            if index < 7
            else {"history_status": "no_price"}
        )
        for index in range(10)
    }
    record_history_batch(conn, job["job_id"], results)

    blocked = finalize_cleaner_job(conn, job["job_id"])

    assert blocked["status"] == "blocked"
    assert blocked["counts"]["proposed"] == 0
    assert "usable history" in blocked["note"]
