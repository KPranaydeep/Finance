import sqlite3
from datetime import datetime, timezone

import pandas as pd

from trade_deferrals import (
    action_row_style,
    active_trade_deferrals,
    defer_trades_for_quarter,
    exclude_deferred_candidates,
    optimal_weight_pause_rows,
    split_deferred_trades,
)


def connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_selected_trade_is_paused_for_exactly_90_days():
    conn = connection()
    now = datetime(2026, 10, 5, 6, 30, tzinfo=timezone.utc)
    defer_trades_for_quarter(
        conn,
        "owner",
        [{"Yahoo Ticker": "ABC", "Symbol": "ABC", "Action": "Buy"}],
        now=now,
    )

    active = active_trade_deferrals(conn, "owner", now=now)
    assert active.loc[0, "expires_at"] == "2027-01-03T06:30:00+00:00"

    plan = pd.DataFrame(
        [
            {"Yahoo Ticker": "ABC", "Symbol": "ABC", "Action": "Buy"},
            {"Yahoo Ticker": "XYZ", "Symbol": "XYZ", "Action": "Sell"},
        ]
    )
    executable, deferred = split_deferred_trades(plan, active)
    assert executable["Yahoo Ticker"].tolist() == ["XYZ"]
    assert deferred["Yahoo Ticker"].tolist() == ["ABC"]


def test_expired_pause_is_removed_and_owner_scoped():
    conn = connection()
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = [{"Yahoo Ticker": "ABC", "Symbol": "ABC", "Action": "Sell"}]
    defer_trades_for_quarter(conn, "first", row, now=start)
    defer_trades_for_quarter(conn, "second", row, now=start)

    before_expiry = active_trade_deferrals(
        conn, "first", now=datetime(2026, 3, 31, tzinfo=timezone.utc)
    )
    after_expiry = active_trade_deferrals(
        conn, "first", now=datetime(2026, 4, 2, tzinfo=timezone.utc)
    )
    second_owner = active_trade_deferrals(
        conn, "second", now=datetime(2026, 3, 31, tzinfo=timezone.utc)
    )

    assert len(before_expiry) == 1
    assert after_expiry.empty
    assert len(second_owner) == 1


def test_active_pause_excludes_security_from_universal_candidates():
    conn = connection()
    now = datetime(2026, 10, 5, tzinfo=timezone.utc)
    defer_trades_for_quarter(
        conn,
        "owner",
        [{"Yahoo Ticker": "ABC", "Symbol": "ABC", "Action": "Buy"}],
        now=now,
    )
    active = active_trade_deferrals(conn, "owner", now=now)
    candidates = pd.DataFrame(
        {"Yahoo Ticker": ["ABC", "XYZ"], "Symbol": ["ABC", "XYZ"]}
    )

    filtered, count = exclude_deferred_candidates(candidates, active)

    assert count == 1
    assert filtered["Yahoo Ticker"].tolist() == ["XYZ"]


def test_action_row_style_tints_buy_and_sell_without_recoloring_other_rows():
    buy_style = action_row_style(pd.Series({"Action": "Buy", "Symbol": "ABC"}))
    sell_style = action_row_style(pd.Series({"Action": "Sell", "Symbol": "XYZ"}))
    hold_style = action_row_style(pd.Series({"Action": "Hold", "Symbol": "DEF"}))

    assert all("46, 125, 50" in value for value in buy_style)
    assert all("198, 40, 40" in value for value in sell_style)
    assert hold_style == ["", ""]


def test_optimal_weight_pause_needs_no_trade_and_preserves_listing_identity():
    weights = [
        {"ticker": "same.ns", "target_weight": .4},
        {"ticker": "SAME", "target_weight": .2},
        {"ticker": "same.ns", "target_weight": .4},
        {"ticker": "EXIT", "target_weight": 0},
    ]
    rows = optimal_weight_pause_rows(weights)
    assert rows["Yahoo Ticker"].tolist() == ["SAME.NS", "SAME"]
    assert rows["Optimal Weight"].tolist() == [.4, .2]
    conn = connection()
    now = datetime(2026, 11, 5, tzinfo=timezone.utc)
    defer_trades_for_quarter(conn, "owner", rows.iloc[:1].to_dict("records"), now=now)
    active = active_trade_deferrals(conn, "owner", now=now)
    kept, count = exclude_deferred_candidates(rows, active)
    assert count == 1 and kept["Yahoo Ticker"].tolist() == ["SAME"]
    assert weights[0]["target_weight"] == .4
    assert active_trade_deferrals(conn, "different-owner", now=now).empty


def test_live_weight_rows_reject_missing_and_invalid_allocations():
    rows = optimal_weight_pause_rows(pd.DataFrame({
        "Ticker": ["HOLD", None, "ZERO", "INF", "NEGATIVE", "MISSING"],
        "Optimal Weight": [.25, .2, 0, float("inf"), -.2, None],
    }))
    assert rows.to_dict("records") == [{"Yahoo Ticker": "HOLD", "Optimal Weight": .25}]
