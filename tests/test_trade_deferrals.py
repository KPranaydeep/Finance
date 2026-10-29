import sqlite3
from datetime import datetime, timezone

import pandas as pd

from trade_deferrals import (
    action_row_style,
    active_trade_deferrals,
    defer_trades_for_quarter,
    exclude_deferred_candidates,
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
