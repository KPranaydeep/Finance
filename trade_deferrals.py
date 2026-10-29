"""Persistent, owner-scoped pauses for executable rebalancing trades."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


TRADE_DEFERRALS_DDL = """
CREATE TABLE IF NOT EXISTS trade_deferrals (
    owner TEXT NOT NULL,
    yahoo_ticker TEXT NOT NULL,
    symbol TEXT NOT NULL,
    requested_action TEXT NOT NULL,
    deferred_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY (owner, yahoo_ticker)
)
"""


def ensure_trade_deferral_schema(conn) -> None:
    conn.execute(TRADE_DEFERRALS_DDL)


def _as_utc(value: datetime | str | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def defer_trades_for_quarter(conn, owner: str, rows, now=None) -> list[dict]:
    """Pause selected tickers for exactly 90 days and return saved records."""
    ensure_trade_deferral_schema(conn)
    deferred_at = _as_utc(now)
    expires_at = deferred_at + timedelta(days=90)
    records = []

    for row in rows:
        ticker = str(row.get("Yahoo Ticker") or "").strip().upper()
        if not ticker:
            continue
        symbol = str(row.get("Symbol") or ticker).strip().upper()
        action = str(row.get("Action") or "Trade").strip().title()
        conn.execute(
            """
            INSERT INTO trade_deferrals
                (owner, yahoo_ticker, symbol, requested_action, deferred_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner, yahoo_ticker) DO UPDATE SET
                symbol = excluded.symbol,
                requested_action = excluded.requested_action,
                deferred_at = excluded.deferred_at,
                expires_at = excluded.expires_at
            """,
            (
                str(owner),
                ticker,
                symbol,
                action,
                deferred_at.isoformat(timespec="seconds"),
                expires_at.isoformat(timespec="seconds"),
            ),
        )
        records.append(
            {
                "Yahoo Ticker": ticker,
                "Symbol": symbol,
                "Action": action,
                "Deferred At": deferred_at.isoformat(timespec="seconds"),
                "Deferred Until": expires_at.isoformat(timespec="seconds"),
            }
        )
    conn.commit()
    return records


def active_trade_deferrals(conn, owner: str, now=None) -> pd.DataFrame:
    """Return non-expired pauses and remove expired rows from the registry."""
    ensure_trade_deferral_schema(conn)
    current = _as_utc(now).isoformat(timespec="seconds")
    conn.execute(
        "DELETE FROM trade_deferrals WHERE owner = ? AND expires_at <= ?",
        (str(owner), current),
    )
    rows = conn.execute(
        """
        SELECT yahoo_ticker, symbol, requested_action, deferred_at, expires_at
        FROM trade_deferrals
        WHERE owner = ? AND expires_at > ?
        ORDER BY expires_at, yahoo_ticker
        """,
        (str(owner), current),
    ).fetchall()
    conn.commit()
    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=[
            "yahoo_ticker",
            "symbol",
            "requested_action",
            "deferred_at",
            "expires_at",
        ],
    )


def split_deferred_trades(plan: pd.DataFrame, deferrals: pd.DataFrame):
    """Split a plan without changing optimizer weights or recommendations."""
    if plan is None:
        return plan, pd.DataFrame()
    if plan.empty or deferrals is None or deferrals.empty:
        return plan.copy(), plan.iloc[0:0].copy()

    expiry_by_ticker = dict(
        zip(
            deferrals["yahoo_ticker"].astype(str).str.upper(),
            deferrals["expires_at"].astype(str),
        )
    )
    tickers = plan["Yahoo Ticker"].astype(str).str.upper()
    paused = tickers.isin(expiry_by_ticker)
    executable = plan.loc[~paused].copy().reset_index(drop=True)
    deferred = plan.loc[paused].copy().reset_index(drop=True)
    if not deferred.empty:
        deferred["Deferred Until"] = (
            deferred["Yahoo Ticker"].astype(str).str.upper().map(expiry_by_ticker)
        )
    return executable, deferred


def action_row_style(row):
    """Return a quiet full-row tint for Buy and Sell plan actions."""
    action = str(row.get("Action", "")).strip().casefold()
    if action == "buy":
        style = "background-color: rgba(46, 125, 50, 0.18);"
    elif action == "sell":
        style = "background-color: rgba(198, 40, 40, 0.18);"
    else:
        style = ""
    return [style] * len(row)


def exclude_deferred_candidates(candidates: pd.DataFrame, deferrals: pd.DataFrame):
    """Remove paused tickers from a universal candidate frame."""
    if candidates is None:
        return candidates, 0
    if candidates.empty or deferrals is None or deferrals.empty:
        return candidates.copy(), 0
    paused_tickers = set(deferrals["yahoo_ticker"].astype(str).str.upper())
    paused = (
        candidates["Yahoo Ticker"].fillna("").astype(str).str.upper()
        .isin(paused_tickers)
    )
    return candidates.loc[~paused].copy(), int(paused.sum())
