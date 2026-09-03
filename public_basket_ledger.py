from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_MARKET = "NSE_EQ"
REBALANCE_RULE = "FIRST_OPEN_SESSION_OF_CALENDAR_WEEK"


PUBLIC_BASKET_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS public_baskets (
    basket_id TEXT PRIMARY KEY,
    basket_name TEXT NOT NULL,
    calendar_market TEXT NOT NULL,
    rebalance_rule TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_calendar_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    range_start TEXT NOT NULL,
    range_end TEXT NOT NULL,
    source TEXT NOT NULL,
    source_ref TEXT,
    loaded_at TEXT NOT NULL,
    source_sha256 TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calendar_snapshot_market_range
ON market_calendar_snapshots (
    market,
    range_start,
    range_end,
    loaded_at
);

CREATE TABLE IF NOT EXISTS market_sessions (
    snapshot_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    is_open INTEGER NOT NULL CHECK (is_open IN (0, 1)),
    session_type TEXT NOT NULL DEFAULT 'NORMAL',
    notes TEXT,
    PRIMARY KEY (snapshot_id, session_date),
    FOREIGN KEY (snapshot_id)
        REFERENCES market_calendar_snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS signal_runs (
    signal_run_id TEXT PRIMARY KEY,
    basket_id TEXT NOT NULL,
    week_start_date TEXT NOT NULL,
    scheduled_session_date TEXT NOT NULL,
    generated_at TEXT NOT NULL,

    strategy_version TEXT NOT NULL,
    git_commit_sha TEXT,

    settings_json TEXT NOT NULL,
    portfolio_before_json TEXT NOT NULL,
    optimizer_output_json TEXT NOT NULL,
    signal_output_json TEXT NOT NULL,

    payload_sha256 TEXT NOT NULL,

    UNIQUE (basket_id, week_start_date),

    FOREIGN KEY (basket_id)
        REFERENCES public_baskets(basket_id)
);

CREATE TABLE IF NOT EXISTS weekly_rebalance_cycles (
    cycle_id TEXT PRIMARY KEY,
    basket_id TEXT NOT NULL,
    week_start_date TEXT NOT NULL,

    calendar_snapshot_id TEXT NOT NULL,
    scheduled_session_date TEXT,

    evaluated_at TEXT,

    status TEXT NOT NULL CHECK (
        status IN (
            'REBALANCED',
            'NO_CHANGE',
            'NO_OPEN_SESSION'
        )
    ),

    signal_run_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',

    UNIQUE (basket_id, week_start_date),

    FOREIGN KEY (basket_id)
        REFERENCES public_baskets(basket_id),

    FOREIGN KEY (calendar_snapshot_id)
        REFERENCES market_calendar_snapshots(snapshot_id),

    FOREIGN KEY (signal_run_id)
        REFERENCES signal_runs(signal_run_id)
);

CREATE TABLE IF NOT EXISTS rebalance_events (
    rebalance_id TEXT PRIMARY KEY,
    basket_id TEXT NOT NULL,
    signal_run_id TEXT NOT NULL,

    created_at TEXT NOT NULL,
    effective_session_date TEXT NOT NULL,

    model_status TEXT NOT NULL,
    rationale TEXT,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,

    UNIQUE (signal_run_id),

    FOREIGN KEY (basket_id)
        REFERENCES public_baskets(basket_id),

    FOREIGN KEY (signal_run_id)
        REFERENCES signal_runs(signal_run_id)
);

CREATE TABLE IF NOT EXISTS trade_orders (
    order_id TEXT PRIMARY KEY,
    rebalance_id TEXT NOT NULL,

    symbol TEXT NOT NULL,
    isin TEXT,

    action TEXT NOT NULL CHECK (
        action IN ('BUY', 'SELL', 'HOLD')
    ),

    current_weight REAL,
    target_weight REAL,

    signal_quantity REAL,
    executable_quantity REAL,

    signal_price REAL,
    expected_return_lift REAL,

    execution_rule TEXT NOT NULL,

    payload_sha256 TEXT NOT NULL,

    FOREIGN KEY (rebalance_id)
        REFERENCES rebalance_events(rebalance_id)
);

CREATE TABLE IF NOT EXISTS trade_executions (
    execution_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,

    executed_at TEXT NOT NULL,
    quantity REAL NOT NULL,

    market_price REAL NOT NULL,
    slippage_bps REAL NOT NULL DEFAULT 0,
    execution_price REAL NOT NULL,

    fees_inr REAL NOT NULL DEFAULT 0,
    cash_change_inr REAL NOT NULL,

    payload_sha256 TEXT NOT NULL,

    FOREIGN KEY (order_id)
        REFERENCES trade_orders(order_id)
);

CREATE TABLE IF NOT EXISTS daily_nav (
    basket_id TEXT NOT NULL,
    nav_date TEXT NOT NULL,

    calculation_version INTEGER NOT NULL,

    nav REAL NOT NULL,
    portfolio_value REAL NOT NULL,
    cash_value REAL NOT NULL,

    input_sha256 TEXT NOT NULL,
    calculated_at TEXT NOT NULL,

    PRIMARY KEY (
        basket_id,
        nav_date,
        calculation_version
    ),

    FOREIGN KEY (basket_id)
        REFERENCES public_baskets(basket_id)
);

CREATE TABLE IF NOT EXISTS public_basket_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,

    event_at TEXT NOT NULL,

    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,

    payload_json TEXT NOT NULL,

    previous_hash TEXT,
    event_hash TEXT NOT NULL
);


/* Core investment decisions are append-only. */

CREATE TRIGGER IF NOT EXISTS signal_runs_no_update
BEFORE UPDATE ON signal_runs
BEGIN
    SELECT RAISE(ABORT, 'signal_runs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS signal_runs_no_delete
BEFORE DELETE ON signal_runs
BEGIN
    SELECT RAISE(ABORT, 'signal_runs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS rebalance_events_no_update
BEFORE UPDATE ON rebalance_events
BEGIN
    SELECT RAISE(ABORT, 'rebalance_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS rebalance_events_no_delete
BEFORE DELETE ON rebalance_events
BEGIN
    SELECT RAISE(ABORT, 'rebalance_events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trade_orders_no_update
BEFORE UPDATE ON trade_orders
BEGIN
    SELECT RAISE(ABORT, 'trade_orders are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trade_orders_no_delete
BEFORE DELETE ON trade_orders
BEGIN
    SELECT RAISE(ABORT, 'trade_orders are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trade_executions_no_update
BEFORE UPDATE ON trade_executions
BEGIN
    SELECT RAISE(ABORT, 'trade_executions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trade_executions_no_delete
BEFORE DELETE ON trade_executions
BEGIN
    SELECT RAISE(ABORT, 'trade_executions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON public_basket_audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit log is immutable');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON public_basket_audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit log is immutable');
END;
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def week_start(value: date | str) -> date:
    value = _as_date(value)
    return value - timedelta(days=value.weekday())


def init_public_basket_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(PUBLIC_BASKET_DDL)
    conn.commit()


def _append_audit(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str,
    event_type: str,
    payload: Any,
) -> str:

    previous = conn.execute(
        """
        SELECT event_hash
        FROM public_basket_audit_log
        ORDER BY audit_id DESC
        LIMIT 1
        """
    ).fetchone()

    previous_hash = previous[0] if previous else ""

    canonical_payload = _canonical_json(
        {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "event_type": event_type,
            "payload": payload,
        }
    )

    event_hash = _sha256(
        previous_hash + "|" + canonical_payload
    )

    conn.execute(
        """
        INSERT INTO public_basket_audit_log (
            event_at,
            entity_type,
            entity_id,
            event_type,
            payload_json,
            previous_hash,
            event_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            entity_type,
            entity_id,
            event_type,
            canonical_payload,
            previous_hash or None,
            event_hash,
        ),
    )

    return event_hash


def create_public_basket(
    conn: sqlite3.Connection,
    basket_id: str,
    basket_name: str,
    strategy_version: str,
    calendar_market: str = DEFAULT_MARKET,
) -> None:

    existing = conn.execute(
        """
        SELECT basket_id
        FROM public_baskets
        WHERE basket_id = ?
        """,
        (basket_id,),
    ).fetchone()

    if existing:
        return

    conn.execute(
        """
        INSERT INTO public_baskets (
            basket_id,
            basket_name,
            calendar_market,
            rebalance_rule,
            strategy_version,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            basket_id,
            basket_name,
            calendar_market,
            REBALANCE_RULE,
            strategy_version,
            utc_now(),
        ),
    )

    _append_audit(
        conn,
        "basket",
        basket_id,
        "BASKET_CREATED",
        {
            "name": basket_name,
            "calendar_market": calendar_market,
            "rebalance_rule": REBALANCE_RULE,
            "strategy_version": strategy_version,
        },
    )

    conn.commit()


def build_calendar_rows(
    range_start: date,
    range_end: date,
    closed_dates: Iterable[date] = (),
    special_open_dates: Iterable[date] = (),
) -> list[dict]:

    closed = set(closed_dates)
    special_open = set(special_open_dates)

    rows = []
    current = range_start

    while current <= range_end:

        if current in special_open:
            is_open = True
            session_type = "SPECIAL"

        elif current.weekday() >= 5:
            is_open = False
            session_type = "WEEKEND"

        elif current in closed:
            is_open = False
            session_type = "EXCHANGE_HOLIDAY"

        else:
            is_open = True
            session_type = "NORMAL"

        rows.append(
            {
                "session_date": current.isoformat(),
                "is_open": is_open,
                "session_type": session_type,
                "notes": None,
            }
        )

        current += timedelta(days=1)

    return rows


def store_calendar_snapshot(
    conn: sqlite3.Connection,
    market: str,
    range_start: date,
    range_end: date,
    sessions: list[dict],
    source: str,
    source_ref: str | None = None,
) -> str:

    expected_dates = set()

    current = range_start
    while current <= range_end:
        expected_dates.add(current.isoformat())
        current += timedelta(days=1)

    provided_dates = {
        str(row["session_date"])
        for row in sessions
    }

    if provided_dates != expected_dates:
        missing = sorted(expected_dates - provided_dates)
        extra = sorted(provided_dates - expected_dates)

        raise ValueError(
            "Calendar snapshot must describe every calendar date. "
            f"Missing={missing}; extra={extra}"
        )

    source_payload = {
        "market": market,
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "source": source,
        "source_ref": source_ref,
        "sessions": sessions,
    }

    source_hash = _sha256(_canonical_json(source_payload))

    snapshot_id = (
        f"CAL-{market}-"
        f"{range_start:%Y%m%d}-"
        f"{range_end:%Y%m%d}-"
        f"{source_hash[:12]}"
    )

    existing = conn.execute(
        """
        SELECT snapshot_id
        FROM market_calendar_snapshots
        WHERE snapshot_id = ?
        """,
        (snapshot_id,),
    ).fetchone()

    if existing:
        return snapshot_id

    conn.execute(
        """
        INSERT INTO market_calendar_snapshots (
            snapshot_id,
            market,
            range_start,
            range_end,
            source,
            source_ref,
            loaded_at,
            source_sha256
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            market,
            range_start.isoformat(),
            range_end.isoformat(),
            source,
            source_ref,
            utc_now(),
            source_hash,
        ),
    )

    for row in sessions:
        conn.execute(
            """
            INSERT INTO market_sessions (
                snapshot_id,
                session_date,
                is_open,
                session_type,
                notes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                str(row["session_date"]),
                1 if row["is_open"] else 0,
                row.get("session_type", "NORMAL"),
                row.get("notes"),
            ),
        )

    _append_audit(
        conn,
        "calendar",
        snapshot_id,
        "CALENDAR_SNAPSHOT_STORED",
        source_payload,
    )

    conn.commit()

    return snapshot_id


def _calendar_snapshot_for_week(
    conn: sqlite3.Connection,
    market: str,
    monday: date,
):

    sunday = monday + timedelta(days=6)

    return conn.execute(
        """
        SELECT *
        FROM market_calendar_snapshots
        WHERE market = ?
          AND range_start <= ?
          AND range_end >= ?
        ORDER BY loaded_at DESC
        LIMIT 1
        """,
        (
            market,
            monday.isoformat(),
            sunday.isoformat(),
        ),
    ).fetchone()


def resolve_first_trading_day(
    conn: sqlite3.Connection,
    basket_id: str,
    reference_date: date,
) -> dict:

    monday = week_start(reference_date)
    sunday = monday + timedelta(days=6)

    basket = conn.execute(
        """
        SELECT calendar_market
        FROM public_baskets
        WHERE basket_id = ?
        """,
        (basket_id,),
    ).fetchone()

    if basket is None:
        raise ValueError(f"Unknown basket: {basket_id}")

    market = basket[0]

    snapshot = _calendar_snapshot_for_week(
        conn,
        market,
        monday,
    )

    if snapshot is None:
        return {
            "status": "CALENDAR_INCOMPLETE",
            "week_start": monday,
            "first_trading_day": None,
            "snapshot_id": None,
        }

    first_open = conn.execute(
        """
        SELECT session_date
        FROM market_sessions
        WHERE snapshot_id = ?
          AND session_date >= ?
          AND session_date <= ?
          AND is_open = 1
        ORDER BY session_date
        LIMIT 1
        """,
        (
            snapshot["snapshot_id"],
            monday.isoformat(),
            sunday.isoformat(),
        ),
    ).fetchone()

    if first_open is None:
        return {
            "status": "NO_OPEN_SESSION",
            "week_start": monday,
            "first_trading_day": None,
            "snapshot_id": snapshot["snapshot_id"],
        }

    return {
        "status": "RESOLVED",
        "week_start": monday,
        "first_trading_day": date.fromisoformat(first_open[0]),
        "snapshot_id": snapshot["snapshot_id"],
    }


def rebalance_gate(
    conn: sqlite3.Connection,
    basket_id: str,
    today: date,
) -> dict:

    monday = week_start(today)

    existing = conn.execute(
        """
        SELECT status, scheduled_session_date, signal_run_id
        FROM weekly_rebalance_cycles
        WHERE basket_id = ?
          AND week_start_date = ?
        """,
        (
            basket_id,
            monday.isoformat(),
        ),
    ).fetchone()

    if existing:
        return {
            "status": "ALREADY_EVALUATED",
            "week_start": monday,
            "cycle_status": existing["status"],
            "scheduled_session_date": existing[
                "scheduled_session_date"
            ],
            "signal_run_id": existing["signal_run_id"],
        }

    schedule = resolve_first_trading_day(
        conn,
        basket_id,
        today,
    )

    if schedule["status"] != "RESOLVED":
        return schedule

    scheduled = schedule["first_trading_day"]

    if today != scheduled:
        return {
            **schedule,
            "status": "NOT_DUE",
        }

    return {
        **schedule,
        "status": "DUE",
    }


def record_weekly_signal(
    conn: sqlite3.Connection,
    basket_id: str,
    today: date,
    strategy_version: str,
    git_commit_sha: str | None,
    settings: dict,
    portfolio_before: dict | list,
    optimizer_output: dict | list,
    signal_output: dict | list,
    decision_status: str,
) -> str:

    if decision_status not in {
        "REBALANCED",
        "NO_CHANGE",
    }:
        raise ValueError(
            "decision_status must be REBALANCED or NO_CHANGE"
        )

    gate = rebalance_gate(
        conn,
        basket_id,
        today,
    )

    if gate["status"] == "ALREADY_EVALUATED":
        return gate["signal_run_id"]

    if gate["status"] != "DUE":
        raise RuntimeError(
            f"Weekly signal cannot run: {gate['status']}"
        )

    week = gate["week_start"]

    payload = {
        "basket_id": basket_id,
        "week_start_date": week.isoformat(),
        "scheduled_session_date":
            gate["first_trading_day"].isoformat(),
        "strategy_version": strategy_version,
        "git_commit_sha": git_commit_sha,
        "settings": settings,
        "portfolio_before": portfolio_before,
        "optimizer_output": optimizer_output,
        "signal_output": signal_output,
    }

    payload_hash = _sha256(
        _canonical_json(payload)
    )

    signal_run_id = (
        f"SIG-{week:%Y%m%d}-{payload_hash[:12]}"
    )

    generated_at = utc_now()

    try:
        conn.execute("BEGIN")

        conn.execute(
            """
            INSERT INTO signal_runs (
                signal_run_id,
                basket_id,
                week_start_date,
                scheduled_session_date,
                generated_at,
                strategy_version,
                git_commit_sha,
                settings_json,
                portfolio_before_json,
                optimizer_output_json,
                signal_output_json,
                payload_sha256
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_run_id,
                basket_id,
                week.isoformat(),
                gate["first_trading_day"].isoformat(),
                generated_at,
                strategy_version,
                git_commit_sha,
                _canonical_json(settings),
                _canonical_json(portfolio_before),
                _canonical_json(optimizer_output),
                _canonical_json(signal_output),
                payload_hash,
            ),
        )

        cycle_id = (
            f"CYCLE-{basket_id}-{week:%Y%m%d}"
        )

        conn.execute(
            """
            INSERT INTO weekly_rebalance_cycles (
                cycle_id,
                basket_id,
                week_start_date,
                calendar_snapshot_id,
                scheduled_session_date,
                evaluated_at,
                status,
                signal_run_id,
                details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cycle_id,
                basket_id,
                week.isoformat(),
                gate["snapshot_id"],
                gate["first_trading_day"].isoformat(),
                generated_at,
                decision_status,
                signal_run_id,
                _canonical_json(
                    {
                        "payload_sha256": payload_hash,
                    }
                ),
            ),
        )

        _append_audit(
            conn,
            "signal_run",
            signal_run_id,
            decision_status,
            payload,
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    return signal_run_id
