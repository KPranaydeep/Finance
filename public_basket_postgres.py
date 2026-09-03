from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from nse_market_calendar import (
    MARKET,
    SOURCE_AS_OF,
    SOURCE_NAME,
    SOURCE_REF,
    calendar_snapshot_payload_2026,
)


PUBLIC_BASKET_SCHEMA_VERSION = 1

DEFAULT_BASKET_ID = "PRANAYDEEP-PUBLIC-01"
DEFAULT_BASKET_NAME = "Public Dynamic Portfolio"

DEFAULT_STRATEGY_VERSION = "portfolio-rebalancer-v1"

REBALANCE_RULE = "FIRST_OPEN_SESSION_OF_CALENDAR_WEEK"

# Used only to serialize creation of chained audit hashes.
AUDIT_ADVISORY_LOCK = 9485217


# =====================================================================
# DATABASE CONFIGURATION
# =====================================================================


def get_public_basket_database_url() -> str | None:
    """
    Resolve the durable PostgreSQL URL.

    Resolution order:

    1. Environment variable:
       PUBLIC_BASKET_DATABASE_URL

    2. Streamlit secrets:
       [public_basket]
       database_url = "postgresql://..."

    No SQLite fallback is allowed here.

    This is deliberate: production public records must never silently
    fall back to Streamlit's temporary filesystem.
    """

    env_url = os.getenv("PUBLIC_BASKET_DATABASE_URL")

    if env_url:
        return env_url.strip()

    try:
        import streamlit as st

        section = st.secrets.get("public_basket", {})

        database_url = section.get("database_url")

        if database_url:
            return str(database_url).strip()

    except Exception:
        pass

    return None


def connect_public_basket_db(
    database_url: str | None = None,
):
    """
    Open the durable PostgreSQL connection.

    autocommit=True means all mutations below explicitly create
    transactions, avoiding accidental long-running Streamlit
    transactions.
    """

    database_url = (
        database_url
        or get_public_basket_database_url()
    )

    if not database_url:
        raise RuntimeError(
            "PUBLIC BASKET DATABASE IS NOT CONFIGURED. "
            "Add PUBLIC_BASKET_DATABASE_URL or Streamlit "
            "secret [public_basket].database_url. "
            "The application intentionally refuses to use "
            "local SQLite for authoritative public records."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        autocommit=True,
    )


# =====================================================================
# HELPERS
# =====================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_text(value: str | bytes) -> str:

    if isinstance(value, str):
        value = value.encode("utf-8")

    return hashlib.sha256(value).hexdigest()


def week_start(reference_date: date) -> date:
    return (
        reference_date
        - timedelta(days=reference_date.weekday())
    )


def week_end(reference_date: date) -> date:
    return week_start(reference_date) + timedelta(days=6)


# =====================================================================
# POSTGRESQL SCHEMA
# =====================================================================


SCHEMA_STATEMENTS = (

    """
    CREATE TABLE IF NOT EXISTS public_baskets (
        basket_id TEXT PRIMARY KEY,

        basket_name TEXT NOT NULL,

        calendar_market TEXT NOT NULL,

        rebalance_rule TEXT NOT NULL,

        strategy_version TEXT NOT NULL,

        schema_version INTEGER NOT NULL,

        created_at TIMESTAMPTZ NOT NULL
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS market_calendar_snapshots (
        snapshot_id TEXT PRIMARY KEY,

        market TEXT NOT NULL,

        range_start DATE NOT NULL,

        range_end DATE NOT NULL,

        source TEXT NOT NULL,

        source_ref TEXT,

        source_as_of DATE,

        loaded_at TIMESTAMPTZ NOT NULL,

        source_sha256 TEXT NOT NULL
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_public_calendar_market_range

    ON market_calendar_snapshots (
        market,
        range_start,
        range_end,
        loaded_at
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS market_sessions (
        snapshot_id TEXT NOT NULL,

        session_date DATE NOT NULL,

        is_open BOOLEAN NOT NULL,

        session_type TEXT NOT NULL,

        notes TEXT,

        PRIMARY KEY (
            snapshot_id,
            session_date
        ),

        FOREIGN KEY (snapshot_id)
            REFERENCES market_calendar_snapshots(snapshot_id)
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_public_market_sessions_date

    ON market_sessions (
        session_date,
        is_open
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS signal_runs (
        signal_run_id TEXT PRIMARY KEY,

        basket_id TEXT NOT NULL,

        week_start_date DATE NOT NULL,

        scheduled_session_date DATE NOT NULL,

        generated_at TIMESTAMPTZ NOT NULL,

        strategy_version TEXT NOT NULL,

        git_commit_sha TEXT,

        settings_json JSONB NOT NULL,

        portfolio_before_json JSONB NOT NULL,

        optimizer_output_json JSONB NOT NULL,

        signal_output_json JSONB NOT NULL,

        payload_sha256 TEXT NOT NULL,

        UNIQUE (
            basket_id,
            week_start_date
        ),

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS weekly_rebalance_cycles (
        cycle_id TEXT PRIMARY KEY,

        basket_id TEXT NOT NULL,

        week_start_date DATE NOT NULL,

        calendar_snapshot_id TEXT NOT NULL,

        scheduled_session_date DATE,

        evaluated_at TIMESTAMPTZ,

        status TEXT NOT NULL CHECK (
            status IN (
                'REBALANCED',
                'NO_CHANGE',
                'NO_OPEN_SESSION'
            )
        ),

        signal_run_id TEXT,

        details_json JSONB NOT NULL
            DEFAULT '{}'::jsonb,

        UNIQUE (
            basket_id,
            week_start_date
        ),

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id),

        FOREIGN KEY (calendar_snapshot_id)
            REFERENCES market_calendar_snapshots(snapshot_id),

        FOREIGN KEY (signal_run_id)
            REFERENCES signal_runs(signal_run_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS rebalance_events (
        rebalance_id TEXT PRIMARY KEY,

        basket_id TEXT NOT NULL,

        signal_run_id TEXT NOT NULL UNIQUE,

        created_at TIMESTAMPTZ NOT NULL,

        effective_session_date DATE NOT NULL,

        model_status TEXT NOT NULL,

        rationale TEXT,

        payload_json JSONB NOT NULL,

        payload_sha256 TEXT NOT NULL,

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id),

        FOREIGN KEY (signal_run_id)
            REFERENCES signal_runs(signal_run_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS trade_orders (
        order_id TEXT PRIMARY KEY,

        rebalance_id TEXT NOT NULL,

        symbol TEXT NOT NULL,

        isin TEXT,

        action TEXT NOT NULL CHECK (
            action IN (
                'BUY',
                'SELL',
                'HOLD'
            )
        ),

        current_weight DOUBLE PRECISION,

        target_weight DOUBLE PRECISION,

        signal_quantity DOUBLE PRECISION,

        executable_quantity DOUBLE PRECISION,

        signal_price DOUBLE PRECISION,

        expected_return_lift DOUBLE PRECISION,

        execution_rule TEXT NOT NULL,

        payload_sha256 TEXT NOT NULL,

        FOREIGN KEY (rebalance_id)
            REFERENCES rebalance_events(rebalance_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS trade_executions (
        execution_id TEXT PRIMARY KEY,

        order_id TEXT NOT NULL,

        executed_at TIMESTAMPTZ NOT NULL,

        quantity DOUBLE PRECISION NOT NULL,

        market_price DOUBLE PRECISION NOT NULL,

        slippage_bps DOUBLE PRECISION NOT NULL
            DEFAULT 0,

        execution_price DOUBLE PRECISION NOT NULL,

        fees_inr DOUBLE PRECISION NOT NULL
            DEFAULT 0,

        cash_change_inr DOUBLE PRECISION NOT NULL,

        payload_sha256 TEXT NOT NULL,

        FOREIGN KEY (order_id)
            REFERENCES trade_orders(order_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS daily_nav (
        basket_id TEXT NOT NULL,

        nav_date DATE NOT NULL,

        calculation_version INTEGER NOT NULL,

        nav DOUBLE PRECISION NOT NULL,

        portfolio_value DOUBLE PRECISION NOT NULL,

        cash_value DOUBLE PRECISION NOT NULL,

        input_sha256 TEXT NOT NULL,

        calculated_at TIMESTAMPTZ NOT NULL,

        PRIMARY KEY (
            basket_id,
            nav_date,
            calculation_version
        ),

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS public_basket_audit_log (
        audit_id BIGSERIAL PRIMARY KEY,

        event_at TIMESTAMPTZ NOT NULL,

        entity_type TEXT NOT NULL,

        entity_id TEXT NOT NULL,

        event_type TEXT NOT NULL,

        payload_json JSONB NOT NULL,

        previous_hash TEXT,

        event_hash TEXT NOT NULL UNIQUE
    )
    """,

    """
    CREATE OR REPLACE FUNCTION
        reject_public_basket_mutation()

    RETURNS TRIGGER
    AS $$
    BEGIN

        RAISE EXCEPTION
            'Table % is append-only and immutable',
            TG_TABLE_NAME;

    END;
    $$
    LANGUAGE plpgsql
    """,

    """
    DO $$
    DECLARE

        table_name TEXT;

    BEGIN

        FOREACH table_name IN ARRAY ARRAY[
            'market_calendar_snapshots',
            'market_sessions',
            'signal_runs',
            'weekly_rebalance_cycles',
            'rebalance_events',
            'trade_orders',
            'trade_executions',
            'daily_nav',
            'public_basket_audit_log'
        ]

        LOOP

            IF NOT EXISTS (

                SELECT 1

                FROM pg_trigger

                WHERE tgname = 'immutable_guard'

                AND tgrelid =
                    to_regclass(table_name)

            ) THEN

                EXECUTE format(
                    'CREATE TRIGGER immutable_guard
                     BEFORE UPDATE OR DELETE
                     ON %I
                     FOR EACH ROW
                     EXECUTE FUNCTION
                     reject_public_basket_mutation()',
                    table_name
                );

            END IF;

        END LOOP;

    END
    $$
    """,
)


def init_public_basket_schema(conn) -> None:
    """
    Create the durable public-basket PostgreSQL schema.

    This is additive/idempotent.
    """

    with conn.transaction():

        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


# =====================================================================
# AUDIT CHAIN
# =====================================================================


def append_audit(
    conn,
    entity_type: str,
    entity_id: str,
    event_type: str,
    payload: Any,
) -> str:
    """
    Append a SHA-256 chained audit event.

    Call only from inside an active transaction.
    """

    # Prevent two concurrent sessions from creating competing
    # "previous hash" values.
    conn.execute(
        "SELECT pg_advisory_xact_lock(%s)",
        (AUDIT_ADVISORY_LOCK,),
    )

    previous = conn.execute(
        """
        SELECT event_hash

        FROM public_basket_audit_log

        ORDER BY audit_id DESC

        LIMIT 1
        """
    ).fetchone()

    previous_hash = (
        previous["event_hash"]
        if previous
        else ""
    )

    audit_payload = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "event_type": event_type,
        "payload": payload,
    }

    canonical = canonical_json(
        audit_payload
    )

    event_hash = sha256_text(
        previous_hash
        + "|"
        + canonical
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

        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        """,
        (
            utc_now(),
            entity_type,
            entity_id,
            event_type,
            Jsonb(audit_payload),
            previous_hash or None,
            event_hash,
        ),
    )

    return event_hash


# =====================================================================
# PUBLIC BASKET
# =====================================================================


def create_public_basket(
    conn,
    basket_id: str = DEFAULT_BASKET_ID,
    basket_name: str = DEFAULT_BASKET_NAME,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> bool:
    """
    Create the model basket once.

    Returns:
        True  -> created now
        False -> already existed
    """

    with conn.transaction():

        row = conn.execute(
            """
            INSERT INTO public_baskets (
                basket_id,
                basket_name,
                calendar_market,
                rebalance_rule,
                strategy_version,
                schema_version,
                created_at
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )

            ON CONFLICT (basket_id)
            DO NOTHING

            RETURNING basket_id
            """,
            (
                basket_id,
                basket_name,
                MARKET,
                REBALANCE_RULE,
                strategy_version,
                PUBLIC_BASKET_SCHEMA_VERSION,
                utc_now(),
            ),
        ).fetchone()

        if row is None:
            return False

        append_audit(
            conn=conn,
            entity_type="basket",
            entity_id=basket_id,
            event_type="BASKET_CREATED",
            payload={
                "basket_name": basket_name,
                "market": MARKET,
                "rebalance_rule": REBALANCE_RULE,
                "strategy_version": strategy_version,
            },
        )

        return True


# =====================================================================
# MARKET CALENDAR
# =====================================================================


def store_calendar_snapshot(
    conn,
    payload: dict,
) -> str:
    """
    Store one immutable exchange-calendar snapshot.
    """

    canonical = canonical_json(
        payload
    )

    source_hash = sha256_text(
        canonical
    )

    range_start = date.fromisoformat(
        payload["range_start"]
    )

    range_end = date.fromisoformat(
        payload["range_end"]
    )

    snapshot_id = (
        f"CAL-{payload['market']}-"
        f"{range_start:%Y%m%d}-"
        f"{range_end:%Y%m%d}-"
        f"{source_hash[:12]}"
    )

    with conn.transaction():

        existing = conn.execute(
            """
            SELECT snapshot_id

            FROM market_calendar_snapshots

            WHERE snapshot_id = %s
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
                source_as_of,
                loaded_at,
                source_sha256
            )

            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                snapshot_id,
                payload["market"],
                range_start,
                range_end,
                payload["source"],
                payload.get("source_ref"),
                date.fromisoformat(
                    payload["source_as_of"]
                ),
                utc_now(),
                source_hash,
            ),
        )

        with conn.cursor() as cursor:

            cursor.executemany(
                """
                INSERT INTO market_sessions (
                    snapshot_id,
                    session_date,
                    is_open,
                    session_type,
                    notes
                )

                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                [
                    (
                        snapshot_id,
                        date.fromisoformat(
                            row["session_date"]
                        ),
                        bool(row["is_open"]),
                        row["session_type"],
                        row.get("notes"),
                    )
                    for row
                    in payload["sessions"]
                ],
            )

        append_audit(
            conn=conn,
            entity_type="calendar",
            entity_id=snapshot_id,
            event_type="CALENDAR_SNAPSHOT_STORED",
            payload={
                "market": payload["market"],
                "range_start": payload[
                    "range_start"
                ],
                "range_end": payload[
                    "range_end"
                ],
                "source": payload["source"],
                "source_ref": payload.get(
                    "source_ref"
                ),
                "source_as_of": payload.get(
                    "source_as_of"
                ),
                "source_sha256": source_hash,
            },
        )

    return snapshot_id


def seed_nse_2026_calendar(
    conn,
) -> str:
    """
    Store the bundled verified 2026 NSE equity calendar.

    Safe to call repeatedly.
    """

    return store_calendar_snapshot(
        conn,
        calendar_snapshot_payload_2026(),
    )


def get_calendar_snapshot_for_week(
    conn,
    market: str,
    monday: date,
):
    """
    Return the newest calendar snapshot that fully contains
    Monday-Sunday.
    """

    sunday = monday + timedelta(days=6)

    return conn.execute(
        """
        SELECT *

        FROM market_calendar_snapshots

        WHERE market = %s

          AND range_start <= %s

          AND range_end >= %s

        ORDER BY loaded_at DESC

        LIMIT 1
        """,
        (
            market,
            monday,
            sunday,
        ),
    ).fetchone()


# =====================================================================
# WEEKLY SCHEDULER
# =====================================================================


def resolve_first_trading_day(
    conn,
    basket_id: str,
    reference_date: date,
) -> dict:
    """
    Resolve first actual open market session of the week.

    The calendar snapshot used is returned so a future signal
    can permanently reference it.
    """

    monday = week_start(
        reference_date
    )

    sunday = monday + timedelta(
        days=6
    )

    basket = conn.execute(
        """
        SELECT
            basket_id,
            calendar_market,
            rebalance_rule,
            strategy_version

        FROM public_baskets

        WHERE basket_id = %s
        """,
        (basket_id,),
    ).fetchone()

    if basket is None:
        raise ValueError(
            f"Unknown public basket: {basket_id}"
        )

    snapshot = get_calendar_snapshot_for_week(
        conn=conn,
        market=basket["calendar_market"],
        monday=monday,
    )

    if snapshot is None:

        return {
            "status": "CALENDAR_INCOMPLETE",
            "week_start": monday,
            "week_end": sunday,
            "first_trading_day": None,
            "snapshot_id": None,
        }

    first_open = conn.execute(
        """
        SELECT
            session_date,
            session_type,
            notes

        FROM market_sessions

        WHERE snapshot_id = %s

          AND session_date >= %s

          AND session_date <= %s

          AND is_open = TRUE

        ORDER BY session_date

        LIMIT 1
        """,
        (
            snapshot["snapshot_id"],
            monday,
            sunday,
        ),
    ).fetchone()

    if first_open is None:

        return {
            "status": "NO_OPEN_SESSION",
            "week_start": monday,
            "week_end": sunday,
            "first_trading_day": None,
            "snapshot_id": snapshot[
                "snapshot_id"
            ],
        }

    return {
        "status": "RESOLVED",

        "week_start": monday,
        "week_end": sunday,

        "first_trading_day":
            first_open["session_date"],

        "session_type":
            first_open["session_type"],

        "session_notes":
            first_open["notes"],

        "snapshot_id":
            snapshot["snapshot_id"],
    }


def rebalance_gate(
    conn,
    basket_id: str,
    today: date,
) -> dict:
    """
    Determine whether an official public-basket optimization is
    allowed today.

    Possible states:

    DUE
        Today is the first NSE trading session of the week.

    NOT_DUE
        The week's first market session is still in the future.

    MISSED
        The first market session already passed and the public
        basket was not evaluated. We DO NOT catch up later in the
        week because that would violate the strategy.

    ALREADY_EVALUATED
        An official weekly cycle already exists.

    CALENDAR_INCOMPLETE
        No verified exchange calendar exists.

    NO_OPEN_SESSION
        No trading session exists during the entire week.
    """

    monday = week_start(
        today
    )

    existing = conn.execute(
        """
        SELECT
            cycle_id,
            status,
            scheduled_session_date,
            signal_run_id,
            evaluated_at

        FROM weekly_rebalance_cycles

        WHERE basket_id = %s

          AND week_start_date = %s
        """,
        (
            basket_id,
            monday,
        ),
    ).fetchone()

    if existing:

        return {
            "status": "ALREADY_EVALUATED",

            "week_start": monday,

            "cycle_status":
                existing["status"],

            "scheduled_session_date":
                existing[
                    "scheduled_session_date"
                ],

            "signal_run_id":
                existing["signal_run_id"],

            "evaluated_at":
                existing["evaluated_at"],
        }

    schedule = resolve_first_trading_day(
        conn=conn,
        basket_id=basket_id,
        reference_date=today,
    )

    if schedule["status"] != "RESOLVED":
        return schedule

    scheduled = schedule[
        "first_trading_day"
    ]

    if today < scheduled:

        return {
            **schedule,
            "status": "NOT_DUE",
        }

    if today > scheduled:

        return {
            **schedule,
            "status": "MISSED",
        }

    return {
        **schedule,
        "status": "DUE",
    }


# =====================================================================
# STATUS / UI HELPERS
# =====================================================================


def list_week_sessions(
    conn,
    basket_id: str,
    reference_date: date,
) -> list[dict]:
    """
    Return Monday-Sunday market state for display.
    """

    schedule = resolve_first_trading_day(
        conn=conn,
        basket_id=basket_id,
        reference_date=reference_date,
    )

    snapshot_id = schedule.get(
        "snapshot_id"
    )

    if snapshot_id is None:
        return []

    monday = week_start(
        reference_date
    )

    sunday = monday + timedelta(
        days=6
    )

    rows = conn.execute(
        """
        SELECT
            session_date,
            is_open,
            session_type,
            notes

        FROM market_sessions

        WHERE snapshot_id = %s

          AND session_date >= %s

          AND session_date <= %s

        ORDER BY session_date
        """,
        (
            snapshot_id,
            monday,
            sunday,
        ),
    ).fetchall()

    return [dict(row) for row in rows]


def get_basket_record(
    conn,
    basket_id: str,
):
    return conn.execute(
        """
        SELECT *

        FROM public_baskets

        WHERE basket_id = %s
        """,
        (basket_id,),
    ).fetchone()


def public_database_healthcheck(
    conn,
) -> dict:

    result = conn.execute(
        """
        SELECT
            current_database() AS database_name,
            NOW() AS database_time
        """
    ).fetchone()

    return dict(result)
