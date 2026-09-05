from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


# =====================================================================
# CONFIGURATION
# =====================================================================

PUBLIC_BASKET_SCHEMA_VERSION = 2

DEFAULT_BASKET_ID = "PUBLIC-01"
DEFAULT_BASKET_NAME = "Public Dynamic Portfolio"
DEFAULT_STRATEGY_VERSION = "portfolio-rebalancer-v1"

AUDIT_ADVISORY_LOCK = 9485217


# =====================================================================
# TIME / HASH HELPERS
# =====================================================================


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    """
    Deterministic JSON representation used for hashing.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )


def sha256_text(value: str | bytes) -> str:
    """
    SHA-256 helper.
    """
    if isinstance(value, str):
        value = value.encode("utf-8")

    return hashlib.sha256(value).hexdigest()


def new_id(prefix: str) -> str:
    """
    Generate globally unique immutable event identifier.
    """
    return f"{prefix}-{uuid.uuid4().hex}"


# =====================================================================
# DATABASE CONFIGURATION
# =====================================================================


def get_public_basket_database_url() -> Optional[str]:
    """
    Resolve durable PostgreSQL URL.

    Order:
    1. PUBLIC_BASKET_DATABASE_URL environment variable
    2. Streamlit secrets:
         [public_basket]
         database_url = "postgresql://..."

    There is intentionally NO local-file or SQLite fallback.
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
    database_url: Optional[str] = None,
):
    """
    Connect to durable PostgreSQL.

    Fail closed if database configuration is unavailable.
    """

    database_url = (
        database_url
        or get_public_basket_database_url()
    )

    if not database_url:
        raise RuntimeError(
            "Public basket PostgreSQL is not configured. "
            "Set PUBLIC_BASKET_DATABASE_URL or "
            "[public_basket].database_url in Streamlit secrets."
        )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
        autocommit=True,
    )


# =====================================================================
# DATABASE SCHEMA
# =====================================================================


SCHEMA_STATEMENTS = (

    """
    CREATE TABLE IF NOT EXISTS public_baskets (
        basket_id TEXT PRIMARY KEY,
        basket_name TEXT NOT NULL,
        base_currency TEXT NOT NULL DEFAULT 'INR',
        strategy_version TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        status TEXT NOT NULL DEFAULT 'ACTIVE'
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS signal_runs (
        signal_run_id TEXT PRIMARY KEY,

        basket_id TEXT NOT NULL,

        generated_at TIMESTAMPTZ NOT NULL,

        data_as_of TIMESTAMPTZ,

        strategy_version TEXT NOT NULL,

        git_commit_sha TEXT,

        input_snapshot_sha256 TEXT,

        settings_json JSONB NOT NULL,

        portfolio_before_json JSONB NOT NULL,

        optimizer_output_json JSONB NOT NULL,

        signal_output_json JSONB NOT NULL,

        decision_status TEXT NOT NULL CHECK (
            decision_status IN (
                'REBALANCED',
                'NO_CHANGE'
            )
        ),

        payload_sha256 TEXT NOT NULL,

        created_at TIMESTAMPTZ NOT NULL,

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id)
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_signal_runs_basket_generated

    ON signal_runs (
        basket_id,
        generated_at DESC
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS rebalance_events (
        rebalance_id TEXT PRIMARY KEY,

        basket_id TEXT NOT NULL,

        signal_run_id TEXT NOT NULL UNIQUE,

        created_at TIMESTAMPTZ NOT NULL,

        effective_at TIMESTAMPTZ,

        status TEXT NOT NULL CHECK (
            status IN (
                'CREATED',
                'APPROVED',
                'CANCELLED',
                'COMPLETED'
            )
        ),

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
    CREATE INDEX IF NOT EXISTS
        idx_rebalance_events_basket_created

    ON rebalance_events (
        basket_id,
        created_at DESC
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS trade_orders (
        order_id TEXT PRIMARY KEY,

        rebalance_id TEXT NOT NULL,

        created_at TIMESTAMPTZ NOT NULL,

        symbol TEXT NOT NULL,

        yahoo_ticker TEXT,

        isin TEXT,

        side TEXT NOT NULL CHECK (
            side IN (
                'BUY',
                'SELL'
            )
        ),

        current_weight DOUBLE PRECISION,

        target_weight DOUBLE PRECISION,

        theoretical_quantity DOUBLE PRECISION,

        requested_quantity DOUBLE PRECISION NOT NULL,

        reference_price DOUBLE PRECISION,

        execution_rule TEXT,

        order_status TEXT NOT NULL DEFAULT 'CREATED' CHECK (
            order_status IN (
                'CREATED',
                'PENDING',
                'PARTIALLY_FILLED',
                'FILLED',
                'CANCELLED',
                'REJECTED'
            )
        ),

        payload_json JSONB NOT NULL,

        payload_sha256 TEXT NOT NULL,

        FOREIGN KEY (rebalance_id)
            REFERENCES rebalance_events(rebalance_id)
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_trade_orders_rebalance

    ON trade_orders (
        rebalance_id,
        created_at
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS trade_executions (
        execution_id TEXT PRIMARY KEY,

        order_id TEXT NOT NULL,

        executed_at TIMESTAMPTZ NOT NULL,

        quantity DOUBLE PRECISION NOT NULL,

        market_price DOUBLE PRECISION,

        execution_price DOUBLE PRECISION NOT NULL,

        fees_inr DOUBLE PRECISION NOT NULL DEFAULT 0,

        taxes_inr DOUBLE PRECISION NOT NULL DEFAULT 0,

        slippage_bps DOUBLE PRECISION,

        cash_change_inr DOUBLE PRECISION NOT NULL,

        payload_json JSONB NOT NULL,

        payload_sha256 TEXT NOT NULL,

        FOREIGN KEY (order_id)
            REFERENCES trade_orders(order_id)
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_trade_executions_order

    ON trade_executions (
        order_id,
        executed_at
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS cash_ledger (
        cash_event_id TEXT PRIMARY KEY,

        basket_id TEXT NOT NULL,

        event_at TIMESTAMPTZ NOT NULL,

        event_type TEXT NOT NULL,

        amount_inr DOUBLE PRECISION NOT NULL,

        reference_type TEXT,

        reference_id TEXT,

        notes TEXT,

        payload_sha256 TEXT NOT NULL,

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id)
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS
        idx_cash_ledger_basket

    ON cash_ledger (
        basket_id,
        event_at
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS daily_positions (
        basket_id TEXT NOT NULL,

        position_date DATE NOT NULL,

        symbol TEXT NOT NULL,

        quantity DOUBLE PRECISION NOT NULL,

        close_price DOUBLE PRECISION NOT NULL,

        market_value DOUBLE PRECISION NOT NULL,

        weight DOUBLE PRECISION,

        calculation_version INTEGER NOT NULL DEFAULT 1,

        input_sha256 TEXT NOT NULL,

        calculated_at TIMESTAMPTZ NOT NULL,

        PRIMARY KEY (
            basket_id,
            position_date,
            symbol,
            calculation_version
        ),

        FOREIGN KEY (basket_id)
            REFERENCES public_baskets(basket_id)
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

        total_value DOUBLE PRECISION NOT NULL,

        daily_return DOUBLE PRECISION,

          drawdown DOUBLE PRECISION,

          gross_nav DOUBLE PRECISION,

          net_nav DOUBLE PRECISION,

          gross_daily_return DOUBLE PRECISION,

          turnover DOUBLE PRECISION NOT NULL DEFAULT 0,

          estimated_drag DOUBLE PRECISION NOT NULL DEFAULT 0,

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

    "ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS gross_nav DOUBLE PRECISION",
    "ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS net_nav DOUBLE PRECISION",
    "ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS gross_daily_return DOUBLE PRECISION",
    "ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS turnover DOUBLE PRECISION NOT NULL DEFAULT 0",
    "ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS estimated_drag DOUBLE PRECISION NOT NULL DEFAULT 0",

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
    CREATE OR REPLACE FUNCTION reject_public_basket_mutation()
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
        trigger_name TEXT;
    BEGIN

        FOREACH table_name IN ARRAY ARRAY[
            'signal_runs',
            'rebalance_events',
            'trade_orders',
            'trade_executions',
            'cash_ledger',
            'public_basket_audit_log'
        ]

        LOOP

            trigger_name :=
                'immutable_guard_' || table_name;

            IF NOT EXISTS (
                SELECT 1
                FROM pg_trigger
                WHERE tgname = trigger_name
            )
            THEN

                EXECUTE format(
                    'CREATE TRIGGER %I
                     BEFORE UPDATE OR DELETE
                     ON %I
                     FOR EACH ROW
                     EXECUTE FUNCTION
                     reject_public_basket_mutation()',
                    trigger_name,
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
    Initialize durable public-basket schema.

    Safe to run repeatedly.
    """

    with conn.transaction():

        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


# =====================================================================
# AUDIT LOG
# =====================================================================


def append_audit(
    conn,
    entity_type: str,
    entity_id: str,
    event_type: str,
    payload: Any,
) -> str:
    """
    Append tamper-evident SHA-256 chained audit record.

    Must be called inside an active transaction.
    """

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

    event_hash = sha256_text(
        previous_hash
        + "|"
        + canonical_json(audit_payload)
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
# BASKET
# =====================================================================


def create_public_basket(
    conn,
    basket_id: str = DEFAULT_BASKET_ID,
    basket_name: str = DEFAULT_BASKET_NAME,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> bool:
    """
    Create public basket if absent.

    Returns:
        True  -> newly created
        False -> already existed
    """

    with conn.transaction():

        row = conn.execute(
            """
            INSERT INTO public_baskets (
                basket_id,
                basket_name,
                base_currency,
                strategy_version,
                schema_version,
                created_at,
                status
            )

            VALUES (
                %s,
                %s,
                'INR',
                %s,
                %s,
                %s,
                'ACTIVE'
            )

            ON CONFLICT (basket_id)
            DO NOTHING

            RETURNING basket_id
            """,
            (
                basket_id,
                basket_name,
                strategy_version,
                PUBLIC_BASKET_SCHEMA_VERSION,
                utc_now(),
            ),
        ).fetchone()

        if row is None:
            return False

        append_audit(
            conn,
            entity_type="basket",
            entity_id=basket_id,
            event_type="BASKET_CREATED",
            payload={
                "basket_name": basket_name,
                "strategy_version": strategy_version,
            },
        )

        return True


def get_basket_record(
    conn,
    basket_id: str = DEFAULT_BASKET_ID,
):
    return conn.execute(
        """
        SELECT *

        FROM public_baskets

        WHERE basket_id = %s
        """,
        (basket_id,),
    ).fetchone()


# =====================================================================
# SIGNAL RUNS
# =====================================================================


def record_signal_run(
    conn,
    basket_id: str,
    strategy_version: str,
    settings: Dict[str, Any],
    portfolio_before: Any,
    optimizer_output: Any,
    signal_output: Any,
    decision_status: str,
    *,
    generated_at: Optional[datetime] = None,
    data_as_of: Optional[datetime] = None,
    git_commit_sha: Optional[str] = None,
    input_snapshot_sha256: Optional[str] = None,
) -> str:
    """
    Persist one immutable optimizer run.

    Unlimited runs are allowed per day.

    Every invocation creates a unique signal_run_id.
    """

    if decision_status not in {
        "REBALANCED",
        "NO_CHANGE",
    }:
        raise ValueError(
            "decision_status must be REBALANCED or NO_CHANGE"
        )

    generated_at = (
        generated_at
        or utc_now()
    )

    signal_run_id = new_id(
        "SIG"
    )

    payload = {
        "signal_run_id": signal_run_id,
        "basket_id": basket_id,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "strategy_version": strategy_version,
        "git_commit_sha": git_commit_sha,
        "input_snapshot_sha256": input_snapshot_sha256,
        "settings": settings,
        "portfolio_before": portfolio_before,
        "optimizer_output": optimizer_output,
        "signal_output": signal_output,
        "decision_status": decision_status,
    }

    payload_hash = sha256_text(
        canonical_json(payload)
    )

    with conn.transaction():

        basket = get_basket_record(
            conn,
            basket_id,
        )

        if basket is None:
            raise ValueError(
                f"Unknown basket: {basket_id}"
            )

        conn.execute(
            """
            INSERT INTO signal_runs (
                signal_run_id,
                basket_id,
                generated_at,
                data_as_of,
                strategy_version,
                git_commit_sha,
                input_snapshot_sha256,
                settings_json,
                portfolio_before_json,
                optimizer_output_json,
                signal_output_json,
                decision_status,
                payload_sha256,
                created_at
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
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                signal_run_id,
                basket_id,
                generated_at,
                data_as_of,
                strategy_version,
                git_commit_sha,
                input_snapshot_sha256,
                Jsonb(settings),
                Jsonb(portfolio_before),
                Jsonb(optimizer_output),
                Jsonb(signal_output),
                decision_status,
                payload_hash,
                utc_now(),
            ),
        )

        append_audit(
            conn,
            entity_type="signal_run",
            entity_id=signal_run_id,
            event_type="SIGNAL_RECORDED",
            payload={
                "basket_id": basket_id,
                "decision_status": decision_status,
                "payload_sha256": payload_hash,
            },
        )

    return signal_run_id


# Backward-compatible wrapper name for upstream code.
def record_weekly_signal(
    conn,
    basket_id: str,
    today,
    strategy_version: str,
    git_commit_sha: Optional[str],
    settings: Dict[str, Any],
    portfolio_before: Any,
    optimizer_output: Any,
    signal_output: Any,
    decision_status: str,
):
    """
    Compatibility wrapper.

    Despite the old function name, this is NO LONGER weekly-gated.

    Every call creates a new immutable signal run.
    """

    return record_signal_run(
        conn=conn,
        basket_id=basket_id,
        strategy_version=strategy_version,
        settings=settings,
        portfolio_before=portfolio_before,
        optimizer_output=optimizer_output,
        signal_output=signal_output,
        decision_status=decision_status,
        git_commit_sha=git_commit_sha,
    )


def get_signal_run(
    conn,
    signal_run_id: str,
):
    return conn.execute(
        """
        SELECT *

        FROM signal_runs

        WHERE signal_run_id = %s
        """,
        (signal_run_id,),
    ).fetchone()


def list_signal_runs(
    conn,
    basket_id: str = DEFAULT_BASKET_ID,
    limit: int = 100,
):
    return conn.execute(
        """
        SELECT *

        FROM signal_runs

        WHERE basket_id = %s

        ORDER BY generated_at DESC

        LIMIT %s
        """,
        (
            basket_id,
            limit,
        ),
    ).fetchall()


# =====================================================================
# REBALANCE EVENTS
# =====================================================================


def create_rebalance_event(
    conn,
    basket_id: str,
    signal_run_id: str,
    *,
    rationale: Optional[str] = None,
    effective_at: Optional[datetime] = None,
    status: str = "CREATED",
) -> str:
    """
    Convert one immutable optimizer signal into a model rebalance event.

    Signal and rebalance remain separate entities.
    """

    allowed_statuses = {
        "CREATED",
        "APPROVED",
        "CANCELLED",
        "COMPLETED",
    }

    if status not in allowed_statuses:
        raise ValueError(
            f"Invalid rebalance status: {status}"
        )

    rebalance_id = new_id(
        "REB"
    )

    payload = {
        "rebalance_id": rebalance_id,
        "basket_id": basket_id,
        "signal_run_id": signal_run_id,
        "effective_at": effective_at,
        "status": status,
        "rationale": rationale,
    }

    payload_hash = sha256_text(
        canonical_json(payload)
    )

    with conn.transaction():

        signal = get_signal_run(
            conn,
            signal_run_id,
        )

        if signal is None:
            raise ValueError(
                f"Unknown signal_run_id: {signal_run_id}"
            )

        if signal["basket_id"] != basket_id:
            raise ValueError(
                "Signal does not belong to supplied basket."
            )

        conn.execute(
            """
            INSERT INTO rebalance_events (
                rebalance_id,
                basket_id,
                signal_run_id,
                created_at,
                effective_at,
                status,
                rationale,
                payload_json,
                payload_sha256
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
                rebalance_id,
                basket_id,
                signal_run_id,
                utc_now(),
                effective_at,
                status,
                rationale,
                Jsonb(payload),
                payload_hash,
            ),
        )

        append_audit(
            conn,
            entity_type="rebalance",
            entity_id=rebalance_id,
            event_type="REBALANCE_CREATED",
            payload={
                "signal_run_id": signal_run_id,
                "status": status,
                "payload_sha256": payload_hash,
            },
        )

    return rebalance_id


# =====================================================================
# TRADE ORDERS
# =====================================================================


def create_trade_order(
    conn,
    rebalance_id: str,
    symbol: str,
    side: str,
    requested_quantity: float,
    *,
    yahoo_ticker: Optional[str] = None,
    isin: Optional[str] = None,
    current_weight: Optional[float] = None,
    target_weight: Optional[float] = None,
    theoretical_quantity: Optional[float] = None,
    reference_price: Optional[float] = None,
    execution_rule: Optional[str] = None,
) -> str:
    """
    Create one immutable model trade order.
    """

    side = side.upper()

    if side not in {
        "BUY",
        "SELL",
    }:
        raise ValueError(
            "side must be BUY or SELL"
        )

    if requested_quantity <= 0:
        raise ValueError(
            "requested_quantity must be positive"
        )

    order_id = new_id(
        "ORD"
    )

    payload = {
        "order_id": order_id,
        "rebalance_id": rebalance_id,
        "symbol": symbol,
        "yahoo_ticker": yahoo_ticker,
        "isin": isin,
        "side": side,
        "current_weight": current_weight,
        "target_weight": target_weight,
        "theoretical_quantity": theoretical_quantity,
        "requested_quantity": requested_quantity,
        "reference_price": reference_price,
        "execution_rule": execution_rule,
    }

    payload_hash = sha256_text(
        canonical_json(payload)
    )

    with conn.transaction():

        rebalance = conn.execute(
            """
            SELECT rebalance_id

            FROM rebalance_events

            WHERE rebalance_id = %s
            """,
            (rebalance_id,),
        ).fetchone()

        if rebalance is None:
            raise ValueError(
                f"Unknown rebalance_id: {rebalance_id}"
            )

        conn.execute(
            """
            INSERT INTO trade_orders (
                order_id,
                rebalance_id,
                created_at,
                symbol,
                yahoo_ticker,
                isin,
                side,
                current_weight,
                target_weight,
                theoretical_quantity,
                requested_quantity,
                reference_price,
                execution_rule,
                order_status,
                payload_json,
                payload_sha256
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
                %s,
                %s,
                %s,
                %s,
                %s,
                'CREATED',
                %s,
                %s
            )
            """,
            (
                order_id,
                rebalance_id,
                utc_now(),
                symbol,
                yahoo_ticker,
                isin,
                side,
                current_weight,
                target_weight,
                theoretical_quantity,
                requested_quantity,
                reference_price,
                execution_rule,
                Jsonb(payload),
                payload_hash,
            ),
        )

        append_audit(
            conn,
            entity_type="trade_order",
            entity_id=order_id,
            event_type="TRADE_ORDER_CREATED",
            payload={
                "rebalance_id": rebalance_id,
                "symbol": symbol,
                "side": side,
                "quantity": requested_quantity,
                "payload_sha256": payload_hash,
            },
        )

    return order_id


# =====================================================================
# TRADE EXECUTIONS
# =====================================================================


def record_trade_execution(
    conn,
    order_id: str,
    quantity: float,
    execution_price: float,
    cash_change_inr: float,
    *,
    executed_at: Optional[datetime] = None,
    market_price: Optional[float] = None,
    fees_inr: float = 0.0,
    taxes_inr: float = 0.0,
    slippage_bps: Optional[float] = None,
) -> str:
    """
    Record one immutable execution/fill.

    An order may therefore have multiple partial executions.
    """

    if quantity <= 0:
        raise ValueError(
            "quantity must be positive"
        )

    if execution_price <= 0:
        raise ValueError(
            "execution_price must be positive"
        )

    execution_id = new_id(
        "EXE"
    )

    executed_at = (
        executed_at
        or utc_now()
    )

    payload = {
        "execution_id": execution_id,
        "order_id": order_id,
        "executed_at": executed_at,
        "quantity": quantity,
        "market_price": market_price,
        "execution_price": execution_price,
        "fees_inr": fees_inr,
        "taxes_inr": taxes_inr,
        "slippage_bps": slippage_bps,
        "cash_change_inr": cash_change_inr,
    }

    payload_hash = sha256_text(
        canonical_json(payload)
    )

    with conn.transaction():

        order = conn.execute(
            """
            SELECT order_id

            FROM trade_orders

            WHERE order_id = %s
            """,
            (order_id,),
        ).fetchone()

        if order is None:
            raise ValueError(
                f"Unknown order_id: {order_id}"
            )

        conn.execute(
            """
            INSERT INTO trade_executions (
                execution_id,
                order_id,
                executed_at,
                quantity,
                market_price,
                execution_price,
                fees_inr,
                taxes_inr,
                slippage_bps,
                cash_change_inr,
                payload_json,
                payload_sha256
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
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                execution_id,
                order_id,
                executed_at,
                quantity,
                market_price,
                execution_price,
                fees_inr,
                taxes_inr,
                slippage_bps,
                cash_change_inr,
                Jsonb(payload),
                payload_hash,
            ),
        )

        append_audit(
            conn,
            entity_type="trade_execution",
            entity_id=execution_id,
            event_type="TRADE_EXECUTION_RECORDED",
            payload={
                "order_id": order_id,
                "quantity": quantity,
                "execution_price": execution_price,
                "payload_sha256": payload_hash,
            },
        )

    return execution_id


# =====================================================================
# CASH LEDGER
# =====================================================================


def record_cash_event(
    conn,
    basket_id: str,
    event_type: str,
    amount_inr: float,
    *,
    event_at: Optional[datetime] = None,
    reference_type: Optional[str] = None,
    reference_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Append one immutable cash event.
    """

    cash_event_id = new_id(
        "CASH"
    )

    event_at = (
        event_at
        or utc_now()
    )

    payload = {
        "cash_event_id": cash_event_id,
        "basket_id": basket_id,
        "event_at": event_at,
        "event_type": event_type,
        "amount_inr": amount_inr,
        "reference_type": reference_type,
        "reference_id": reference_id,
        "notes": notes,
    }

    payload_hash = sha256_text(
        canonical_json(payload)
    )

    with conn.transaction():

        conn.execute(
            """
            INSERT INTO cash_ledger (
                cash_event_id,
                basket_id,
                event_at,
                event_type,
                amount_inr,
                reference_type,
                reference_id,
                notes,
                payload_sha256
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
                cash_event_id,
                basket_id,
                event_at,
                event_type,
                amount_inr,
                reference_type,
                reference_id,
                notes,
                payload_hash,
            ),
        )

        append_audit(
            conn,
            entity_type="cash_event",
            entity_id=cash_event_id,
            event_type="CASH_EVENT_RECORDED",
            payload={
                "basket_id": basket_id,
                "event_type": event_type,
                "amount_inr": amount_inr,
                "payload_sha256": payload_hash,
            },
        )

    return cash_event_id


# =====================================================================
# HEALTH / INSPECTION
# =====================================================================


def public_database_healthcheck(
    conn,
) -> dict:
    """
    Verify PostgreSQL connectivity.
    """

    row = conn.execute(
        """
        SELECT
            current_database() AS database_name,
            NOW() AS database_time,
            version() AS postgres_version
        """
    ).fetchone()

    return dict(row)


def get_public_basket_counts(
    conn,
    basket_id: str = DEFAULT_BASKET_ID,
) -> dict:
    """
    Useful for Streamlit status/debug page.
    """

    row = conn.execute(
        """
        SELECT

            (
                SELECT COUNT(*)
                FROM signal_runs
                WHERE basket_id = %s
            ) AS signal_runs,

            (
                SELECT COUNT(*)
                FROM rebalance_events
                WHERE basket_id = %s
            ) AS rebalance_events,

            (
                SELECT COUNT(*)
                FROM trade_orders o
                JOIN rebalance_events r
                    ON r.rebalance_id = o.rebalance_id
                WHERE r.basket_id = %s
            ) AS trade_orders,

            (
                SELECT COUNT(*)
                FROM trade_executions e
                JOIN trade_orders o
                    ON o.order_id = e.order_id
                JOIN rebalance_events r
                    ON r.rebalance_id = o.rebalance_id
                WHERE r.basket_id = %s
            ) AS trade_executions

        """,
        (
            basket_id,
            basket_id,
            basket_id,
            basket_id,
        ),
    ).fetchone()

    return dict(row)


def list_rebalance_events(
    conn,
    basket_id: str = DEFAULT_BASKET_ID,
    limit: int = 100,
):
    return conn.execute(
        """
        SELECT *

        FROM rebalance_events

        WHERE basket_id = %s

        ORDER BY created_at DESC

        LIMIT %s
        """,
        (
            basket_id,
            limit,
        ),
    ).fetchall()


def list_trade_orders(
    conn,
    rebalance_id: str,
):
    return conn.execute(
        """
        SELECT *

        FROM trade_orders

        WHERE rebalance_id = %s

        ORDER BY created_at
        """,
        (rebalance_id,),
    ).fetchall()


def list_trade_executions(
    conn,
    order_id: str,
):
    return conn.execute(
        """
        SELECT *

        FROM trade_executions

        WHERE order_id = %s

        ORDER BY executed_at
        """,
        (order_id,),
    ).fetchall()
