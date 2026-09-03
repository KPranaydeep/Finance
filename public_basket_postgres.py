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

# Opt-in: allow the weekly signal to be authored on any day of the week.
# When true, the first run in the week will be allowed on any day (but only once).
PUBLIC_BASKET_ALLOW_ANY_DAY = os.getenv("PUBLIC_BASKET_ALLOW_ANY_DAY", "false").lower() in ("1", "true", "yes")


# =====================================================================
# DATABASE CONFIGURATION
# =====================================================================


def get_public_basket_database_url() -> str | None:
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
    return reference_date - timedelta(days=reference_date.weekday())


def week_end(reference_date: date) -> date:
    return week_start(reference_date) + timedelta(days=6)


# =====================================================================
# POSTGRESQL SCHEMA
# =====================================================================

SCHEMA_STATEMENTS = (
    # (omitted here for brevity; keep the schema statements from your repo)
)

def init_public_basket_schema(conn) -> None:
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
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (AUDIT_ADVISORY_LOCK,))
    previous = conn.execute(
        """
        SELECT event_hash
        FROM public_basket_audit_log
        ORDER BY audit_id DESC
        LIMIT 1
        """
    ).fetchone()
    previous_hash = previous["event_hash"] if previous else ""
    audit_payload = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "event_type": event_type,
        "payload": payload,
    }
    canonical = canonical_json(audit_payload)
    event_hash = sha256_text(previous_hash + "|" + canonical)
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
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (utc_now(), entity_type, entity_id, event_type, Jsonb(audit_payload), previous_hash or None, event_hash),
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
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO public_baskets (
                basket_id, basket_name, calendar_market, rebalance_rule,
                strategy_version, schema_version, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (basket_id) DO NOTHING
            RETURNING basket_id
            """,
            (basket_id, basket_name, MARKET, REBALANCE_RULE, strategy_version, PUBLIC_BASKET_SCHEMA_VERSION, utc_now()),
        ).fetchone()
        if row is None:
            return False
        append_audit(
            conn=conn,
            entity_type="basket",
            entity_id=basket_id,
            event_type="BASKET_CREATED",
            payload={"basket_name": basket_name, "market": MARKET, "rebalance_rule": REBALANCE_RULE, "strategy_version": strategy_version},
        )
        return True


# =====================================================================
# MARKET CALENDAR
# =====================================================================

def store_calendar_snapshot(conn, payload: dict) -> str:
    canonical = canonical_json(payload)
    source_hash = sha256_text(canonical)
    range_start = date.fromisoformat(payload["range_start"])
    range_end = date.fromisoformat(payload["range_end"])
    snapshot_id = f"CAL-{payload['market']}-{range_start:%Y%m%d}-{range_end:%Y%m%d}-{source_hash[:12]}"
    with conn.transaction():
        existing = conn.execute("SELECT snapshot_id FROM market_calendar_snapshots WHERE snapshot_id = %s", (snapshot_id,)).fetchone()
        if existing:
            return snapshot_id
        conn.execute(
            """
            INSERT INTO market_calendar_snapshots (
                snapshot_id, market, range_start, range_end, source,
                source_ref, source_as_of, loaded_at, source_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (snapshot_id, payload["market"], range_start, range_end, payload["source"], payload.get("source_ref"), date.fromisoformat(payload["source_as_of"]), utc_now(), source_hash),
        )
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO market_sessions (
                    snapshot_id, session_date, is_open, session_type, notes
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (snapshot_id, date.fromisoformat(row["session_date"]), bool(row["is_open"]), row.get("session_type", "NORMAL"), row.get("notes"))
                    for row in payload["sessions"]
                ],
            )
        append_audit(conn=conn, entity_type="calendar", entity_id=snapshot_id, event_type="CALENDAR_SNAPSHOT_STORED", payload={"market": payload["market"], "range_start": payload["range_start"], "range_end": payload["range_end"], "source": payload["source"], "source_ref": payload.get("source_ref"), "source_as_of": payload.get("source_as_of"), "source_sha256": source_hash})
    return snapshot_id

def seed_nse_2026_calendar(conn) -> str:
    return store_calendar_snapshot(conn, calendar_snapshot_payload_2026())

def get_calendar_snapshot_for_week(conn, market: str, monday: date):
    sunday = monday + timedelta(days=6)
    return conn.execute(
        """
        SELECT *
        FROM market_calendar_snapshots
        WHERE market = %s AND range_start <= %s AND range_end >= %s
        ORDER BY loaded_at DESC
        LIMIT 1
        """,
        (market, monday, sunday),
    ).fetchone()


# =====================================================================
# WEEKLY SCHEDULER
# =====================================================================

def resolve_first_trading_day(conn, basket_id: str, reference_date: date) -> dict:
    monday = week_start(reference_date)
    sunday = monday + timedelta(days=6)
    basket = conn.execute(
        "SELECT basket_id, calendar_market, rebalance_rule, strategy_version FROM public_baskets WHERE basket_id = %s",
        (basket_id,),
    ).fetchone()
    if basket is None:
        raise ValueError(f"Unknown public basket: {basket_id}")
    snapshot = get_calendar_snapshot_for_week(conn=conn, market=basket["calendar_market"], monday=monday)
    if snapshot is None:
        return {"status": "CALENDAR_INCOMPLETE", "week_start": monday, "week_end": sunday, "first_trading_day": None, "snapshot_id": None}
    first_open = conn.execute(
        """
        SELECT session_date, session_type, notes
        FROM market_sessions
        WHERE snapshot_id = %s AND session_date >= %s AND session_date <= %s AND is_open = TRUE
        ORDER BY session_date
        LIMIT 1
        """,
        (snapshot["snapshot_id"], monday, sunday),
    ).fetchone()
    if first_open is None:
        return {"status": "NO_OPEN_SESSION", "week_start": monday, "week_end": sunday, "first_trading_day": None, "snapshot_id": snapshot["snapshot_id"]}
    return {"status": "RESOLVED", "week_start": monday, "week_end": sunday, "first_trading_day": first_open["session_date"], "session_type": first_open["session_type"], "session_notes": first_open["notes"], "snapshot_id": snapshot["snapshot_id"]}


def rebalance_gate(conn, basket_id: str, today: date) -> dict:
    monday = week_start(today)
    existing = conn.execute(
        """
        SELECT cycle_id, status, scheduled_session_date, signal_run_id, evaluated_at
        FROM weekly_rebalance_cycles
        WHERE basket_id = %s AND week_start_date = %s
        """,
        (basket_id, monday),
    ).fetchone()
    if existing:
        return {"status": "ALREADY_EVALUATED", "week_start": monday, "cycle_status": existing["status"], "scheduled_session_date": existing["scheduled_session_date"], "signal_run_id": existing["signal_run_id"], "evaluated_at": existing["evaluated_at"]}
    schedule = resolve_first_trading_day(conn=conn, basket_id=basket_id, reference_date=today)
    if schedule["status"] != "RESOLVED":
        return schedule
    scheduled = schedule["first_trading_day"]
    if PUBLIC_BASKET_ALLOW_ANY_DAY:
        return {**schedule, "first_trading_day": today, "status": "DUE"}
    if today < scheduled:
        return {**schedule, "status": "NOT_DUE"}
    if today > scheduled:
        return {**schedule, "status": "MISSED"}
    return {**schedule, "status": "DUE"}


def record_weekly_signal(
    conn,
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
    if decision_status not in {"REBALANCED", "NO_CHANGE"}:
        raise ValueError("decision_status must be REBALANCED or NO_CHANGE")
    gate = rebalance_gate(conn, basket_id, today)
    if gate["status"] == "ALREADY_EVALUATED":
        return gate["signal_run_id"]
    if gate["status"] != "DUE":
        raise RuntimeError(f"Weekly signal cannot run: {gate['status']}")
    week = gate["week_start"]
    payload = {
        "basket_id": basket_id,
        "week_start_date": week.isoformat(),
        "scheduled_session_date": gate["first_trading_day"].isoformat(),
        "strategy_version": strategy_version,
        "git_commit_sha": git_commit_sha,
        "settings": settings,
        "portfolio_before": portfolio_before,
        "optimizer_output": optimizer_output,
        "signal_output": signal_output,
    }
    payload_hash = sha256_text(canonical_json(payload))
    signal_run_id = f"SIG-{week:%Y%m%d}-{payload_hash[:12]}"
    generated_at = utc_now()
    cycle_id = f"CYCLE-{basket_id}-{week:%Y%m%d}"
    try:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO signal_runs (
                    signal_run_id, basket_id, week_start_date, scheduled_session_date,
                    generated_at, strategy_version, git_commit_sha,
                    settings_json, portfolio_before_json, optimizer_output_json, signal_output_json, payload_sha256
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    signal_run_id,
                    basket_id,
                    week,
                    gate["first_trading_day"],
                    generated_at,
                    strategy_version,
                    git_commit_sha,
                    Jsonb(settings),
                    Jsonb(portfolio_before),
                    Jsonb(optimizer_output),
                    Jsonb(signal_output),
                    payload_hash,
                ),
            )
            conn.execute(
                """
                INSERT INTO weekly_rebalance_cycles (
                    cycle_id, basket_id, week_start_date, calendar_snapshot_id,
                    scheduled_session_date, evaluated_at, status, signal_run_id, details_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    cycle_id,
                    basket_id,
                    week,
                    gate.get("snapshot_id"),
                    gate["first_trading_day"],
                    generated_at,
                    decision_status,
                    signal_run_id,
                    Jsonb({"payload_sha256": payload_hash}),
                ),
            )
            append_audit(conn, "signal_run", signal_run_id, decision_status, payload)
    except Exception:
        raise
    return signal_run_id


# =====================================================================
# STATUS / UI HELPERS
# =====================================================================

def list_week_sessions(conn, basket_id: str, reference_date: date) -> list[dict]:
    schedule = resolve_first_trading_day(conn=conn, basket_id=basket_id, reference_date=reference_date)
    snapshot_id = schedule.get("snapshot_id")
    if snapshot_id is None:
        return []
    monday = week_start(reference_date)
    sunday = monday + timedelta(days=6)
    rows = conn.execute(
        """
        SELECT session_date, is_open, session_type, notes
        FROM market_sessions
        WHERE snapshot_id = %s AND session_date >= %s AND session_date <= %s
        ORDER BY session_date
        """,
        (snapshot_id, monday, sunday),
    ).fetchall()
    return [dict(row) for row in rows]


def get_basket_record(conn, basket_id: str):
    return conn.execute("SELECT * FROM public_baskets WHERE basket_id = %s", (basket_id,)).fetchone()


def public_database_healthcheck(conn) -> dict:
    result = conn.execute("SELECT current_database() AS database_name, NOW() AS database_time").fetchone()
    return dict(result)
