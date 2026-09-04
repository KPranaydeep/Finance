from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from public_basket_postgres import (
    PUBLIC_BASKET_SCHEMA_VERSION,
    connect_public_basket_db,
    init_public_basket_schema,
)


MIGRATION_ADVISORY_LOCK = 9485218

V1_TABLES = (
    "market_sessions",
    "market_calendar_snapshots",
    "weekly_rebalance_cycles",
    "trade_executions",
    "trade_orders",
    "rebalance_events",
    "signal_runs",
    "daily_nav",
    "public_basket_audit_log",
    "public_baskets",
)

HISTORY_TABLES = (
    "signal_runs",
    "rebalance_events",
    "trade_orders",
    "trade_executions",
    "daily_nav",
)

IMMUTABLE_ARCHIVES = (
    "market_sessions",
    "market_calendar_snapshots",
    "weekly_rebalance_cycles",
    "trade_executions",
    "trade_orders",
    "rebalance_events",
    "signal_runs",
    "daily_nav",
    "public_basket_audit_log",
)


@dataclass(frozen=True)
class MigrationPlan:
    detected_schema: str
    backend_schema_version: int
    basket_versions: tuple[int, ...]
    row_counts: dict[str, int]
    archive_conflicts: tuple[str, ...]
    safe_to_apply: bool
    reason: str


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT to_regclass(%s) AS table_name",
        (table_name,),
    ).fetchone()
    return bool(row and row["table_name"])


def _columns(conn: Any, table_name: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
        """,
        (table_name,),
    ).fetchall()
    return {row["column_name"] for row in rows}


def _count(conn: Any, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    # table_name is selected exclusively from constants in this module.
    return int(conn.execute(f'SELECT COUNT(*) AS count FROM "{table_name}"').fetchone()["count"])


def inspect_migration(conn: Any) -> MigrationPlan:
    if not _table_exists(conn, "public_baskets"):
        return MigrationPlan(
            detected_schema="EMPTY",
            backend_schema_version=PUBLIC_BASKET_SCHEMA_VERSION,
            basket_versions=(),
            row_counts={},
            archive_conflicts=(),
            safe_to_apply=False,
            reason="No version-1 public_baskets table exists; initialize version 2 normally.",
        )

    basket_columns = _columns(conn, "public_baskets")
    if {"base_currency", "status"}.issubset(basket_columns) and "rebalance_rule" not in basket_columns:
        detected_schema = "V2"
    elif {"calendar_market", "rebalance_rule"}.issubset(basket_columns):
        detected_schema = "V1"
    else:
        detected_schema = "UNKNOWN"

    versions = tuple(
        sorted(
            {
                int(row["schema_version"])
                for row in conn.execute(
                    "SELECT schema_version FROM public_baskets"
                ).fetchall()
            }
        )
    )
    counts = {table: _count(conn, table) for table in V1_TABLES}
    conflicts = tuple(
        table
        for table in V1_TABLES
        if _table_exists(conn, f"{table}_v1_archive")
    )
    populated_history = {
        table: counts[table]
        for table in HISTORY_TABLES
        if counts.get(table, 0) > 0
    }

    if detected_schema == "V2":
        safe, reason = False, "Database already has the version-2 table shape."
    elif detected_schema != "V1":
        safe, reason = False, "Unrecognized schema shape; refusing an automatic migration."
    elif PUBLIC_BASKET_SCHEMA_VERSION != 2:
        safe, reason = False, "The imported backend is not schema version 2."
    elif conflicts:
        safe, reason = False, "One or more version-1 archive table names already exist."
    elif populated_history:
        safe, reason = False, (
            "Version-1 investment history exists and needs a bespoke record mapping: "
            + json.dumps(populated_history, sort_keys=True)
        )
    else:
        safe, reason = True, (
            "No version-1 signals, rebalances, orders, executions, or NAV rows exist. "
            "The metadata-only migration can proceed without losing investment history."
        )

    return MigrationPlan(
        detected_schema=detected_schema,
        backend_schema_version=PUBLIC_BASKET_SCHEMA_VERSION,
        basket_versions=versions,
        row_counts=counts,
        archive_conflicts=conflicts,
        safe_to_apply=safe,
        reason=reason,
    )


def _drop_archive_triggers(conn: Any, archive_table: str) -> None:
    rows = conn.execute(
        """
        SELECT trigger_name
        FROM information_schema.triggers
        WHERE event_object_schema = current_schema()
          AND event_object_table = %s
        """,
        (archive_table,),
    ).fetchall()
    for row in rows:
        trigger_name = row["trigger_name"].replace('"', '""')
        table_name = archive_table.replace('"', '""')
        conn.execute(f'DROP TRIGGER "{trigger_name}" ON "{table_name}"')


def _archive_object_name(prefix: str, original_name: str) -> str:
    digest = hashlib.sha256(original_name.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{original_name[:48]}_{digest}"[:63]


def _rename_archive_indexes(conn: Any, archive_table: str) -> None:
    rows = conn.execute(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename = %s
        ORDER BY indexname
        """,
        (archive_table,),
    ).fetchall()
    for row in rows:
        old_name = row["indexname"].replace('"', '""')
        new_name = _archive_object_name("v1idx", row["indexname"]).replace('"', '""')
        conn.execute(f'ALTER INDEX "{old_name}" RENAME TO "{new_name}"')


def _rename_archive_sequences(conn: Any, archive_table: str) -> None:
    rows = conn.execute(
        """
        SELECT DISTINCT sequence.relname AS sequence_name
        FROM pg_class AS archived_table
        JOIN pg_attribute AS column_definition
          ON column_definition.attrelid = archived_table.oid
        JOIN pg_depend AS dependency
          ON dependency.refobjid = archived_table.oid
         AND dependency.refobjsubid = column_definition.attnum
         AND dependency.deptype = 'a'
        JOIN pg_class AS sequence
          ON sequence.oid = dependency.objid
         AND sequence.relkind = 'S'
        WHERE archived_table.relname = %s
          AND archived_table.relnamespace = current_schema()::regnamespace
        ORDER BY sequence.relname
        """,
        (archive_table,),
    ).fetchall()
    for row in rows:
        old_name = row["sequence_name"].replace('"', '""')
        new_name = _archive_object_name("v1seq", row["sequence_name"]).replace('"', '""')
        conn.execute(f'ALTER SEQUENCE "{old_name}" RENAME TO "{new_name}"')


def _protect_archive(conn: Any, original_table: str) -> None:
    archive_table = f"{original_table}_v1_archive"
    trigger_name = f"immutable_guard_{archive_table}"
    conn.execute(
        f"""
        CREATE TRIGGER "{trigger_name}"
        BEFORE UPDATE OR DELETE ON "{archive_table}"
        FOR EACH ROW EXECUTE FUNCTION reject_public_basket_mutation()
        """
    )


def apply_migration(conn: Any) -> MigrationPlan:
    with conn.transaction():
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (MIGRATION_ADVISORY_LOCK,),
        )
        plan = inspect_migration(conn)
        if not plan.safe_to_apply:
            raise RuntimeError(plan.reason)

        basket_rows = conn.execute(
            """
            SELECT basket_id, basket_name, strategy_version, created_at
            FROM public_baskets
            ORDER BY basket_id
            """
        ).fetchall()

        existing_tables = [table for table in V1_TABLES if _table_exists(conn, table)]
        for table in existing_tables:
            conn.execute(f'LOCK TABLE "{table}" IN ACCESS EXCLUSIVE MODE')

        for table in existing_tables:
            archive_table = f"{table}_v1_archive"
            conn.execute(f'ALTER TABLE "{table}" RENAME TO "{archive_table}"')
            _drop_archive_triggers(conn, archive_table)
            _rename_archive_indexes(conn, archive_table)
            _rename_archive_sequences(conn, archive_table)

        # This creates the authoritative version-2 tables inside the same outer
        # transaction. psycopg implements the nested transaction as a savepoint.
        init_public_basket_schema(conn)

        for row in basket_rows:
            conn.execute(
                """
                INSERT INTO public_baskets (
                    basket_id,
                    basket_name,
                    base_currency,
                    strategy_version,
                    schema_version,
                    created_at,
                    status
                ) VALUES (%s, %s, 'INR', %s, 2, %s, 'ACTIVE')
                """,
                (
                    row["basket_id"],
                    row["basket_name"],
                    row["strategy_version"],
                    row["created_at"],
                ),
            )

        if _table_exists(conn, "public_basket_audit_log_v1_archive"):
            conn.execute(
                """
                INSERT INTO public_basket_audit_log (
                    audit_id,
                    event_at,
                    entity_type,
                    entity_id,
                    event_type,
                    payload_json,
                    previous_hash,
                    event_hash
                )
                SELECT
                    audit_id,
                    event_at,
                    entity_type,
                    entity_id,
                    event_type,
                    payload_json,
                    previous_hash,
                    event_hash
                FROM public_basket_audit_log_v1_archive
                ORDER BY audit_id
                """
            )
            conn.execute(
                """
                SELECT setval(
                    pg_get_serial_sequence('public_basket_audit_log', 'audit_id'),
                    COALESCE((SELECT MAX(audit_id) FROM public_basket_audit_log), 1),
                    EXISTS (SELECT 1 FROM public_basket_audit_log)
                )
                """
            )

        for table in IMMUTABLE_ARCHIVES:
            if _table_exists(conn, f"{table}_v1_archive"):
                _protect_archive(conn, table)

        migrated = inspect_migration(conn)
        if migrated.detected_schema != "V2":
            raise RuntimeError("Post-migration verification did not detect schema version 2")

    return migrated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely migrate an empty-history public basket from schema v1 to v2."
    )
    parser.add_argument(
        "--database-url",
        help="Override PUBLIC_BASKET_DATABASE_URL for this invocation.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, only print the migration plan.",
    )
    args = parser.parse_args()

    conn = connect_public_basket_db(args.database_url)
    try:
        plan = inspect_migration(conn)
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))

        if not args.apply:
            print("Preview only. Re-run with --apply after reviewing the plan.")
            return 0 if plan.safe_to_apply else 2
        if not plan.safe_to_apply:
            raise RuntimeError(plan.reason)

        result = apply_migration(conn)
        print("Migration committed successfully.")
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
