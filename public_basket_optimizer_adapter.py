from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    append_audit,
    canonical_json,
    connect_public_basket_db,
    rebalance_gate,
    sha256_text,
    utc_now,
)


INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
SCHEDULER_ADVISORY_LOCK = 9485218
VALID_DECISIONS = {"REBALANCED", "NO_CHANGE"}


def current_india_date() -> date:
    return datetime.now(INDIA_TIMEZONE).date()


def git_commit_sha() -> str | None:
    configured_sha = os.getenv("PUBLIC_BASKET_GIT_COMMIT_SHA")
    if configured_sha:
        return configured_sha.strip()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def load_optimizer_payload(scheduled_session_date: date) -> dict[str, Any]:
    """
    Call the public optimizer adapter.

    The adapter must reuse the existing optimizer functions without changing their
    mathematics. Keeping it separate prevents this scheduler from importing the
    interactive Streamlit application and accidentally running its user interface.
    """
    try:
        from public_basket_optimizer_adapter import build_public_signal
    except ImportError as exc:
        raise RuntimeError(
            "public_basket_optimizer_adapter.py is required. It must expose "
            "build_public_signal(scheduled_session_date=...) and return the frozen "
            "optimizer inputs and outputs."
        ) from exc

    payload = build_public_signal(
        scheduled_session_date=scheduled_session_date,
    )

    if not isinstance(payload, dict):
        raise TypeError("build_public_signal() must return a dictionary")

    required = {
        "strategy_version",
        "settings",
        "portfolio_before",
        "optimizer_output",
        "signal_output",
        "decision_status",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "Optimizer adapter result is missing: " + ", ".join(missing)
        )

    decision = str(payload["decision_status"]).upper()
    if decision not in VALID_DECISIONS:
        raise ValueError(
            "decision_status must be REBALANCED or NO_CHANGE"
        )

    payload["decision_status"] = decision
    return payload


def store_official_signal(
    conn,
    *,
    basket_id: str,
    run_date: date,
    optimizer_payload: dict[str, Any],
    commit_sha: str | None,
) -> dict[str, Any]:
    """Atomically store one official signal and close its weekly cycle."""
    with conn.transaction():
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (SCHEDULER_ADVISORY_LOCK,),
        )

        gate = rebalance_gate(
            conn=conn,
            basket_id=basket_id,
            today=run_date,
        )

        if gate["status"] == "ALREADY_EVALUATED":
            return {
                "status": "ALREADY_EVALUATED",
                "signal_run_id": gate.get("signal_run_id"),
                "week_start": str(gate["week_start"]),
            }

        if gate["status"] != "DUE":
            raise RuntimeError(
                f"Official signal cannot be stored: gate is {gate['status']}"
            )

        generated_at = utc_now()
        week_start = gate["week_start"]
        scheduled_session = gate["first_trading_day"]

        immutable_payload = {
            "basket_id": basket_id,
            "week_start_date": week_start.isoformat(),
            "scheduled_session_date": scheduled_session.isoformat(),
            "strategy_version": str(optimizer_payload["strategy_version"]),
            "git_commit_sha": commit_sha,
            "settings": optimizer_payload["settings"],
            "portfolio_before": optimizer_payload["portfolio_before"],
            "optimizer_output": optimizer_payload["optimizer_output"],
            "signal_output": optimizer_payload["signal_output"],
        }
        payload_hash = sha256_text(canonical_json(immutable_payload))
        signal_run_id = f"SIG-{week_start:%Y%m%d}-{payload_hash[:12]}"
        cycle_id = f"CYCLE-{basket_id}-{week_start:%Y%m%d}"

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
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            """,
            (
                signal_run_id,
                basket_id,
                week_start,
                scheduled_session,
                generated_at,
                immutable_payload["strategy_version"],
                commit_sha,
                Jsonb(immutable_payload["settings"]),
                Jsonb(immutable_payload["portfolio_before"]),
                Jsonb(immutable_payload["optimizer_output"]),
                Jsonb(immutable_payload["signal_output"]),
                payload_hash,
            ),
        )

        decision = optimizer_payload["decision_status"]
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                cycle_id,
                basket_id,
                week_start,
                gate["snapshot_id"],
                scheduled_session,
                generated_at,
                decision,
                signal_run_id,
                Jsonb({"payload_sha256": payload_hash}),
            ),
        )

        append_audit(
            conn=conn,
            entity_type="signal_run",
            entity_id=signal_run_id,
            event_type=decision,
            payload=immutable_payload,
        )

    return {
        "status": decision,
        "signal_run_id": signal_run_id,
        "cycle_id": cycle_id,
        "week_start": week_start.isoformat(),
        "scheduled_session_date": scheduled_session.isoformat(),
        "payload_sha256": payload_hash,
    }


def run_scheduler(
    *,
    basket_id: str = DEFAULT_BASKET_ID,
    run_date: date | None = None,
) -> dict[str, Any]:
    """
    Evaluate the gate and run the optimizer only on the scheduled NSE session.

    The database schema, basket, and versioned market-calendar snapshot must already
    have been initialized by deployment/bootstrap tooling.
    """
    effective_date = run_date or current_india_date()
    conn = connect_public_basket_db()

    try:
        initial_gate = rebalance_gate(
            conn=conn,
            basket_id=basket_id,
            today=effective_date,
        )

        if initial_gate["status"] != "DUE":
            return {
                "status": initial_gate["status"],
                "basket_id": basket_id,
                "run_date": effective_date.isoformat(),
                "details": {
                    key: str(value) if isinstance(value, (date, datetime)) else value
                    for key, value in initial_gate.items()
                    if key != "status"
                },
            }

        scheduled_session = initial_gate["first_trading_day"]
        optimizer_payload = load_optimizer_payload(scheduled_session)

        stored = store_official_signal(
            conn,
            basket_id=basket_id,
            run_date=effective_date,
            optimizer_payload=optimizer_payload,
            commit_sha=git_commit_sha(),
        )
        return {
            **stored,
            "basket_id": basket_id,
            "run_date": effective_date.isoformat(),
        }
    except UniqueViolation:
        # A concurrent worker won the database race. Re-read the authoritative row.
        gate = rebalance_gate(
            conn=conn,
            basket_id=basket_id,
            today=effective_date,
        )
        return {
            "status": gate["status"],
            "basket_id": basket_id,
            "run_date": effective_date.isoformat(),
            "signal_run_id": gate.get("signal_run_id"),
        }
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official public-basket weekly scheduler."
    )
    parser.add_argument("--basket-id", default=DEFAULT_BASKET_ID)
    parser.add_argument(
        "--date",
        dest="run_date",
        type=date.fromisoformat,
        help="YYYY-MM-DD override for controlled testing only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_scheduler(
            basket_id=args.basket_id,
            run_date=args.run_date,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAILED", "error": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
