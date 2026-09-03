"""
Dry-run the optimizer adapter for Option A: verify the pipeline produces a
sane signal WITHOUT writing anything to the ledger.

Usage:
    export PUBLIC_BASKET_DATABASE_URL="postgresql://..."
    export PUBLIC_BASKET_INPUT_PATH="/path/to/sample_public_basket_input.json"
    python dry_run_signal.py

Before running, edit sample_public_basket_input.json so its
"scheduled_session_date" matches the date this script prints below
(build_public_signal() rejects a mismatch on purpose, to stop stale
inputs from being reused).
"""

from __future__ import annotations

import json
from datetime import date

import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb


def main() -> None:
    conn = pb.connect_public_basket_db()
    try:
        info = pb.resolve_first_trading_day(conn, pb.DEFAULT_BASKET_ID, date.today())
    finally:
        conn.close()

    if info["status"] != "RESOLVED":
        raise SystemExit(f"Cannot dry-run: calendar gate status is {info['status']}")

    scheduled_session_date = info["first_trading_day"]
    print(
        f"Building signal as-of scheduled_session_date={scheduled_session_date} "
        "-- this must match 'scheduled_session_date' in your input JSON."
    )

    signal = adapter.build_public_signal(scheduled_session_date=scheduled_session_date)

    print("\n--- Dry-run signal (NOT written to the ledger) ---")
    print(f"decision_status: {signal['decision_status']}")
    print(f"target allocation: {len(signal['optimizer_output']['target_allocation'])} assets")
    print(json.dumps(signal["signal_output"], indent=2)[:2000])

    with open("dry_run_signal_output.json", "w", encoding="utf-8") as f:
        json.dump(signal, f, indent=2, sort_keys=True, default=str)
    print("\nFull output written to dry_run_signal_output.json for review.")


if __name__ == "__main__":
    main()
