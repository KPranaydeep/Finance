"""
Production weekly runner. Safe to invoke every day (e.g. via cron / GitHub
Actions) - it only records a new signal on the first open NSE session of
the calendar week, and is a no-op every other day.

Usage:
    export PUBLIC_BASKET_DATABASE_URL="postgresql://..."
    export PUBLIC_BASKET_INPUT_PATH="/path/to/todays_input.json"
    python run_weekly_signal.py

IMPORTANT - two unverified assumptions, please confirm against your real
public_basket_postgres.py:
1. rebalance_gate() is the function that actually computes MISSED (inferred
   from the Public_Basket_Status page's behavior, not from source I've read).
   If MISSED is computed elsewhere, or under a different status string,
   this script's branching below won't catch it and could silently no-op
   on a missed week -- exactly the bug this script previously had.
2. record_weekly_signal()'s exact signature. I mirrored the SQLite ledger
   version since I could not see far enough into the Postgres file.
Confirm both once, e.g. by running this with a print(gate) added, before
trusting it unattended on a real scheduled morning.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb


def main() -> int:
    conn = pb.connect_public_basket_db()
    try:
        today = date.today()
        gate = pb.rebalance_gate(conn, pb.DEFAULT_BASKET_ID, today)

        if gate["status"] == "MISSED":
            # By design this system never catches up. This is not a
            # transient "try again" state -- it means this calendar week's
            # public record has a permanent gap. Exit non-zero so CI
            # failure notifications actually fire; do not retry.
            print(
                f"GATE STATUS = MISSED for week starting {gate.get('week_start')}. "
                "This week's signal will NOT be recorded -- the scheduler must "
                "not catch up later. Investigate why the runner didn't fire on "
                f"{gate.get('first_trading_day')}."
            )
            return 1

        if gate["status"] in ("NOT_DUE", "ALREADY_EVALUATED"):
            print(f"Gate status={gate['status']}; nothing to do today.")
            return 0

        if gate["status"] in ("CALENDAR_INCOMPLETE", "NO_OPEN_SESSION"):
            print(f"Gate status={gate['status']}; calendar issue, not a miss. Investigate.")
            return 1

        if gate["status"] != "DUE":
            print(f"Unrecognized gate status={gate['status']!r}. Treating as failure "
                  "so this doesn't fail silently -- confirm the actual status values "
                  "rebalance_gate() can return in public_basket_postgres.py.")
            return 1

        print(f"Today ({today}) is the scheduled session. Building signal...")
        signal = adapter.build_public_signal(scheduled_session_date=today)

        signal_run_id = pb.record_weekly_signal(
            conn,
            basket_id=pb.DEFAULT_BASKET_ID,
            today=today,
            strategy_version=signal["strategy_version"],
            git_commit_sha=os.getenv("GITHUB_SHA"),
            settings=signal["settings"],
            portfolio_before=signal["portfolio_before"],
            optimizer_output=signal["optimizer_output"],
            signal_output=signal["signal_output"],
            decision_status=signal["decision_status"],
        )
        print(f"Recorded signal_run_id={signal_run_id}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
