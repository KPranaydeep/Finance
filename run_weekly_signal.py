"""
Production weekly runner. Safe to invoke every day (e.g. via cron / GitHub
Actions) - it only records a new signal on the first open NSE session of
the calendar week, and is a no-op every other day.

Usage:
    export PUBLIC_BASKET_DATABASE_URL="postgresql://..."
    export PUBLIC_BASKET_INPUT_PATH="/path/to/todays_input.json"
    python run_weekly_signal.py

IMPORTANT: this file assumes public_basket_postgres.py exposes a
record_weekly_signal() with the same signature as the one in
public_basket_ledger.py (the SQLite version). Confirm that in your repo --
the postgres module's source was long enough that I could not see that far
into it -- and adjust the call below if the name or arguments differ.
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
        gate = pb.resolve_first_trading_day(conn, pb.DEFAULT_BASKET_ID, today)

        if gate["status"] != "RESOLVED":
            print(f"Gate status={gate['status']}; nothing to do today.")
            return 0

        if gate["first_trading_day"] != today:
            print(f"Not due today. First open session this week is {gate['first_trading_day']}.")
            return 0

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
