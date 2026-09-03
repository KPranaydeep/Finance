"""
One-time setup for the public model portfolio (Option A: safe bootstrap).

Run this once, on any day of the week. It does NOT record a weekly signal --
it only prepares the durable schema, registers the basket, seeds the market
calendar, and tells you the actual date the *first real* signal will be due.

Safe to re-run: every step here is idempotent (schema DDL is IF NOT EXISTS,
create_public_basket / seed_nse_2026_calendar both no-op if already done).

Usage:
    export PUBLIC_BASKET_DATABASE_URL="postgresql://user:pass@host:port/dbname"
    python bootstrap_public_basket.py
"""

from __future__ import annotations

from datetime import date

import public_basket_postgres as pb


def main() -> None:
    conn = pb.connect_public_basket_db()
    try:
        print("Initializing schema (idempotent, safe to re-run)...")
        pb.init_public_basket_schema(conn)

        created = pb.create_public_basket(conn)
        print("Basket created." if created else "Basket already existed - no changes made.")

        print("Seeding NSE market calendar for the current year...")
        snapshot_id = pb.seed_nse_2026_calendar(conn)
        print(f"Calendar snapshot stored/confirmed: {snapshot_id}")

        today = date.today()
        info = pb.resolve_first_trading_day(conn, pb.DEFAULT_BASKET_ID, today)

        print("\n--- Status ---")
        print(f"Today:                {today.isoformat()} ({today.strftime('%A')})")
        print(f"Week start (Monday):  {info['week_start']}")
        print(f"Gate status:          {info['status']}")
        print(f"First trading day:    {info['first_trading_day']}")
        print(f"Calendar snapshot id: {info['snapshot_id']}")

        if info["status"] == "RESOLVED" and info["first_trading_day"] != today:
            print(
                "\nThis week's first-open-session has already passed "
                f"({info['first_trading_day']}). Nothing will be recorded "
                "for this week. The ledger will pick up automatically on the "
                "first open session of NEXT week once run_weekly_signal.py "
                "is scheduled (see the accompanying GitHub Actions workflow)."
            )
        elif info["status"] == "CALENDAR_INCOMPLETE":
            print(
                "\nNo calendar snapshot covers this week yet. Extend the "
                "seeded range with store_calendar_snapshot() before going live."
            )
        elif info["status"] == "NO_OPEN_SESSION":
            print("\nNo NSE session is open this calendar week (holiday week).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
