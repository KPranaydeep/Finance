"""Idempotent scheduled job for accountable public 28-calendar-day outlooks."""

from __future__ import annotations

from public_nav_snapshots import load_nav_snapshot, encode_snapshot

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_portfolio_config import load_public_portfolio_config
from public_portfolio_publications import evaluate_due_forecasts, load_trust_records, record_forecast
from public_outlook import calendar_outlook, VERSION


def main() -> int:
    config=load_public_portfolio_config()
    basket_id=config.basket_id
    url=get_public_basket_database_url()
    if not url: raise RuntimeError("Public PostgreSQL is not configured")
    with connect_public_basket_db(url) as conn:
        trust=load_trust_records(conn,basket_id)
        if not trust["current"]: return 0
        rows=load_nav_snapshot(conn,basket_id)
        frame=pd.DataFrame(rows).sort_values("nav_date") if rows else pd.DataFrame()
        forecast_date=datetime.now(ZoneInfo(config.timezone)).date()
        if not frame.empty:
            payload=calendar_outlook(frame.to_dict("records"),forecast_date)
            if payload:
                _,snapshot_hash=encode_snapshot(rows)
                payload["nav_snapshot_sha256"]=snapshot_hash
                payload["history_source"]=("DEVELOPMENT_BACKFILL" if any(bool(row.get("is_backfill")) for row in rows)
                                             else "LIVE_POST_PUBLICATION")
                record_forecast(conn,basket_id=basket_id,publication_id=trust["current"]["publication_id"],
                    forecast_date=forecast_date,calculation_version=f"{VERSION}-nav-{snapshot_hash[:12]}",forecast=payload)
        evaluate_due_forecasts(conn,basket_id)
    return 0


if __name__=="__main__": raise SystemExit(main())
