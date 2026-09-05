"""Idempotent scheduled job for accountable public 14-day forecasts."""

from __future__ import annotations

from datetime import date

import pandas as pd

from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_portfolio_config import load_public_portfolio_config
from public_portfolio_publications import evaluate_due_forecasts, load_trust_records, record_forecast
from public_portfolio_trust import CALCULATION_VERSION, bootstrap_outlook, fingerprint, forecast_payload


def main() -> int:
    config=load_public_portfolio_config()
    basket_id=config.basket_id
    url=get_public_basket_database_url()
    if not url: raise RuntimeError("Public PostgreSQL is not configured")
    with connect_public_basket_db(url) as conn:
        trust=load_trust_records(conn,basket_id)
        if not trust["current"]: return 0
        rows=conn.execute("""SELECT DISTINCT ON (nav_date) nav_date,nav,is_backfill FROM daily_nav WHERE basket_id=%s
            ORDER BY nav_date,calculation_version DESC""",(basket_id,)).fetchall()
        frame=pd.DataFrame(rows).sort_values("nav_date") if rows else pd.DataFrame()
        if len(frame)>=61:
            returns=frame["nav"].astype(float).pct_change().dropna()
            dates=frame.loc[returns.index,"nav_date"].tolist()
            seed=int(fingerprint({"publication":trust["current"]["publication_id"],"date":str(date.today())})[:8],16)
            result=bootstrap_outlook(returns.tolist(),dates,seed=seed)
            if result:
                payload=forecast_payload(result)
                payload["history_source"]=("DEVELOPMENT_BACKFILL" if any(bool(row.get("is_backfill")) for row in rows)
                                             else "LIVE_POST_PUBLICATION")
                record_forecast(conn,basket_id=basket_id,publication_id=trust["current"]["publication_id"],
                    forecast_date=date.today(),calculation_version=config.forecast_calculation_version,forecast=payload)
        evaluate_due_forecasts(conn,basket_id)
    return 0


if __name__=="__main__": raise SystemExit(main())
