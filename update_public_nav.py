"""Build an observed, version-aware public model index from published weights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pandas as pd
import yfinance as yf

from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_portfolio_config import load_public_portfolio_config
from public_portfolio_publications import load_trust_records
from public_portfolio_trust import fingerprint, versioned_model_nav

CALCULATION_VERSION=5  # gross and estimated-net model NAV; excludes voided publications


def main() -> int:
    config=load_public_portfolio_config()
    basket_id=config.basket_id
    url=get_public_basket_database_url()
    if not url: raise RuntimeError("Public PostgreSQL is not configured")
    with connect_public_basket_db(url) as conn:
        trust=load_trust_records(conn,basket_id)
        if not trust["active_publications"]: return 0
        versions=[]; tickers=set()
        for publication in reversed(trust["active_publications"]):
            rows=conn.execute("SELECT ticker,target_weight FROM public_portfolio_positions WHERE publication_id=%s",(publication["publication_id"],)).fetchall()
            weights={row["ticker"]:float(row["target_weight"]) for row in rows}; tickers.update(weights)
            versions.append({"publication_id":publication["publication_id"],"as_of":publication["as_of"],"weights":weights})
        backfill_days=config.model_backfill_trading_days
        if backfill_days:
            start=datetime.now(timezone.utc).date()-timedelta(days=max(30,math.ceil(backfill_days*1.7)+14))
        else:
            start=min(item["as_of"] for item in versions).date()-timedelta(days=7)
        data=yf.download(sorted(tickers),start=start.isoformat(),end=(datetime.now(timezone.utc).date()+timedelta(days=1)).isoformat(),
                         auto_adjust=True,progress=False,threads=False,group_by="column")
        closes=data["Close"] if isinstance(data.columns,pd.MultiIndex) else data[["Close"]].rename(columns={"Close":next(iter(tickers))})
        actual_publication_date=trust["current"]["as_of"].date()
        if backfill_days:
            closes=closes.tail(backfill_days+1)
            versions=[{
                "publication_id":trust["current"]["publication_id"],
                "as_of":pd.Timestamp(closes.index[0]).to_pydatetime(),
                "weights":next(item["weights"] for item in versions if item["publication_id"]==trust["current"]["publication_id"]),
            }]
        nav=versioned_model_nav(closes,versions)
        if not nav.empty:
            nav["is_backfill"]=nav["nav_date"].map(lambda day: bool(backfill_days and day<actual_publication_date))
        now=datetime.now(timezone.utc)
        conn.execute("ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS gross_nav DOUBLE PRECISION")
        conn.execute("ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS net_nav DOUBLE PRECISION")
        conn.execute("ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS gross_daily_return DOUBLE PRECISION")
        conn.execute("ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS turnover DOUBLE PRECISION NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS estimated_drag DOUBLE PRECISION NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE daily_nav ADD COLUMN IF NOT EXISTS is_backfill BOOLEAN NOT NULL DEFAULT FALSE")
        for row in nav.to_dict("records"):
            material={"basket_id":basket_id,**row,"calculation_version":CALCULATION_VERSION}
            conn.execute("""INSERT INTO daily_nav (basket_id,nav_date,calculation_version,nav,portfolio_value,cash_value,total_value,
                daily_return,drawdown,input_sha256,calculated_at,gross_nav,net_nav,gross_daily_return,turnover,estimated_drag,is_backfill)
                VALUES (%s,%s,%s,%s,0,0,0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (basket_id,nav_date,calculation_version) DO NOTHING""",
                (basket_id,row["nav_date"],CALCULATION_VERSION,row["nav"],row["daily_return"],row["drawdown"],
                 fingerprint(material),now,row["gross_nav"],row["net_nav"],row["gross_daily_return"],
                 row["turnover"],row["estimated_drag"],row["is_backfill"]))
    return 0


if __name__=="__main__": raise SystemExit(main())
