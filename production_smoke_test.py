"""Non-destructive production acceptance checks. Exits non-zero on any failure."""

from __future__ import annotations

import json
import sys

from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_portfolio_config import load_public_portfolio_config
from public_portfolio_publications import load_trust_records, verify_trust_audit
from public_portfolio_trust import performance_metrics
from public_release_checks import inspect_public_data


def current_forecast_is_due(nav_rows: list[dict], trust: dict, minimum_nav_rows: int = 61) -> bool:
    """Return whether the current publication has enough NAV history to require a forecast."""
    if len(nav_rows) < minimum_nav_rows or not trust.get("current"):
        return False
    current_id=trust["current"]["publication_id"]
    return not any(row.get("publication_id") == current_id for row in trust.get("forecasts", []))


def main() -> int:
    config=load_public_portfolio_config(require_production=True)
    basket_id=config.basket_id
    failures=[]; url=get_public_basket_database_url()
    if not url: failures.append("production database configuration is missing")
    else:
        with connect_public_basket_db(url) as conn:
            trust=load_trust_records(conn,basket_id)
            nav=conn.execute("SELECT nav_date,nav,total_value FROM daily_nav WHERE basket_id=%s ORDER BY nav_date",(basket_id,)).fetchall()
        if not trust["current"]: failures.append("latest published portfolio is missing")
        total=sum(float(row["target_weight"]) for row in trust["constituents"])+(float(trust["current"]["cash_weight"]) if trust["current"] else 0)
        if trust["current"] and abs(total-1)>1e-6: failures.append("portfolio weights do not sum to one")
        if not performance_metrics([dict(r) for r in nav]): failures.append("performance history is unavailable")
        if current_forecast_is_due([dict(r) for r in nav],trust):
            failures.append("14-day forecast is unavailable for the current publication")
        if not verify_trust_audit(trust["audit"],basket_id)[0]: failures.append("basket audit verification failed")
        findings=inspect_public_data({**trust,"nav":[dict(r) for r in nav]},production=True)
        if findings: failures.extend(findings)
    print(json.dumps({"status":"FAIL" if failures else "PASS","checks_failed":failures},indent=2))
    return 1 if failures else 0


if __name__=="__main__": raise SystemExit(main())
