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


def filter_configured_backfill_findings(
    findings: list[str], trust: dict, *, backfill_trading_days: int
) -> list[str]:
    """Allow only the explicit forecast provenance marker while backfill is enabled.

    The general production scanner remains strict.  This narrow exception disappears
    automatically when PUBLIC_MODEL_BACKFILL_TRADING_DAYS is returned to zero.
    """
    if backfill_trading_days <= 0:
        return findings

    allowed_findings=set()
    for collection in ("active_forecasts", "forecasts"):
        for index,row in enumerate(trust.get(collection, [])):
            forecast_json=row.get("forecast_json") or {}
            if forecast_json.get("history_source") == "DEVELOPMENT_BACKFILL":
                allowed_findings.add(
                    f"Non-production marker at $.{collection}[{index}].forecast_json.history_source"
                )
    return [finding for finding in findings if finding not in allowed_findings]


def main() -> int:
    config=load_public_portfolio_config(require_production=True)
    basket_id=config.basket_id
    failures=[]; warnings=[]; url=get_public_basket_database_url()
    if not url: failures.append("production database configuration is missing")
    else:
        with connect_public_basket_db(url) as conn:
            trust=load_trust_records(conn,basket_id)
            nav=conn.execute("""SELECT DISTINCT ON (nav_date) nav_date,nav,total_value,is_backfill FROM daily_nav
                WHERE basket_id=%s ORDER BY nav_date,calculation_version DESC""",(basket_id,)).fetchall()
        if not trust["current"]: failures.append("latest published portfolio is missing")
        total=sum(float(row["target_weight"]) for row in trust["constituents"])+(float(trust["current"]["cash_weight"]) if trust["current"] else 0)
        if trust["current"] and abs(total-1)>1e-6: failures.append("portfolio weights do not sum to one")
        if not performance_metrics([dict(r) for r in nav]): failures.append("performance history is unavailable")
        if current_forecast_is_due([dict(r) for r in nav],trust):
            failures.append("14-day forecast is unavailable for the current publication")
        if not verify_trust_audit(trust["audit"],basket_id)[0]: failures.append("basket audit verification failed")
        nav_rows=[dict(r) for r in nav]
        contains_backfill=any(bool(row.get("is_backfill")) for row in nav_rows)
        if contains_backfill and config.model_backfill_trading_days <= 0:
            failures.append("development backfill remains while PUBLIC_MODEL_BACKFILL_TRADING_DAYS is zero")
        elif contains_backfill:
            warnings.append(
                f"development backfill is enabled for up to {config.model_backfill_trading_days} trading days"
            )
        findings=inspect_public_data({**trust,"nav":nav_rows},production=True)
        findings=filter_configured_backfill_findings(
            findings,trust,backfill_trading_days=config.model_backfill_trading_days
        )
        if findings: failures.extend(findings)
    print(json.dumps({"status":"FAIL" if failures else "PASS","checks_failed":failures,"warnings":warnings},indent=2))
    return 1 if failures else 0


if __name__=="__main__": raise SystemExit(main())
