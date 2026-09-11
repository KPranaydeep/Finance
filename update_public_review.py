"""Owner-configured background monitor, isolated from NAV/forecast production jobs."""
import json
import os
import sys
from datetime import datetime, timezone
from public_review.config import load_policy
from public_review.notifications import send


def main():
    if os.getenv("PUBLIC_REVIEW_ENABLED", "false").lower() != "true":
        print(json.dumps({"status": "DISABLED", "reason": "Owner setup required; no database writes"}))
        return 0
    try:
        policy = load_policy(datetime.now(timezone.utc).date())
        from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
        from public_review.service import run, publications
        from public_review.instruments import complete_policy
        url = get_public_basket_database_url()
        if not url:
            raise ValueError("DATABASE_CONFIGURATION_REQUIRED")
        with connect_public_basket_db(url) as conn:
            pubs = publications(conn, os.getenv("PUBLIC_BASKET_ID", "PUBLIC-01"))
            policy = complete_policy(policy, {t for p in pubs for t in p["weights"]})
            result = run(conn, os.getenv("PUBLIC_BASKET_ID", "PUBLIC-01"), policy,
                         acknowledge=os.getenv("PUBLIC_REVIEW_ACK_BASELINE") or None)
        print(json.dumps(result))
        return 1 if result["failed"] else 0
    except Exception as exc:
        # Workflow failure notifications remain a backup if the chosen channel fails.
        try:
            send("Public portfolio monitoring is unavailable. Check the review policy, credentials and workflow. No trade instruction was generated.")
        except Exception:
            pass
        safe_codes = {"POLICY_APPROVAL_REQUIRED", "UNSUPPORTED_TAX_OR_ACCOUNT_PROFILE",
                      "TARIFF_REVIEW_REQUIRED", "CALENDAR_REVIEW_REQUIRED", "INTEGER_POLICY_REQUIRED",
                      "NSE_CLASSIFICATION_REQUIRED", "INVALID_CAPITAL", "DATABASE_CONFIGURATION_REQUIRED",
                      "NO_ACTIVE_PUBLICATION", "UNKNOWN_BASELINE", "INSTRUMENT_CLASSIFICATION_REQUIRED",
                      "INSTRUMENT_METADATA_UNAVAILABLE", "POLICY_UPDATE_BUSY", "POLICY_CHANGED_RETRY",
                      "FOREIGN_REVIEW_COST_MODEL_REQUIRED", "INVALID_POLICY_ENTRY_QUOTE_INTERVAL",
                      "ENTRY_INTRADAY_HISTORY_UNAVAILABLE"}
        code = str(exc) if isinstance(exc, ValueError) and str(exc) in safe_codes else "REVIEW_SETUP_OR_STORAGE_FAILED"
        print(json.dumps({"status": "CANNOT_ASSESS", "reason": code,
                          "note": "No trade was submitted. Raw exceptions and credentials are not logged."}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
