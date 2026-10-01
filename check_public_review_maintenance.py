"""Fail expired review-policy maintenance and warn before it becomes blocking."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from public_review.maintenance import policy_maintenance_status


DEFAULT_POLICY = Path(__file__).resolve().parent / "public_review_policy.json"


def _message(kind: str, item: dict) -> str:
    if kind == "tariff":
        subject = "Transaction-cost assumptions"
    else:
        subject = "Market calendar"
    if item["state"] == "expired":
        return f"{subject} expired after {item['valid_through']}; verify and update the policy evidence."
    return (
        f"{subject} must be reverified by {item['valid_through']} "
        f"({item['days_remaining']} days remaining)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default=os.environ.get("PUBLIC_REVIEW_POLICY_PATH", str(DEFAULT_POLICY)),
    )
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--warning-days", type=int, default=30)
    args = parser.parse_args(argv)

    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    result = policy_maintenance_status(
        policy, today=args.as_of, warning_days=args.warning_days
    )
    print(json.dumps(result, sort_keys=True))

    for kind in ("tariff", "calendar"):
        item = result[kind]
        if item["state"] == "review_due":
            print(f"::warning title=Public review maintenance::{_message(kind, item)}")
        elif item["state"] == "expired":
            print(f"::error title=Public review maintenance::{_message(kind, item)}")
    return 1 if result["status"] == "expired" else 0


if __name__ == "__main__":
    raise SystemExit(main())
