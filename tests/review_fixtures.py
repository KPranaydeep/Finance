import json
from pathlib import Path
from public_review.core import freeze


def policy():
    p = json.loads((Path(__file__).resolve().parents[1] / "public_review_policy.json").read_text())
    p["policy_approved"] = True
    p["instrument_kinds"] = {"A.NS": "equity", "B.NS": "listed_non_equity_etf"}
    return p


def baseline():
    p = policy()
    return freeze({"publication_id": "PUB-TEST", "basket_id": "TEST", "portfolio_version": 1,
                   "published_at": "2026-05-01T10:00:00+00:00"},
                  {"A.NS": .5, "B.NS": .5}, {"A.NS": 100., "B.NS": 50.},
                  "2026-05-04", 10000., p["instrument_kinds"], p, "2026-05-04T12:00:00+00:00")

