"""Export the latest verified public read model after a successful writer run."""

from __future__ import annotations

import argparse

from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    connect_public_basket_db,
    get_public_basket_database_url,
)
from public_card_feed import load_public_record_from_database
from public_record_snapshot import build_snapshot, write_snapshot
from public_review.store import read as read_review_events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="public_record_snapshot.json")
    parser.add_argument("--basket-id", default=DEFAULT_BASKET_ID)
    args = parser.parse_args()

    url = get_public_basket_database_url()
    if not url:
        raise RuntimeError("Public PostgreSQL is not configured")
    with connect_public_basket_db(url) as conn:
        record = load_public_record_from_database(conn, args.basket_id)
        review_events = read_review_events(conn, args.basket_id)
    snapshot = build_snapshot(
        record,
        basket_id=args.basket_id,
        review_events=review_events,
    )
    write_snapshot(args.output, snapshot)
    print(
        f"verified public snapshot written: {args.output} "
        f"({len(review_events)} review events)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
