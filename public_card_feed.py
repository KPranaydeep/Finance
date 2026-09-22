"""Shared, versioned card-feed contract for public portfolio evidence pages."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_nav_snapshots import load_nav_snapshot
from public_portfolio_publications import load_trust_records


IST = ZoneInfo("Asia/Kolkata")
CARD_FEED_SCHEMA = "public-portfolio-card-feed"
CARD_FEED_SCHEMA_VERSION = 1


@st.cache_data(ttl=300, show_spinner=False)
def load_public_record(basket_id: str) -> dict[str, Any]:
    """Load the immutable publication history needed by public read-only pages."""
    url = get_public_basket_database_url()
    if not url:
        raise RuntimeError("Public record is not configured")
    with connect_public_basket_db(url) as conn:
        basket = conn.execute(
            "SELECT * FROM public_baskets WHERE basket_id=%s", (basket_id,)
        ).fetchone()
        if not basket:
            return {"basket": None}
        nav = load_nav_snapshot(conn, basket_id)
        trust = load_trust_records(conn, basket_id)
        publication_positions = conn.execute(
            """SELECT p.publication_id,v.portfolio_version,p.ticker,p.target_weight
               FROM public_portfolio_positions p
               JOIN public_portfolio_versions v ON v.publication_id=p.publication_id
               WHERE v.basket_id=%s ORDER BY v.portfolio_version,p.ticker""",
            (basket_id,),
        ).fetchall()
    return {
        "basket": dict(basket),
        "nav": [dict(row) for row in nav],
        "publication_positions": [dict(row) for row in publication_positions],
        **trust,
    }


def _display_date(value: Any) -> str:
    if hasattr(value, "astimezone"):
        return value.astimezone(IST).date().isoformat()
    return str(value)[:10]


def build_card_feed(record: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Build the stable producer/consumer contract for recommendation evidence."""
    published_at = current.get("published_at")
    publication_date = _display_date(published_at)
    active_publications = record.get("active_publications") or [
        publication
        for publication in record.get("publications", [])
        if publication.get("effective_status", "ACTIVE") == "ACTIVE"
    ]
    ordered_publications = sorted(
        active_publications,
        key=lambda publication: int(publication.get("portfolio_version") or 0),
    )
    publication_dates: dict[str, str] = {}
    for publication in ordered_publications:
        publication_id = publication.get("publication_id")
        published = publication.get("published_at") or publication.get("as_of")
        if publication_id and published:
            publication_dates[publication_id] = _display_date(published)

    active_publication_ids = set(publication_dates)
    holdings_by_publication: dict[str, set[str]] = {
        publication_id: set() for publication_id in active_publication_ids
    }
    for position in record.get("publication_positions", []):
        publication_id = position.get("publication_id")
        if publication_id not in active_publication_ids:
            continue
        ticker = str(position.get("ticker", "")).strip()
        if ticker:
            holdings_by_publication[publication_id].add(ticker)

    constituents = record.get("constituents", [])
    current_by_ticker = {
        str(row["ticker"]).strip(): row for row in constituents
    }
    publication_sequence = [
        publication.get("publication_id")
        for publication in ordered_publications
        if publication.get("publication_id") in publication_dates
    ]
    all_tickers = sorted(
        set().union(*(holdings_by_publication.values()))
        if holdings_by_publication
        else set()
    )
    securities = []
    for ticker in all_tickers:
        current_row = current_by_ticker.get(ticker)
        held_indices = [
            index
            for index, publication_id in enumerate(publication_sequence)
            if ticker in holdings_by_publication[publication_id]
        ]
        if not held_indices:
            continue
        last_held_index = max(held_indices)
        lifecycle_start_index = last_held_index
        while (
            lifecycle_start_index > 0
            and ticker
            in holdings_by_publication[publication_sequence[lifecycle_start_index - 1]]
        ):
            lifecycle_start_index -= 1
        entry_date = publication_dates[publication_sequence[lifecycle_start_index]]
        last_allocation_date = publication_dates[publication_sequence[last_held_index]]
        exit_date = (
            publication_dates[publication_sequence[last_held_index + 1]]
            if current_row is None and last_held_index + 1 < len(publication_sequence)
            else None
        )
        securities.append(
            {
                "ticker": ticker,
                "target_weight": float(current_row["target_weight"]) if current_row else 0.0,
                "publication_date": entry_date,
                "entry_date": entry_date,
                "last_allocation_date": last_allocation_date,
                "exit_date": exit_date,
                "status": "active" if current_row else "removed",
                "publication_id": current["publication_id"] if current_row else None,
                "portfolio_version": (
                    f"P{int(current['portfolio_version']):03d}" if current_row else None
                ),
            }
        )

    return {
        "schema": CARD_FEED_SCHEMA,
        "schema_version": CARD_FEED_SCHEMA_VERSION,
        "generated_at": datetime.now(IST).isoformat(),
        "source": "public_card_feed.py",
        "basket_id": current["basket_id"],
        "publication_id": current["publication_id"],
        "portfolio_version": f"P{int(current['portfolio_version']):03d}",
        "publication_date": publication_date,
        "data_as_of": current.get("as_of"),
        "securities": securities,
    }
