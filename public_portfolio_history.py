"""Pure helpers for presenting immutable public-portfolio version history."""

from __future__ import annotations

from typing import Iterable


def latest_entry_exit_changes(
    publications: Iterable[dict], publication_positions: Iterable[dict]
) -> dict | None:
    """Return sized entries and exits for the latest active publication pair."""
    active = sorted(
        (
            dict(row)
            for row in publications
            if row.get("effective_status", "ACTIVE") == "ACTIVE"
        ),
        key=lambda row: int(row["portfolio_version"]),
    )
    if len(active) < 2:
        return None

    weights_by_publication: dict[str, dict[str, float]] = {}
    for row in publication_positions:
        weights_by_publication.setdefault(str(row["publication_id"]), {})[
            str(row["ticker"])
        ] = float(row["target_weight"])

    previous, current = active[-2:]
    before = weights_by_publication.get(str(previous["publication_id"]), {})
    after = weights_by_publication.get(str(current["publication_id"]), {})
    entries = tuple(
        sorted(
            (
                (ticker, after[ticker])
                for ticker in set(after) | set(before)
                if before.get(ticker, 0) <= 0 < after.get(ticker, 0)
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    exits = tuple(
        sorted(
            (
                (ticker, before[ticker])
                for ticker in set(after) | set(before)
                if after.get(ticker, 0) <= 0 < before.get(ticker, 0)
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return {
        "previous_version": f"P{int(previous['portfolio_version']):03d}",
        "current_version": f"P{int(current['portfolio_version']):03d}",
        "entries": entries,
        "exits": exits,
    }


def build_allocation_change_rows(
    publications: Iterable[dict], publication_positions: Iterable[dict]
) -> list[dict]:
    """Summarize target changes between consecutive active publications."""
    active=sorted(
        (dict(row) for row in publications if row.get("effective_status","ACTIVE") == "ACTIVE"),
        key=lambda row:int(row["portfolio_version"]),
    )
    weights_by_publication: dict[str,dict[str,float]]={}
    for row in publication_positions:
        weights_by_publication.setdefault(str(row["publication_id"]),{})[
            str(row["ticker"])
        ]=float(row["target_weight"])

    changes=[]
    for previous,current in zip(active,active[1:]):
        before=weights_by_publication.get(str(previous["publication_id"]),{})
        after=weights_by_publication.get(str(current["publication_id"]),{})
        tickers=sorted(set(before)|set(after))
        added=[ticker for ticker in tickers if before.get(ticker,0)<=0<after.get(ticker,0)]
        removed=[ticker for ticker in tickers if after.get(ticker,0)<=0<before.get(ticker,0)]
        increased=[ticker for ticker in tickers if after.get(ticker,0)>before.get(ticker,0)+1e-12 and ticker not in added]
        decreased=[ticker for ticker in tickers if after.get(ticker,0)<before.get(ticker,0)-1e-12 and ticker not in removed]
        turnover=0.5*sum(abs(after.get(ticker,0)-before.get(ticker,0)) for ticker in tickers)
        changes.append({
            "Change":f"P{int(previous['portfolio_version']):03d} → P{int(current['portfolio_version']):03d}",
            "Published":current.get("published_at"),
            "Added":", ".join(added) if added else "—",
            "Removed":", ".join(removed) if removed else "—",
            "Increased":len(increased),
            "Decreased":len(decreased),
            "Target turnover":turnover,
        })
    return list(reversed(changes))
