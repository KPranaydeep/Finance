"""Small, deterministic summaries for the shared optimization universe."""

from __future__ import annotations

import pandas as pd


CLUSTER_COLUMNS = [
    "Listing group",
    "Exchange",
    "Currency",
    "Securities",
    "Share of universe",
]


def _clean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def summarize_universal_portfolio(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return headline counts and exchange/currency clusters for a universe.

    A candidate is counted once by Yahoo ticker (falling back to Symbol). Indian
    listings are identified from their currency, exchange, or Yahoo suffix so
    older backups with partial metadata still classify sensibly.
    """
    empty_summary = {
        "total_symbols": 0,
        "india_listed": 0,
        "overseas_listed": 0,
        "listing_clusters": 0,
        "exchanges": [],
        "currencies": [],
    }
    if frame is None or frame.empty:
        return empty_summary, pd.DataFrame(columns=CLUSTER_COLUMNS)

    work = frame.copy()
    symbol = _clean_series(work, "Symbol").str.upper()
    ticker = _clean_series(work, "Yahoo Ticker").str.upper()
    exchange = _clean_series(work, "Exchange").str.upper()
    currency = _clean_series(work, "Currency").str.upper()

    identity = ticker.where(ticker.ne(""), symbol)
    work = work.assign(
        _identity=identity,
        _ticker=ticker,
        _exchange=exchange,
        _currency=currency,
    )
    work = work[work["_identity"].ne("")].drop_duplicates("_identity", keep="first")
    if work.empty:
        return empty_summary, pd.DataFrame(columns=CLUSTER_COLUMNS)

    india_mask = (
        work["_currency"].eq("INR")
        | work["_exchange"].isin({"NSE", "NSI", "BSE", "BOM"})
        | work["_ticker"].str.endswith((".NS", ".BO"))
    )
    work["Listing group"] = india_mask.map({True: "India", False: "Overseas"})
    work["Exchange"] = work["_exchange"].replace("", "Unknown")
    work["Currency"] = work["_currency"].replace("", "Unknown")

    total = int(len(work))
    clusters = (
        work.groupby(["Listing group", "Exchange", "Currency"], dropna=False)
        .size()
        .rename("Securities")
        .reset_index()
    )
    clusters["Share of universe"] = clusters["Securities"] / total
    group_order = pd.Categorical(
        clusters["Listing group"], categories=["India", "Overseas"], ordered=True
    )
    clusters = (
        clusters.assign(_group_order=group_order)
        .sort_values(
            ["_group_order", "Securities", "Exchange", "Currency"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop(columns="_group_order")
        .reset_index(drop=True)
    )

    known_exchanges = sorted(set(work.loc[work["Exchange"].ne("Unknown"), "Exchange"]))
    known_currencies = sorted(set(work.loc[work["Currency"].ne("Unknown"), "Currency"]))
    summary = {
        "total_symbols": total,
        "india_listed": int(india_mask.sum()),
        "overseas_listed": int((~india_mask).sum()),
        "listing_clusters": int(len(clusters)),
        "exchanges": known_exchanges,
        "currencies": known_currencies,
    }
    return summary, clusters[CLUSTER_COLUMNS]
