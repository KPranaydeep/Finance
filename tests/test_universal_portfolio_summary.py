import pandas as pd

from universal_portfolio_summary import CLUSTER_COLUMNS, summarize_universal_portfolio


def test_empty_universe_has_stable_shape():
    summary, clusters = summarize_universal_portfolio(pd.DataFrame())

    assert summary["total_symbols"] == 0
    assert summary["listing_clusters"] == 0
    assert list(clusters.columns) == CLUSTER_COLUMNS
    assert clusters.empty


def test_summary_classifies_and_clusters_mixed_listings():
    frame = pd.DataFrame(
        [
            {"Symbol": "RELIANCE", "Yahoo Ticker": "RELIANCE.NS", "Exchange": "NSI", "Currency": "INR"},
            {"Symbol": "500325", "Yahoo Ticker": "500325.BO", "Exchange": "BOM", "Currency": "INR"},
            {"Symbol": "VT", "Yahoo Ticker": "VT", "Exchange": "PCX", "Currency": "USD"},
            {"Symbol": "ABBV", "Yahoo Ticker": "ABBV", "Exchange": "NYQ", "Currency": "USD"},
        ]
    )

    summary, clusters = summarize_universal_portfolio(frame)

    assert summary == {
        "total_symbols": 4,
        "india_listed": 2,
        "overseas_listed": 2,
        "listing_clusters": 4,
        "exchanges": ["BOM", "NSI", "NYQ", "PCX"],
        "currencies": ["INR", "USD"],
    }
    assert clusters["Securities"].sum() == 4
    assert clusters["Share of universe"].sum() == 1.0


def test_suffix_fallback_and_duplicate_ticker_are_handled():
    frame = pd.DataFrame(
        [
            {"Symbol": "SBC", "Yahoo Ticker": "SBC.NS", "Exchange": "", "Currency": ""},
            {"Symbol": "SBC duplicate", "Yahoo Ticker": "sbc.ns", "Exchange": "NSI", "Currency": "INR"},
            {"Symbol": "ASX", "Yahoo Ticker": "ASX", "Exchange": "NGM", "Currency": "USD"},
        ]
    )

    summary, clusters = summarize_universal_portfolio(frame)

    assert summary["total_symbols"] == 2
    assert summary["india_listed"] == 1
    assert summary["overseas_listed"] == 1
    india = clusters.loc[clusters["Listing group"].eq("India")].iloc[0]
    assert india["Exchange"] == "Unknown"
    assert india["Currency"] == "Unknown"
