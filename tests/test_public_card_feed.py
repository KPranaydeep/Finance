from datetime import date
from io import BytesIO
from unittest.mock import patch

import matplotlib.image as mpimg
import pandas as pd
import pytest
from matplotlib.figure import Figure

from public_card_feed import basket_since_launch_return, build_card_feed
from public_track_record import (
    allocation_weighted_return,
    analyze,
    batch_summary_card,
    lifecycle_vwap_inr,
    load_portfolio_summaries,
    portfolio_cover_card,
    whatsapp_card,
)


def test_card_feed_tracks_active_and_exited_lifecycles():
    record = {
        "active_publications": [
            {
                "publication_id": "PUB-1",
                "portfolio_version": 1,
                "published_at": "2026-09-01T12:00:00+00:00",
            },
            {
                "publication_id": "PUB-2",
                "portfolio_version": 2,
                "published_at": "2026-09-15T12:00:00+00:00",
            },
        ],
        "publication_positions": [
            {"publication_id": "PUB-1", "ticker": "KEEP.NS", "target_weight": 0.6},
            {"publication_id": "PUB-1", "ticker": "EXIT.NS", "target_weight": 0.4},
            {"publication_id": "PUB-2", "ticker": "KEEP.NS", "target_weight": 1.0},
        ],
        "constituents": [{"ticker": "KEEP.NS", "target_weight": 1.0}],
    }
    current = {
        "basket_id": "PUBLIC-01",
        "publication_id": "PUB-2",
        "portfolio_version": 2,
        "published_at": "2026-09-15T12:00:00+00:00",
        "as_of": "2026-09-15T12:00:00+00:00",
    }

    feed = build_card_feed(record, current)
    rows = {row["ticker"]: row for row in feed["securities"]}

    assert feed["schema"] == "public-portfolio-card-feed"
    assert rows["KEEP.NS"]["status"] == "active"
    assert rows["KEEP.NS"]["entry_date"] == "2026-09-01"
    assert rows["KEEP.NS"]["target_weight"] == 1.0
    assert rows["EXIT.NS"]["status"] == "removed"
    assert rows["EXIT.NS"]["exit_date"] == "2026-09-15"
    assert rows["EXIT.NS"]["target_weight"] == pytest.approx(0.4)


def test_basket_since_launch_excludes_development_backfill():
    rows = [
        {"nav_date": "2026-08-31", "net_nav": 50.0, "is_backfill": True},
        {"nav_date": "2026-09-01", "net_nav": 100.0, "is_backfill": False},
        {"nav_date": "2026-09-15", "net_nav": 108.0, "is_backfill": False},
    ]

    assert basket_since_launch_return(rows) == pytest.approx(0.08)


def test_reentered_security_uses_current_lifecycle_start():
    publications = [
        {
            "publication_id": f"PUB-{version}",
            "portfolio_version": version,
            "published_at": f"2026-09-{version:02d}T12:00:00+00:00",
        }
        for version in range(1, 4)
    ]
    record = {
        "active_publications": publications,
        "publication_positions": [
            {"publication_id": "PUB-1", "ticker": "RETURN.NS", "target_weight": 1.0},
            {"publication_id": "PUB-2", "ticker": "OTHER.NS", "target_weight": 1.0},
            {"publication_id": "PUB-3", "ticker": "RETURN.NS", "target_weight": 1.0},
        ],
        "constituents": [{"ticker": "RETURN.NS", "target_weight": 1.0}],
    }
    current = {
        "basket_id": "PUBLIC-01",
        "publication_id": "PUB-3",
        "portfolio_version": 3,
        "published_at": "2026-09-03T12:00:00+00:00",
        "as_of": "2026-09-03T12:00:00+00:00",
    }

    feed = build_card_feed(record, current)
    row = next(item for item in feed["securities"] if item["ticker"] == "RETURN.NS")

    assert row["status"] == "active"
    assert row["entry_date"] == "2026-09-03"


def _prices(values):
    index = pd.to_datetime(["2026-09-01", "2026-09-02"])
    return pd.DataFrame({"Close": values, "Adj Close": values}, index=index)


def test_us_security_return_is_converted_to_inr_before_comparison():
    ticker = _prices([10.0, 10.0])
    nifty = _prices([100.0, 100.0])
    world = _prices([50.0, 50.0])
    fx = _prices([80.0, 160.0])

    metrics, chart = analyze(
        ticker,
        nifty,
        world,
        fx,
        date(2026, 9, 1),
        ticker_currency="USD",
    )

    assert metrics["ticker_return"] == pytest.approx(1.0)
    assert metrics["world_return"] == pytest.approx(1.0)
    assert metrics["entry_close"] == 10.0
    assert metrics["price_symbol"] == "$"
    assert chart["Ticker"].iloc[-1] == pytest.approx(200.0)


def test_indian_security_return_stays_in_inr():
    ticker = _prices([100.0, 110.0])
    flat = _prices([100.0, 100.0])
    fx = _prices([80.0, 160.0])

    metrics, _ = analyze(
        ticker,
        flat,
        flat,
        fx,
        date(2026, 9, 1),
        ticker_currency="INR",
    )

    assert metrics["ticker_return"] == pytest.approx(0.1)
    assert metrics["price_symbol"] == "₹"


def test_portfolio_cover_slide_requires_no_market_history():
    feed = {
        "portfolio_version": "P004",
        "publication_date": "2026-09-20",
        "publication_id": "PUB-EXAMPLE",
    }
    securities = [
        {"ticker": "AAA.NS", "target_weight": 0.6},
        {"ticker": "BBB.NS", "target_weight": 0.4},
    ]

    image = portfolio_cover_card(
        feed, securities, scope_label="Current holdings"
    )

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert mpimg.imread(BytesIO(image), format="png").shape[:2] == (2000, 1272)
    assert len(image) > 10_000


def test_portfolio_summary_card_renders_empirical_outcomes():
    feed = {
        "portfolio_version": "P008",
        "publication_date": "2026-09-15",
    }
    summaries = [
        {"ticker": "GAIN.NS", "return": 1.4151, "vwap_return": .20,
         "target_weight": .7, "status": "active"},
        {"ticker": "LOSS.NS", "return": -0.0951, "vwap_return": -.05,
         "target_weight": .3, "status": "removed"},
    ]

    captured = {}
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        captured["text"] = "\n".join(item.get_text() for item in figure.texts)
        return savefig(figure, *args, **kwargs)

    with patch.object(Figure, "savefig", inspect):
        image = batch_summary_card(feed, summaries, basket_return=.084)
    pixels = mpimg.imread(BytesIO(image), format="png")

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert pixels.shape[:2] == (1100, 1272)
    assert len(image) > 10_000
    assert "+141.51%" in captured["text"]
    assert "-9.51%" in captured["text"]
    assert "+20.00%" not in captured["text"]
    assert "BASKET SINCE FIRST PUBLICATION" in captured["text"]
    assert "REALIZED EXITS · WEIGHTED RETURN" in captured["text"]


def test_portfolio_summary_card_pending_state_stays_compact():
    feed = {"portfolio_version": "P010", "publication_date": "2026-09-30"}
    image = batch_summary_card(feed, [], basket_return=None)
    pixels = mpimg.imread(BytesIO(image), format="png")

    assert pixels.shape[:2] == (1160, 1272)


def test_lifecycle_vwap_is_inr_adjusted_and_allocation_weighted():
    index = pd.to_datetime(["2026-09-01", "2026-09-02"])
    security = pd.DataFrame(
        {
            "High": [11.0, 13.0], "Low": [9.0, 11.0],
            "Close": [10.0, 12.0], "Volume": [100.0, 300.0],
        },
        index=index,
    )
    fx = pd.DataFrame({"Close": [80.0, 82.0]}, index=index)
    vwap, endpoint = lifecycle_vwap_inr(
        security, fx, date(2026, 9, 1), ticker_currency="USD"
    )
    expected = (10.0 * 80.0 * 100.0 + 12.0 * 82.0 * 300.0) / 400.0
    assert vwap == pytest.approx(expected)
    assert endpoint == pytest.approx(12.0 * 82.0)
    assert allocation_weighted_return([
        {"return": .10, "target_weight": .8},
        {"return": -.20, "target_weight": .2},
    ]) == pytest.approx(.04)


def test_security_share_card_uses_standard_dimensions():
    index = pd.to_datetime(["2026-09-01", "2026-09-02"])
    chart = pd.DataFrame(
        {
            "Ticker": [100.0, 110.0],
            "Nifty 50": [100.0, 101.0],
            "Global stocks — VT (INR)": [100.0, 102.0],
        },
        index=index,
    )
    metrics = {
        "requested_start": "2026-09-01",
        "as_of": "2026-09-02",
        "calendar_days": 1,
        "ticker_sessions": 2,
        "ticker_return": .10,
        "benchmark_return": .01,
        "world_return": .02,
        "excess_return": .09,
        "excess_world_return": .08,
        "entry_close": 100.0,
        "endpoint_price": 110.0,
        "endpoint_label": "Latest completed close",
        "endpoint_as_of": "2026-09-02",
        "price_symbol": "₹",
    }
    image = whatsapp_card("TEST.NS", metrics, chart)
    assert mpimg.imread(BytesIO(image), format="png").shape[:2] == (2000, 1272)


def test_bulk_summary_downloads_history_once_and_converts_us_return(monkeypatch):
    index = pd.to_datetime(["2026-09-01", "2026-09-02"])
    values = {
        "AAA.NS": [100.0, 110.0],
        "USX": [10.0, 10.0],
        "^NSEI": [100.0, 100.0],
        "VT": [50.0, 50.0],
        "INR=X": [80.0, 160.0],
    }
    columns = pd.MultiIndex.from_product(
        [values, ["Close", "Adj Close", "Volume"]], names=["Ticker", "Price"]
    )
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)
    for ticker, prices in values.items():
        frame[(ticker, "Close")] = prices
        frame[(ticker, "Adj Close")] = prices
        frame[(ticker, "Volume")] = [100.0, 200.0]
    calls = []

    def fake_download(*args, **kwargs):
        calls.append((args, kwargs))
        return frame

    monkeypatch.setattr("public_track_record.yf.download", fake_download)
    load_portfolio_summaries.clear()
    summaries, failures = load_portfolio_summaries(
        "PUB-BULK-TEST",
        (
            ("AAA.NS", "2026-09-01", None, "active"),
            ("USX", "2026-09-01", None, "active"),
        ),
        "unique-test-bucket",
    )
    load_portfolio_summaries.clear()

    returns = {row["ticker"]: row["return"] for row in summaries}
    assert failures == []
    assert returns["AAA.NS"] == pytest.approx(0.1)
    assert returns["USX"] == pytest.approx(1.0)
    assert len(calls) == 1
