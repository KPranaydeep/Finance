"""Independent provisional reviews for captured securities before basket entry completes."""
from datetime import date, timedelta
import math
import numpy as np
import pandas as pd

from . import market
from .core import digest
from .costs import charges
from .forecast import estimate


METHOD = "captured-security-one-share-net-xirr-v1"


def _one_share_baseline(publication, entry, policy):
    ticker, kind, price = entry["ticker"], entry["kind"], float(entry["price_inr"])
    fee = charges(price, "BUY", kind, policy)
    if kind == "foreign_us_listing":
        from public_us_funding import fx_gst
        funding = float(fx_gst(price + fee["total"]))
        fee = {**fee, "fx_gst": round(funding, 2),
               "total": round(fee["total"] + funding, 2)}
    outlay = round(price + fee["total"], 2)
    lot = {"ticker": ticker, "quantity": 1, "price": price, "kind": kind,
           "entry_date": entry["entry_date"],
           "entry_at": entry.get("requested_entry_at"),
           "entry_quote_at": entry.get("quote_at"),
           "entry_basis": entry.get("basis"), "entry_source": entry.get("source"),
           "entry_charges": fee}
    return {"baseline_id": "partial:" + publication["publication_id"] + ":" + ticker,
            "publication_id": publication["publication_id"], "basket_id": publication["basket_id"],
            "portfolio_version": int(publication["portfolio_version"]),
            "published_at": str(publication["published_at"]),
            "entry_date": entry["entry_date"], "fully_invested_date": entry["entry_date"],
            "capital": outlay, "cash": 0., "weights": {ticker: 1.}, "lots": [lot],
            "basis": "PROVISIONAL_ONE_SHARE_COSTED_SECURITY_REVIEW"}


def _security_dates(now, entry_date, market_name, policy):
    now = pd.Timestamp(now)
    start = now.date() - timedelta(days=10)
    end = min(now.date() + timedelta(days=120),
              date.fromisoformat(policy["calendar_verified_through"]))
    schedule = market.calendar(start, end, policy, market_name)
    buffer = pd.Timedelta(minutes=policy["assessment_wait_after_close_minutes"])
    completed = schedule[schedule.market_close + buffer <= now]
    if completed.empty:
        raise market.AwaitingMarketEntry(entry_date, schedule.iloc[0].market_close.isoformat())
    as_of = str(completed.index[-1].date())
    future = [str(day.date()) for day in schedule.index
              if str(day.date()) > max(as_of, entry_date)][:policy["max_review_sessions"]]
    if not future:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    return as_of, future


def _single_returns(history, as_of, market_name, policy):
    series = pd.to_numeric(history["Close"], errors="coerce")
    series = series.where(np.isfinite(series) & (series > 0))
    first = series.first_valid_index()
    if first is None:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    dates = [str(day.date()) for day in market.calendar(first, as_of, policy, market_name).index]
    aligned = series.reindex(dates)
    changes = aligned.pct_change(fill_method=None)
    valid = changes.notna() & np.isfinite(changes)
    result = changes.loc[valid].to_frame(history.attrs.get("ticker", "security"))
    result.attrs["session_positions"] = {day: index for index, day in enumerate(dates)}
    return result


def estimate_captured_security(publication, entry, policy, now):
    ticker, market_name = entry["ticker"], entry["market"]
    as_of, future = _security_dates(now, entry["entry_date"], market_name, policy)
    histories = market.fetch([ticker], entry["entry_date"], as_of, policy,
                             allow_incomplete_end=True)
    history = histories[ticker]
    history.attrs["ticker"] = ticker
    returns = _single_returns(history, as_of, market_name, policy)
    if len(returns) < 126:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    current = float(history.loc[as_of, "Close"]) if as_of in history.index else float(entry["price_inr"])
    baseline = _one_share_baseline(publication, entry, policy)
    forecast = estimate(baseline, {ticker: current}, returns, future, policy,
                        baseline["capital"], validation=None)
    crossing = forecast["security_crossings"][0]
    payload = {"status": "PROVISIONAL_RESEARCH", "method": METHOD,
               "publication_id": publication["publication_id"], "ticker": ticker,
               "entry_at": entry.get("requested_entry_at"), "entry_quote_at": entry.get("quote_at"),
               "entry_price_inr": float(entry["price_inr"]), "as_of": as_of,
               "estimated_crossing_date": crossing["crossing_date"],
               "crossing_probability": crossing["probability"],
               "horizon_probability": crossing["horizon_probability"],
               "target_xirr": policy["target_xirr"],
               "notional_basis": "ONE_SHARE_WITH_MODELED_ENTRY_EXIT_COSTS_AND_TAX",
               "checked_at": pd.Timestamp(now).isoformat(),
               "limitations": "Provisional security-only estimate. It excludes basket interactions and is superseded by the completed basket baseline."}
    payload["evidence_hash"] = digest(payload)
    return payload
