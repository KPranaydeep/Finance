"""Completed review sessions and strict INR Yahoo histories; no filling."""
from datetime import timedelta
import math
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


class AwaitingMarketEntry(ValueError):
    """Normal pending state; no usable completed entry session yet."""
    def __init__(self, entry_date, ready_at):
        super().__init__("AWAITING_MARKET_ENTRY")
        self.entry_date = entry_date
        self.ready_at = ready_at


def calendar(start, end, policy):
    if str(end)[:10] > policy["calendar_verified_through"]:
        raise ValueError("CALENDAR_REVIEW_REQUIRED")
    return mcal.get_calendar("NSE").schedule(start_date=start, end_date=end)


def sessions(now, published_at, policy):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("AWARE_TIME_REQUIRED")
    published = pd.Timestamp(published_at)
    if published.tzinfo is None:
        raise ValueError("AWARE_PUBLICATION_TIME_REQUIRED")
    local = now.tz_convert("Asia/Kolkata").date()
    end = min(str(local + timedelta(days=100)), policy["calendar_verified_through"])
    schedule = calendar(published.tz_convert("Asia/Kolkata").date() - timedelta(days=7), end, policy)
    eligible = schedule[schedule.market_open > published]
    if eligible.empty:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    ready_at = eligible.iloc[0].market_close + pd.Timedelta(minutes=30)
    if ready_at > now:
        raise AwaitingMarketEntry(str(eligible.index[0].date()), ready_at.isoformat())
    completed = schedule[schedule.market_close + pd.Timedelta(minutes=30) <= now]
    future = schedule[schedule.market_open > now].iloc[:policy["max_review_sessions"]]
    if completed.empty or len(future) < policy["max_review_sessions"]:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    return str(eligible.index[0].date()), str(completed.index[-1].date()), [str(d.date()) for d in future.index]


def fetch(tickers, entry_day, as_of, policy, *, allow_incomplete_end=False):
    start = min(pd.Timestamp(entry_day), pd.Timestamp(as_of) - pd.DateOffset(years=policy["history_years"]))
    result = {}
    usd = [t for t in tickers if not t.endswith(".NS")]
    fx = None
    if usd:
        fx = yf.Ticker("INR=X").history(start=str(start.date()),
                    end=str((pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()),
                    auto_adjust=False, actions=False, repair=False, timeout=15)
        if fx.empty:
            raise ValueError("MISSING_MARKET_HISTORY")
        fx.index = pd.Index([str(d.date()) for d in fx.index])
        if fx.index.duplicated().any():
            raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
    for t in sorted(tickers):
        history = yf.Ticker(t).history(start=str(start.date()),
                    end=str((pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()),
                    auto_adjust=False, actions=True, repair=False, timeout=15)
        if history.empty:
            raise ValueError("MISSING_MARKET_HISTORY")
        history.index = pd.Index([str(d.date()) for d in history.index])
        if history.index.duplicated().any():
            raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
        if not t.endswith(".NS"):
            # Exact-date conversion only. No forward fill: a missing bank-FX
            # observation makes that security-session unusable.
            aligned = fx.reindex(history.index)
            for column in ("Open", "High", "Low", "Close"):
                if column in history:
                    rate_column = column if column in aligned else "Close"
                    history[column] = pd.to_numeric(history[column], errors="coerce") * pd.to_numeric(aligned[rate_column], errors="coerce")
            if "Dividends" in history:
                history["Dividends"] = pd.to_numeric(history["Dividends"], errors="coerce") * pd.to_numeric(aligned["Close"], errors="coerce")
            history.attrs["quote_currency"] = "USD"
            history.attrs["valuation_currency"] = "INR"
        if not allow_incomplete_end:
            if entry_day not in history.index or as_of not in history.index:
                raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
            values = [history.loc[entry_day, "Open"], history.loc[as_of, "Close"], history.loc[as_of, "Volume"]]
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
                raise ValueError("INVALID_OR_NONTRADING_PRICE")
        result[t] = history.loc[history.index <= as_of]
    return result


def synchronized_dates(histories, entry_day, as_of, *, new_baseline):
    """Choose completed dates shared by NSE, US listings and USD/INR.

    A mixed basket checked after the NSE close normally uses the preceding US
    close. This is deliberate and is surfaced through the returned as-of date.
    """
    common = None
    for history in histories.values():
        valid = set(history.index[pd.to_numeric(history["Close"], errors="coerce").map(
            lambda value: math.isfinite(float(value)) and float(value) > 0)])
        common = valid if common is None else common & valid
    common = sorted(d for d in (common or set()) if entry_day <= d <= as_of)
    if not common:
        if new_baseline:
            raise AwaitingMarketEntry(entry_day, None)
        raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
    entry = common[0] if new_baseline else entry_day
    if entry not in common:
        raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
    for history in histories.values():
        values = (history.loc[entry, "Open"], history.loc[common[-1], "Close"],
                  history.loc[common[-1], "Volume"])
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise ValueError("INVALID_OR_NONTRADING_PRICE")
    return entry, common[-1]
