"""Completed NSE sessions and strict synchronized Yahoo histories; no filling."""
from datetime import timedelta
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


def fetch(tickers, entry_day, as_of, policy):
    start = min(pd.Timestamp(entry_day), pd.Timestamp(as_of) - pd.DateOffset(years=policy["history_years"]))
    result = {}
    for t in sorted(tickers):
        if not t.endswith(".NS"):
            raise ValueError("ONLY_NSE_DELIVERY_SUPPORTED")
        history = yf.Ticker(t).history(start=str(start.date()),
                    end=str((pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()),
                    auto_adjust=False, actions=True, repair=False, timeout=15)
        if history.empty:
            raise ValueError("MISSING_MARKET_HISTORY")
        history.index = pd.Index([str(d.date()) for d in history.index])
        if history.index.duplicated().any() or entry_day not in history.index or as_of not in history.index:
            raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
        if history.loc[entry_day, "Open"] <= 0 or history.loc[as_of, "Close"] <= 0 or history.loc[as_of, "Volume"] <= 0:
            raise ValueError("INVALID_OR_NONTRADING_PRICE")
        result[t] = history.loc[history.index <= as_of]
    return result
