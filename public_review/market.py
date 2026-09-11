"""Completed review sessions and strict INR Yahoo histories; no filling."""
from datetime import datetime, timezone, timedelta
import math
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf


KIND_CALENDARS = {
    "equity": "NSE",
    "equity_etf": "NSE",
    "listed_non_equity_etf": "NSE",
    "specified_debt_etf": "NSE",
    "foreign_us_listing": "NYSE",
}
ENTRY_MODEL_VERSION = "per-security-publication-or-open-plus-wait-v1"


class AwaitingMarketEntry(ValueError):
    """Normal pending state; no usable completed entry session yet."""
    def __init__(self, entry_date, ready_at):
        super().__init__("AWAITING_MARKET_ENTRY")
        self.entry_date = entry_date
        self.ready_at = ready_at


def _data_retry(entry, now, ticker, reason="ENTRY_DATA_RETRY"):
    pending = AwaitingMarketEntry(
        entry["entry_date"],
        (pd.Timestamp(now) + pd.Timedelta(minutes=30)).isoformat())
    pending.planned_entry = entry
    pending.wait_reason = reason
    pending.pending_ticker = ticker
    return pending


def calendar(start, end, policy, market="NSE"):
    if str(end)[:10] > policy["calendar_verified_through"]:
        raise ValueError("CALENDAR_REVIEW_REQUIRED")
    return mcal.get_calendar(market).schedule(start_date=start, end_date=end)


def joint_calendar(start, end, policy, instrument_kinds=None):
    """Sessions shared by every market represented in the portfolio.

    ``market_open`` is the first opening and ``all_markets_open`` the last
    opening on the shared date. ``market_close`` is the last closing. This
    lets entry and assessment use separate, globally correct readiness gates.
    """
    markets = {KIND_CALENDARS.get(kind) for kind in (instrument_kinds or {}).values()}
    if None in markets:
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    markets = markets or {"NSE"}
    schedules = [calendar(start, end, policy, market) for market in sorted(markets)]
    common = set(str(day.date()) for day in schedules[0].index)
    for schedule in schedules[1:]:
        common &= {str(day.date()) for day in schedule.index}
    rows = []
    for day in sorted(common):
        sessions = [schedule.loc[next(index for index in schedule.index
                                     if str(index.date()) == day)] for schedule in schedules]
        rows.append({"date": day,
                     "market_open": min(row.market_open for row in sessions),
                     "all_markets_open": max(row.market_open for row in sessions),
                     "market_close": max(row.market_close for row in sessions)})
    if not rows:
        return pd.DataFrame(columns=["market_open", "all_markets_open", "market_close"])
    return pd.DataFrame(rows).set_index(pd.to_datetime([row["date"] for row in rows]))


def entry_session(now, published_at, policy, instrument_kinds=None):
    """Return the first shared session once every required market has opened."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("AWARE_TIME_REQUIRED")
    published = pd.Timestamp(published_at)
    if published.tzinfo is None:
        raise ValueError("AWARE_PUBLICATION_TIME_REQUIRED")
    local = now.tz_convert("Asia/Kolkata").date()
    end = min(str(local + timedelta(days=100)), policy["calendar_verified_through"])
    schedule = joint_calendar(published.date() - timedelta(days=7), end, policy,
                              instrument_kinds)
    # Every constituent must have been tradable after publication. Comparing
    # against the earliest opening excludes a partially elapsed global date.
    eligible = schedule[schedule.market_open > published]
    if eligible.empty:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    row = eligible.iloc[0]
    ready_at = row.all_markets_open + pd.Timedelta(
        minutes=policy["entry_wait_after_open_minutes"])
    entry_date = str(eligible.index[0].date())
    if ready_at > now:
        raise AwaitingMarketEntry(entry_date, ready_at.isoformat())
    return entry_date, ready_at.isoformat()


def sessions(now, published_at, policy, instrument_kinds=None):
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("AWARE_TIME_REQUIRED")
    published = pd.Timestamp(published_at)
    if published.tzinfo is None:
        raise ValueError("AWARE_PUBLICATION_TIME_REQUIRED")
    local = now.tz_convert("Asia/Kolkata").date()
    end = min(str(local + timedelta(days=100)), policy["calendar_verified_through"])
    schedule = joint_calendar(published.date() - timedelta(days=7), end, policy,
                              instrument_kinds)
    eligible = schedule[schedule.market_open > published]
    if eligible.empty:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    assessment_buffer = pd.Timedelta(minutes=policy["assessment_wait_after_close_minutes"])
    ready_at = eligible.iloc[0].market_close + assessment_buffer
    if ready_at > now:
        raise AwaitingMarketEntry(str(eligible.index[0].date()), ready_at.isoformat())
    completed = schedule[schedule.market_close + assessment_buffer <= now]
    future = schedule[schedule.market_open > now].iloc[:policy["max_review_sessions"]]
    if completed.empty or len(future) < policy["max_review_sessions"]:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    return str(eligible.index[0].date()), str(completed.index[-1].date()), [str(d.date()) for d in future.index]


def fetch_entry(tickers, entry_day, policy, ready_at=None):
    """Fetch a verifiable opening price without requiring the session close."""
    histories = fetch(tickers, entry_day, entry_day, policy, allow_incomplete_end=True)
    for history in histories.values():
        if entry_day not in history.index:
            raise AwaitingMarketEntry(entry_day, ready_at)
        values = (history.loc[entry_day, "Open"], history.loc[entry_day, "Volume"])
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise AwaitingMarketEntry(entry_day, ready_at)
    return histories


def security_entry_schedule(now, published_at, policy, instrument_kinds):
    """Return each ticker's immutable requested entry timestamp.

    If its exchange is trading at publication, the publication timestamp is
    used. Otherwise the timestamp is the next exchange open plus the configured
    waiting period. No shared-market delay is introduced here.
    """
    now = pd.Timestamp(now)
    published = pd.Timestamp(published_at)
    if now.tzinfo is None:
        raise ValueError("AWARE_TIME_REQUIRED")
    if published.tzinfo is None:
        raise ValueError("AWARE_PUBLICATION_TIME_REQUIRED")
    end = min(str(now.tz_convert("Asia/Kolkata").date() + timedelta(days=100)),
              policy["calendar_verified_through"])
    schedules = {}
    result = {}
    for ticker, kind in sorted(instrument_kinds.items()):
        market = KIND_CALENDARS.get(kind)
        if market is None:
            raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
        if market not in schedules:
            schedules[market] = calendar(published.date() - timedelta(days=7), end,
                                         policy, market)
        schedule = schedules[market]
        active = schedule[(schedule.market_open <= published) &
                          (schedule.market_close > published)]
        if not active.empty:
            requested = published
            basis = "PUBLICATION_DURING_MARKET"
        else:
            future = schedule[schedule.market_open > published]
            if future.empty:
                raise ValueError("INCOMPLETE_SESSION_CALENDAR")
            requested = future.iloc[0].market_open + pd.Timedelta(
                minutes=policy["entry_wait_after_open_minutes"])
            basis = "NEXT_OPEN_PLUS_CONFIGURED_WAIT"
        result[ticker] = {"ticker": ticker, "kind": kind,
                          "market": market, "requested_entry_at": requested.isoformat(),
                          "session_open_at": active.iloc[0].market_open.isoformat() if not active.empty
                          else future.iloc[0].market_open.isoformat(),
                          "session_close_at": active.iloc[0].market_close.isoformat() if not active.empty
                          else future.iloc[0].market_close.isoformat(),
                          "entry_date": str(requested.date()), "basis": basis,
                          "ready": bool(requested <= now)}
    return result


def _first_traded_intraday_bar(frame, requested, deadline):
    if frame.empty:
        raise ValueError("ENTRY_INTRADAY_HISTORY_UNAVAILABLE")
    frame = frame.copy()
    if not {"Open", "Volume"}.issubset(frame.columns):
        raise ValueError("ENTRY_INTRADAY_HISTORY_UNAVAILABLE")
    frame.index = pd.to_datetime(frame.index, utc=True)
    requested = pd.Timestamp(requested)
    deadline = pd.Timestamp(deadline)
    price = pd.to_numeric(frame.get("Open"), errors="coerce")
    volume = pd.to_numeric(frame.get("Volume"), errors="coerce")
    valid = frame[(frame.index >= requested) & (frame.index <= deadline) &
                  price.map(lambda value: math.isfinite(float(value)) and float(value) > 0) &
                  volume.map(lambda value: math.isfinite(float(value)) and float(value) > 0)]
    if valid.empty:
        raise ValueError("ENTRY_INTRADAY_HISTORY_UNAVAILABLE")
    quote_at = valid.index[0]
    return float(valid.iloc[0]["Open"]), quote_at


def _intraday_frame(ticker, requested):
    start = str(requested.date())
    end = str((requested + pd.Timedelta(days=1)).date())
    try:
        instrument = yf.Ticker(ticker)
        frame = instrument.history(start=start, end=end, interval="1m",
                                   auto_adjust=False, actions=False,
                                   repair=False, timeout=15)
        if not frame.empty:
            return frame
        # Yahoo occasionally returns an empty start/end response for a valid
        # recent intraday date. Retry through its rolling intraday endpoint,
        # then retain only the requested UTC date.
        frame = instrument.history(period="5d", interval="1m", auto_adjust=False,
                                   actions=False, repair=False, timeout=15)
        if frame.empty:
            return frame
        utc_index = pd.to_datetime(frame.index, utc=True)
        wanted = pd.Timestamp(requested).date()
        return frame[utc_index.date == wanted]
    except Exception:
        raise ValueError("ENTRY_INTRADAY_HISTORY_UNAVAILABLE") from None


def _fx_rate_at_or_before(frame, quote_at, max_age_minutes):
    """Use only FX information observable when the security trade occurred."""
    if frame.empty or "Open" not in frame.columns:
        raise ValueError("FX_ENTRY_QUOTE_UNAVAILABLE")
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    quote_at = pd.Timestamp(quote_at)
    prices = pd.to_numeric(frame["Open"], errors="coerce")
    valid = frame[(frame.index <= quote_at) & prices.map(
        lambda value: math.isfinite(float(value)) and float(value) > 0)]
    if valid.empty:
        raise ValueError("FX_ENTRY_QUOTE_UNAVAILABLE")
    fx_at = valid.index[-1]
    if quote_at - fx_at > pd.Timedelta(minutes=max_age_minutes):
        raise ValueError("FX_ENTRY_QUOTE_UNAVAILABLE")
    return float(valid.iloc[-1]["Open"]), fx_at


def _next_session_entry(entry, policy):
    close = pd.Timestamp(entry["session_close_at"])
    schedule = calendar(close.date(), close.date() + timedelta(days=14), policy,
                        entry["market"])
    future = schedule[schedule.market_open > close]
    if future.empty:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    row = future.iloc[0]
    requested = row.market_open + pd.Timedelta(minutes=policy["entry_wait_after_open_minutes"])
    return {**entry, "requested_entry_at": requested.isoformat(),
            "session_open_at": row.market_open.isoformat(),
            "session_close_at": row.market_close.isoformat(),
            "entry_date": str(requested.date()),
            "basis": "DEFERRED_NEXT_OPEN_PLUS_CONFIGURED_WAIT",
            "ready": False}


def fetch_entry_quote(ticker, entry, policy, now=None):
    """Capture the first actual trade after eligibility, never an earlier bar.

    If no qualifying trade occurs before the configured deadline/session close,
    advance to the same rule on that security's next exchange session.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    candidate = dict(entry)
    for _ in range(15):
        requested = pd.Timestamp(candidate["requested_entry_at"])
        if requested > now:
            pending = AwaitingMarketEntry(candidate["entry_date"], requested.isoformat())
            pending.planned_entry = candidate
            raise pending
        session_close = pd.Timestamp(candidate["session_close_at"])
        deadline = min(session_close, requested + pd.Timedelta(
            minutes=policy["entry_max_quote_delay_minutes"]))
        try:
            bars = _intraday_frame(ticker, requested)
        except ValueError as exc:
            if str(exc) == "ENTRY_INTRADAY_HISTORY_UNAVAILABLE":
                raise _data_retry(candidate, now, ticker)
            raise
        try:
            price, quote_at = _first_traded_intraday_bar(bars, requested, deadline)
            break
        except ValueError as exc:
            if str(exc) != "ENTRY_INTRADAY_HISTORY_UNAVAILABLE":
                raise
            # Do not declare the window empty while it can still receive a trade.
            if now <= deadline:
                pending = AwaitingMarketEntry(candidate["entry_date"], deadline.isoformat())
                pending.planned_entry = candidate
                raise pending
            # Empty and sparse frames both mean that no verifiable trade was
            # observed in the completed eligible window. Never invent a price:
            # apply the entry rule again on this security's next session.
            candidate = _next_session_entry(candidate, policy)
    else:
        raise ValueError("ENTRY_INTRADAY_HISTORY_UNAVAILABLE")
    fx_rate = 1.0
    fx_at = None
    if not ticker.endswith(".NS"):
        try:
            fx = _intraday_frame("INR=X", quote_at)
        except ValueError as exc:
            if str(exc) == "ENTRY_INTRADAY_HISTORY_UNAVAILABLE":
                raise _data_retry(candidate, now, ticker, "FX_DATA_RETRY")
            raise
        # Yahoo FX bars can be sparse. Freeze the latest quote observable at
        # trade time, bounded by owner policy; never use future information.
        try:
            fx_rate, fx_at = _fx_rate_at_or_before(
                fx, quote_at, policy["fx_quote_max_age_minutes"])
        except ValueError as exc:
            if str(exc) == "FX_ENTRY_QUOTE_UNAVAILABLE":
                raise _data_retry(candidate, now, ticker, "FX_DATA_RETRY")
            raise
    return {**candidate, "ready": True, "price_inr": price * fx_rate,
            "native_price": price, "fx_to_inr": fx_rate,
            "fx_quote_at": pd.Timestamp(fx_at).isoformat() if fx_at is not None else None,
            "quote_at": pd.Timestamp(quote_at).isoformat(),
            "source": "Yahoo Finance one-minute security Open and latest policy-fresh FX Open observable at trade time"}


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
