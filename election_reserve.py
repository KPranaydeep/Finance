"""INR-only historical event stress; never infer losses from political seat shares."""
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def history_from_market_cache(payload, path, today):
    """Migrate older saved analyses using their existing local INR prices."""
    from market_history_cache import load_cached_market_history
    weights = (payload.get("election_reserve_history") or {}).get("weights") or {
        str(r["ticker"]): float(r.get("target_weight") or 0) for r in payload.get("optimal_allocation") or []
    }
    currencies = {str(r["Yahoo Ticker"]): str(r["Currency"]).upper() for r in payload.get("current_allocation") or [] if r.get("Currency")}
    if not weights or any(w > 0 and t not in currencies for t, w in weights.items()):
        raise ValueError("Saved allocation lacks complete listing currency metadata")
    tickers = [t for t, w in weights.items() if w > 0 and currencies.get(t) == "INR"]
    if not tickers:
        return pack_inr_history(pd.DataFrame(), weights, currencies)
    prices, _ = load_cached_market_history(path, tickers, "2020-01-01", str(today))
    if any(t not in prices or prices[t].dropna().size < 31 for t in tickers):
        raise ValueError("Local price cache lacks sufficient INR history")
    prices = prices.loc[:, tickers].where(prices.loc[:, tickers] > 0)
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return dict(pack_inr_history(returns, weights, currencies), source="existing persistent price cache")


def pack_inr_history(log_returns, weights, currencies):
    tickers = [t for t, w in weights.items() if w > 0 and currencies.get(t) == "INR"]
    frame = pd.DataFrame(log_returns).reindex(columns=tickers).replace([np.inf, -np.inf], np.nan)
    return dict(weights=weights, currencies=currencies, columns=tickers,
                dates=[str(d) for d in frame.index], values=frame.astype(object).where(frame.notna(), None).values.tolist())


def saved_event_reference(payload, events, today):
    history = payload.get("election_reserve_history") or {}
    frame = pd.DataFrame(history.get("values", []), columns=history.get("columns", []), index=history.get("dates", []))
    if not history:
        weights = {str(r["ticker"]): float(r.get("target_weight") or 0) for r in payload.get("optimal_allocation") or []}
        currencies = {str(r["Yahoo Ticker"]): str(r["Currency"]).upper() for r in payload.get("current_allocation") or [] if r.get("Currency")}
        if not weights:
            return dict(status="missing_inr_history", event_reserve_fraction=None, active_events=[], windows=[])
        return election_reserve_reference(frame, weights, currencies, events, today)
    return election_reserve_reference(frame, history["weights"], history["currencies"], events, today)


def election_reserve_reference(log_returns, weights, currencies, events, today):
    today = pd.Timestamp(today).normalize().tz_localize(None)
    positive = {str(t): float(w) for t, w in weights.items() if np.isfinite(w) and w > 0}
    unknown = [t for t in positive if not currencies.get(t)]
    inr = {t: w for t, w in positive.items() if currencies.get(t) == "INR"}
    share = sum(inr.values())
    result = dict(schema_version=1, inr_weight=share, event_reserve_fraction=None,
                  historical_stress_fraction=None, windows=[], active_events=[], status="no_verified_near_event")
    if unknown or sum(positive.values()) > 1.001:
        return dict(result, status="incomplete_currency_or_weights")
    if not inr:
        return dict(result, status="no_inr_exposure", event_reserve_fraction=0.0)
    calendar = mcal.get_calendar("NSE")
    sessions = calendar.valid_days(start_date=today-pd.Timedelta(days=120), end_date=today+pd.Timedelta(days=120)).tz_localize(None)
    before = sessions[sessions < today]
    after = sessions[sessions >= today]
    lower, upper = before[-30], after[29]
    for event in events:
        if not event.get("verified") or not event.get("source"):
            continue
        counting = pd.to_datetime(event.get("counting_date"), errors="coerce")
        if pd.isna(counting):
            continue
        polls = [pd.Timestamp(d) for d in event.get("poll_dates", [])]
        # Watch the upcoming 30-session horizon, ongoing polls, and the
        # 30-session aftermath. A distant term end does not activate a reserve.
        if lower <= counting <= upper or (polls and min(polls) <= today <= counting):
            preparation = calendar.valid_days(start_date=counting-pd.Timedelta(days=90), end_date=counting-pd.Timedelta(days=1)).tz_localize(None)
            result["active_events"].append(dict(event, preparation_date=str(preparation[-30].date())))
    frame = pd.DataFrame(log_returns).copy()
    if frame.empty or any(t not in frame for t in inr):
        return dict(result, status="missing_inr_history")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[frame.index < today, list(inr)].apply(pd.to_numeric, errors="coerce")
    result["data_through"] = str(frame.index.max().date()) if len(frame) else None
    consumed_end = pd.Timestamp.min
    # Same-day elections and overlapping windows are one stress episode.
    count_dates = sorted({pd.Timestamp(e["counting_date"]) for e in events
                          if e.get("verified") and e.get("source") and e.get("counting_date")})
    for counting in count_dates:
        if counting >= today:
            continue
        expected = calendar.valid_days(start_date=counting-pd.Timedelta(days=60), end_date=counting+pd.Timedelta(days=60)).tz_localize(None)
        left, right = expected[expected < counting][-15:], expected[expected >= counting][:15]
        dates = left.append(right)
        if len(dates) != 30 or dates[-1] >= today or dates[0] <= consumed_end:
            continue
        window = frame.reindex(dates)
        if window.isna().any().any() or not np.isfinite(window.to_numpy()).all():
            continue
        # Buy-and-hold current INR mix, preserving cross-security dependence.
        wealth = np.r_[1.0, np.exp(window.cumsum().to_numpy()) @ np.array(list(inr.values())) / share]
        loss = float(np.max(1 - wealth / np.maximum.accumulate(wealth)))
        result["windows"].append(dict(counting_date=str(counting.date()), loss=loss,
                                      start=str(dates[0].date()), end=str(dates[-1].date())))
        consumed_end = dates[-1]
    if result["windows"]:
        result["historical_stress_fraction"] = share * max(w["loss"] for w in result["windows"])
    if not result["active_events"]:
        return result
    if not result["windows"]:
        return dict(result, status="insufficient_event_history")
    return dict(result, status="active_event_stress", event_reserve_fraction=result["historical_stress_fraction"])


def combined_reserve(base_fraction, reference):
    """Refresh applicability against today's schedule before calling."""
    event = reference.get("event_reserve_fraction")
    return max(float(base_fraction), float(event) if event is not None else 0.0)
