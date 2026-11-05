"""US election scenarios over the entire saved mix, valued in INR."""
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from market_history_cache import load_cached_market_history

US_EVENTS = [dict(name=f"US federal election {year}", event_date=day, country="US", verified=True,
                  source=("https://www.fec.gov/documents/5910/2026pdates.pdf" if year == 2026 else
                          f"https://www.fec.gov/resources/cms-content/documents/{year}pdates.pdf"))
             for year, day in [(2020,"2020-11-03"),(2022,"2022-11-08"),(2024,"2024-11-05"),(2026,"2026-11-03")]]
US_CALENDAR_SOURCE = "https://www.congress.gov/crs_external_products/R/PDF/R46413/R46413.8.pdf"
US_WATCH_DAYS = 20


def us_federal_events(today):
    """Regular federal dates under the statutory rule, not results or specials."""
    through = pd.Timestamp(today).year + 4
    known = {int(e["event_date"][:4]): e for e in US_EVENTS}
    events = []
    for year in range(2020, through + 1, 2):
        november = pd.Timestamp(year=year, month=11, day=1)
        first_monday = november + pd.Timedelta(days=(0 - november.weekday()) % 7)
        event = dict(known.get(year) or {}, event_date=str((first_monday + pd.Timedelta(days=1)).date()),
                     name=f"US {'presidential & congressional election' if year % 4 == 0 else 'midterm election'} {year}",
                     country="US", verified=True, date_basis="statutory regular federal schedule",
                     source=(known.get(year) or {}).get("source", US_CALENDAR_SOURCE))
        events.append(event)
    return events


def us_event_context(today):
    day = pd.Timestamp(today).normalize().tz_localize(None)
    events = us_federal_events(day)
    active = []
    for event in events:
        anchor = pd.Timestamp(event["event_date"])
        if abs((anchor - day).days) <= US_WATCH_DAYS - 1:
            phase = "upcoming" if day < anchor else "election_day" if day == anchor else "post_election"
            active.append(dict(event, phase=phase,
                               watch_ends=str((anchor+pd.Timedelta(days=US_WATCH_DAYS-1)).date()),
                               days_since_election=max(0, (day-anchor).days)))
    return dict(active_events=active,
                latest_event=next((e for e in reversed(events) if pd.Timestamp(e['event_date']) < day), None),
                next_event=next((e for e in events if pd.Timestamp(e['event_date']) >= day), None),
                scope="Regular federal elections; special/runoff schedules and result certification are not assessed")


def latest_joint_session(today):
    day=pd.Timestamp(today).normalize()
    sessions=mcal.get_calendar('NSE').valid_days(day-pd.Timedelta(days=15),day-pd.Timedelta(days=1)).union(
        mcal.get_calendar('NYSE').valid_days(day-pd.Timedelta(days=15),day-pd.Timedelta(days=1)))
    return str(sessions[-1].date())


def next_us_event_review(today):
    day=pd.Timestamp(today).normalize()
    event=next((e for e in us_event_context(day)['active_events'] if e['phase'] in {'upcoming','election_day'}),None)
    if not event:return None
    calendar=mcal.get_calendar('NYSE')
    sessions=calendar.valid_days(day,pd.Timestamp(event['event_date'])).tz_localize(None)
    if not len(sessions):return None
    return str(sessions[0].date())


def allocation_inputs(payload):
    prior = payload.get("election_reserve_history") or {}
    weights = {str(r["ticker"]):float(r.get("target_weight") or 0) for r in payload.get("optimal_allocation") or []} or prior.get("weights") or {}
    weights = {t:float(w) for t,w in weights.items() if float(w)>0}
    metadata = {str(r["Yahoo Ticker"]):r for r in payload.get("current_allocation") or []}
    return weights, metadata


def listing_calendar(row):
    exchange = str(row.get("Exchange", "")).upper()
    if exchange in {"NSI","NSE","BSE","BSEI","BOM"}: return "NSE"
    if exchange in {"NYQ","NMS","NGM","NCM","ASE","PCX","BTS","NYSE","NASDAQ","NYSEARCA"}: return "NYSE"
    if exchange == "CCC": return "24/7"
    raise ValueError("Listing calendar unavailable: " + exchange)


def portfolio_prices_from_cache(payload, path, today):
    """No downloads. Carry prices only on verified closed-market days; never backfill."""
    weights, metadata = allocation_inputs(payload)
    if not weights or abs(sum(weights.values())-1)>0.011:
        raise ValueError("Complete normalized optimal weights required")
    tickers = list(dict.fromkeys([*weights, *[str(r['Yahoo Ticker']) for r in payload.get('current_allocation') or [] if float(r.get('Quantity') or 0)>0]]))
    currencies = {t:str(metadata.get(t,{}).get("Currency", "")).upper() for t in tickers}
    if any(c not in {"INR","USD"} for c in currencies.values()):
        raise ValueError("Unsupported or missing currency; full-portfolio coverage withheld")
    end = pd.Timestamp(today).normalize()
    prices, _ = load_cached_market_history(path, tickers + ["USDINR=X","INR=X"], "2020-01-01", str(today))
    days = pd.date_range("2020-01-01", end-pd.Timedelta(days=1))
    converted, latest = {}, {}
    if "USD" in currencies.values():
        candidates = [prices[t].dropna() for t in ["USDINR=X","INR=X"] if t in prices and len(prices[t].dropna())>30]
        if not candidates: raise ValueError("Dated USD/INR history unavailable")
        candidates.sort(key=lambda s:s.index[-1],reverse=True)
        fx=candidates[0]
        for older in candidates[1:]: fx=fx.combine_first(older)
        fx = fx.reindex(days).ffill(limit=3)  # weekends only; no backward fill or unlimited stale FX
    union = mcal.get_calendar("NSE").valid_days(days[0], days[-1]).tz_localize(None).union(
        mcal.get_calendar("NYSE").valid_days(days[0], days[-1]).tz_localize(None))
    for t in tickers:
        if t not in prices or len(prices[t].dropna())<31: raise ValueError("Insufficient cached history: " + t)
        market = listing_calendar(metadata[t])
        native = prices[t].where(prices[t]>0).reindex(days)
        opened = days if market=="24/7" else mcal.get_calendar(market).valid_days(days[0],days[-1]).tz_localize(None)
        # A missing open-day quote remains missing. Closed days may carry the
        # previous native close, but never across an unresolved open-day gap.
        marked = native.copy()
        carried = native.ffill()
        missing_open = native.isna() & native.index.isin(opened)
        gap = pd.Series(False,index=days,dtype='boolean')
        gap.loc[opened] = missing_open.loc[opened]
        gap = gap.where(gap.index.isin(opened)).ffill().fillna(True).astype(bool)
        closed = ~native.index.isin(opened)
        marked.loc[closed & ~gap] = carried.loc[closed & ~gap]
        converted[t] = marked if currencies[t]=="INR" else marked*fx
        latest[t] = str(native.dropna().index[-1].date())
    frame = pd.DataFrame(converted).reindex(union)
    return dict(schema_version=2, basis="INR", weights=weights, metadata=metadata, columns=tickers,
                dates=[str(d.date()) for d in frame.index], values=frame.astype(object).where(frame.notna(),None).values.tolist(),
                native_dates=latest, expected_through=str(union[-1].date()), source="persistent native prices and dated FX")


def unpack_prices(history):
    if history.get("schema_version")!=2 or history.get("basis")!="INR":
        raise ValueError("Full-portfolio INR price evidence required")
    return pd.DataFrame(history["values"],columns=history["columns"],index=pd.to_datetime(history["dates"])).astype(float)


def us_election_reference(history, today, india_events=(), *, current_weights=None):
    today = pd.Timestamp(today).normalize()
    cal=mcal.get_calendar("NYSE")
    lower,upper=today-pd.Timedelta(days=19),today+pd.Timedelta(days=19)
    context=us_event_context(today)
    active=context['active_events']
    result=dict(status="not_assessed",coverage_complete=False,active_events=active,windows=[],event_reserve_fraction=None,
                calendar=context, reason="US election risk not assessed: whole-portfolio price/FX evidence has not been prepared",historical_stress_fraction=None)
    if not active:
        return dict(result,status="no_active_us_event",coverage_complete=True,
                    reason="No regular US federal election in the 20-calendar-day preparation/aftermath window; specials/runoffs unassessed")
    if not history: return result
    frame=unpack_prices(history)
    weights=history["weights"]
    if current_weights is not None and (set(weights) != set(current_weights) or any(
            not np.isclose(float(weights[t]),float(current_weights[t]),rtol=0,atol=1e-9) for t in weights)):
        return dict(result,status="stale_or_incomplete_prices",reason="US event evidence uses older optimal weights; refresh portfolio evidence")
    result['data_through']=str(frame.loc[:,list(weights)].dropna().index.max().date()) if not frame.loc[:,list(weights)].dropna().empty else None
    if result['data_through']!=history['expected_through'] or history['expected_through']!=latest_joint_session(today):
        return dict(result,status="stale_or_incomplete_prices",reason="US election risk not assessed: current common prices/FX incomplete")
    # Replay the complete INR-valued mix around US elections. There is no
    # inference that quote currency equals issuer domicile or policy sensitivity.
    consumed=pd.Timestamp.min
    for event in us_federal_events(today):
        anchor=pd.Timestamp(event['event_date'])
        if anchor>=today: continue
        sessions=cal.valid_days(anchor-pd.Timedelta(days=90),anchor+pd.Timedelta(days=90)).tz_localize(None)
        before,after=sessions[sessions<anchor][-16:],sessions[sessions>=anchor][:15]
        if len(before)!=16 or len(after)!=15:continue
        start,end=before[0],after[-1]
        if end>=today or start<=consumed:continue
        window=frame.loc[start:end,list(weights)]
        if window.empty or window.index[0]!=start or window.index[-1]!=end or window.isna().any().any():continue
        wealth=(window/window.iloc[0]).to_numpy() @ np.array(list(weights.values()))
        loss=float(np.max(1-wealth/np.maximum.accumulate(wealth)))
        result['windows'].append(dict(event_date=event['event_date'],start=str(start.date()),end=str(end.date()),loss=loss))
        consumed=end
    result['historical_stress_fraction']=max((w['loss'] for w in result['windows']),default=None)
    if not result['windows']:
        return dict(result,status="insufficient_event_history",reason="US election risk not assessed: no complete historical whole-portfolio window")
    # Known simultaneous Indian events cannot be treated as independent and
    # added. With no validated joint analog, label the combined amount partial.
    overlapping=[e for e in india_events if e.get('verified') and e.get('counting_date') and lower<=pd.Timestamp(e['counting_date'])<=upper]
    result.update(status="historical_scenario",coverage_complete=not overlapping,event_reserve_fraction=result['historical_stress_fraction'],
                  reason="Limited historical whole-portfolio stress reference; not an optimal reserve or loss limit")
    if overlapping: result['reason']+="; simultaneous India/US joint stress not established—amounts are not summed"
    return result
