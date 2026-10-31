"""Small monthly election and measured-volatility watch for the personal app."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import yfinance as yf
from market_regime_watch import detect_market_changes
from us_election_risk import US_EVENTS
from election_watch import monthly_snapshot, representation_context, market_context, SOURCE, PARLIAMENT_SOURCE, REPRESENTATION_VERIFIED_AT, _atomic_save


def render_state_timing(context, snapshot, scope):
    """Compact cards: timing and representation, without inferred impact scores."""
    rows = context["rows"]
    if not rows:
        st.caption("No term ends in the next 18 months in this snapshot.")
        return
    share = f"{context['assembly_share']:.1%}" if context['assembly_share'] is not None else "Unknown"
    with st.container(horizontal=True):
        st.metric("Assembly footprint", share)
        st.metric("Lok Sabha footprint", f"{context['lok_share']:.1%}")
        st.metric("Rajya Sabha footprint", f"{context['rajya_share']:.1%}")
    nearest = min(rows, key=lambda row: row["Days to term end"])
    largest = max(rows, key=lambda row: row["Assembly seats"])
    st.markdown(
        f"**Soonest:** {nearest['State']} · {nearest['Days to term end']} days"
        f" · **Largest assembly:** {largest['State']} · {largest['Assembly seats']} seats"
    )
    st.caption("Term ends, not polling dates. Percentages show national geographic representation—not confirmed contests or predicted market impact.")
    order = st.radio("Order states by", ["Soonest", "Largest assembly"], horizontal=True,
                     key=f"state_timing_order_{scope}")
    ordered = sorted(rows, key=(lambda row: (-row["Assembly seats"], row["Term ends"]))
                     if order == "Largest assembly" else (lambda row: row["Term ends"]))
    for offset in range(0, len(ordered), 3):
        cards = st.columns(3, gap="small")
        for card, row in zip(cards, ordered[offset:offset + 3]):
            with card.container(border=True):
                st.markdown(f"**{row['State']}**")
                date_label = pd.Timestamp(row["Term ends"]).strftime("%d %b %Y")
                st.markdown(f"{date_label} · **{row['Days to term end']} days**")
                assembly_share = row["Share of all assembly seats (%)"]
                assembly_label = f"{assembly_share:.1f}%" if assembly_share is not None else "Unknown"
                st.caption(f"{row['Assembly seats']} MLAs · majority {row['Full-house majority']}")
                st.markdown(f"Assembly **{assembly_label}** · LS **{row['Lok Sabha footprint (%)']:.1f}%** · RS **{row['Rajya Sabha footprint (%)']:.1f}%**")
    with st.expander("Definitions, data quality & sources", expanded=False):
        st.caption(f"Denominators: assembly {context['assembly_total'] or 'incomplete snapshot'} · LS 543 · RS 233 elected seats. Watch covers {len(rows)} states over 18 months; allocations verified {REPRESENTATION_VERIFIED_AT}.")
        st.caption("Majority = full-house majority; vacancies/absences may change confidence-vote arithmetic. MLAs form state governments and elect RS members as vacancies arise. Assembly polls directly change no LS seats; RS seats due and coalition effects are unquantified.")
        st.caption(f"Terms verified: {snapshot.get('verified_at', 'Unknown')} · refresh monthly while app is used. Polling/counting schedule unverified.")
        if snapshot.get("refresh_error"):
            st.caption(f"ECI refresh failed; retained verified snapshot. {snapshot['refresh_error']}")
        st.markdown(f"[ECI terms & seats]({SOURCE}) · [ECI announcements](https://www.eci.gov.in/) · [Rajya Sabha representation]({PARLIAMENT_SOURCE})")


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def _market_snapshot(end_date):
    result = {}
    for ticker, name in (("^NSEI", "NIFTY 50"), ("^INDIAVIX", "India VIX")):
        try:
            # Fixed start keeps the detector's initial scale stable across daily
            # refreshes rather than moving the warmup with a rolling 2y request.
            prices = yf.Ticker(ticker).history(start="2020-01-01", end=end_date, auto_adjust=True, timeout=12)
            if prices.empty:
                raise ValueError("no daily prices returned")
            prices = prices.loc[[str(day)[:10] < end_date for day in prices.index]]
            if name == "NIFTY 50":
                result[name] = market_context(prices["Close"])
                result["Regime change"] = detect_market_changes(prices["Close"])
            else:
                series = prices["Close"].dropna()
                result[name] = {"level": float(series.iloc[-1]), "as_of": str(series.index[-1])[:10]}
        except Exception as exc:
            result[name] = {"error": str(exc)[:150]}
    return result


def render_election_watch(scope):
    with st.container(border=True):
        st.markdown("**Market status · election context**")
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        root = Path.home() / ".portfolio_rebalancer"
        refresh = st.button("Refresh election & market context", key=f"election_watch_{scope}")
        try:
            snapshot = monthly_snapshot(root / "election_watch.json", today, force=refresh)
        except OSError as exc:
            st.warning(f"Election snapshot could not be persisted: {exc}")
            return
        market_path = root / "election_market_context.json"
        try:
            market = json.loads(market_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            market = {}
        if refresh or market.get("attempted_date") != str(today):
            if refresh:
                _market_snapshot.clear()
            fetched = _market_snapshot(str(today))
            # Preserve successful data through provider failures, with failure dates.
            market["attempted_month"] = today.strftime("%Y-%m")
            market["attempted_date"] = str(today)
            errors = {}
            for name, values in fetched.items():
                if values.get("error"):
                    errors[name] = values["error"]
                else:
                    market[name] = values
            market["refresh_errors"] = errors
            try:
                _atomic_save(market_path, market)
            except OSError as exc:
                st.caption(f"Market context could not be saved: {exc}")
        index = market.get("NIFTY 50") or {}
        vix = market.get("India VIX") or {}
        if index:
            st.markdown(f"**NIFTY is {abs(index['drawdown']):.1%} below its one-year high** · 30-session volatility is at the **{index['volatility_percentile']:.0f}th percentile** of its observed history.")
            drawdown_col, vol_col, vix_col = st.columns(3)
            drawdown_col.metric("From 252-session high", f"{index['drawdown']:.1%}")
            vol_col.metric("Realised volatility · annualised", f"{index['volatility']:.1%}")
            vix_col.metric("India VIX", f"{vix['level']:.2f}" if vix else "Unavailable")
            st.caption(f"NIFTY data: {index['as_of']} · VIX data: {vix.get('as_of', 'unavailable')}")
        else:
            st.caption("Market status unavailable: refresh market context to retrieve daily prices.")
        regime = market.get("Regime change") or {}
        if regime.get("status") in {"recent_change", "no_recent_change"}:
            if regime["status"] == "recent_change":
                latest = regime["recent_events"][-1]
                st.markdown(f"**Regime review flag:** {latest['Change']} · detected {latest['Detected on']} · data {regime['as_of']}.")
            else:
                st.caption(f"Regime monitor: no ADWIN change detected in the last {regime['recent_sessions']} observed sessions · data {regime['as_of']}.")
            with st.expander("Regime-change evidence", expanded=False):
                st.caption(regime["library"] + ". " + regime["method"])
                for event in regime["events"][-5:]:
                    st.caption(f"{event['Detected on']} · {event['Change']}")
                st.caption("A change flag requests review; it does not identify the cause, predict a bottom, or automatically change reserve/debt allocations. Election events remain separate context.")
        elif regime:
            st.caption(f"Regime monitor unavailable: {regime.get('reason', 'insufficient data')}")
        else:
            st.caption("Refresh election & market context to populate the River regime monitor.")
        st.caption("Decision context: the decline measures the pullback; the volatility percentile measures unusual movement, not upside potential. The portfolio's cost-tested rebalance decision and reserve model determine the response.")
        context = representation_context(snapshot, today)
        upcoming = context["rows"]
        next_us = next((event for event in US_EVENTS if event['event_date'] >= str(today)), None)
        if next_us:
            days=(pd.Timestamp(next_us['event_date']).date()-today).days
            st.markdown(f"**US timing:** [federal election {next_us['event_date']}]({next_us['source']}) · **{days} calendar days away**. Whole-portfolio assessment is in Election reserve above.")
        if upcoming:
            nearest = upcoming[0]
            st.markdown(f"**India timing:** nearest tracked term end is **{nearest['State']} in {nearest['Days to term end']} days**. Polling/counting dates remain unverified.")
            share = f"{context['assembly_share']:.1%}" if context['assembly_share'] is not None else "unknown"
            st.caption(f"18-month watch: {len(upcoming)} states · {share} of assembly seats geographically covered; not confirmed seats at stake or a market-impact probability.")
        with st.expander("States · timing & seat footprint", expanded=False):
            render_state_timing(context, snapshot, scope)

        if market.get("refresh_errors"):
            st.caption("Market refresh incomplete; retained values may be stale: " + str(market["refresh_errors"]))
        st.caption("Calendar checks monthly; market/regime data checks once daily when the app is used, plus manual refresh. A detected change is a reason to reassess portfolio evidence, not proof of election causation or a bottom. This is an on-screen review flag; no background notifications run while the app is closed.")
