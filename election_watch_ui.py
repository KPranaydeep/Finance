"""Small monthly election and measured-volatility watch for the personal app."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import yfinance as yf
from market_regime_watch import detect_market_changes
from election_watch import monthly_snapshot, upcoming_terms, representation_context, market_context, SOURCE, PARLIAMENT_SOURCE, REPRESENTATION_VERIFIED_AT, _atomic_save


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
        if upcoming:
            nearest = upcoming[0]
            st.markdown(f"**Election timing:** nearest tracked term end is **{nearest['State']} in {nearest['Days to term end']} days**. Polling/counting dates remain unverified.")
            share = f"{context['assembly_share']:.1%}" if context['assembly_share'] is not None else "unknown"
            st.caption(f"18-month watch: {len(upcoming)} states · {share} of assembly seats geographically covered; not confirmed seats at stake or a market-impact probability.")
        with st.expander("Election dates, seat exposure & sources", expanded=False):
            st.caption(f"Official terms verified: {snapshot.get('verified_at', 'Unknown')} · checked once per calendar month while the app is used. {snapshot.get('coverage', '')}")
            if snapshot.get("refresh_error"):
                st.caption(f"ECI refresh unavailable; previous verified snapshot retained. {snapshot['refresh_error']}")
            context = representation_context(snapshot, today)
            upcoming = context["rows"]
            if upcoming:
                nearest = upcoming[0]
                st.markdown(f"**Calendar watch · {len(upcoming)} states over 18 months** · nearest term end: **{nearest['State']}**, {nearest['Days to term end']} calendar days away.")
                left, right = st.columns(2)
                left.metric("Direct Lok Sabha seat changes from these assembly polls", "0.0%", border=True)
                right.metric("Confirmed assembly seats at stake", "Awaiting schedule", border=True)
                share = f"{context['assembly_share']:.1%}" if context['assembly_share'] is not None else "Unknown"
                st.caption(f"Geographic footprint, not seats confirmed at stake: assembly {share} ({context['assembly_count']:,}/{context['assembly_total'] or 'unknown'}); Lok Sabha {context['lok_share']:.1%} ({context['lok_count']}/543 elected seats); Rajya Sabha {context['rajya_share']:.1%} ({context['rajya_count']}/233 elected seats).")
                st.caption("Rajya Sabha seats actually up in the relevant cycle: unknown. Coalition-majority impact: unquantified. These are not zero-impact assumptions or market-impact probabilities.")
                st.caption(f"National denominators verified {REPRESENTATION_VERIFIED_AT}; seat allocations, not currently occupied seats.")
                with st.expander("States, denominators and timing", expanded=False):
                    for row in upcoming:
                        assembly_share = f"{row['Share of all assembly seats (%)']:.1f}%" if row['Share of all assembly seats (%)'] is not None else "unknown"
                        st.markdown(f"**{row['State']}** · term ends **{row['Term ends']}** ({row['Days to term end']} days) · {row['Assembly seats']} MLAs, {assembly_share} of all assembly seats · full-house majority {row['Full-house majority']}.")
                        st.caption(f"National geographic footprint: LS {row['Lok Sabha allocation']}/543 ({row['Lok Sabha footprint (%)']:.1f}%) · RS {row['Rajya Sabha allocation']}/233 ({row['Rajya Sabha footprint (%)']:.1f}%). Verified contested percentage: unknown. An announced full-state general election would ordinarily contest 100% of that assembly; that assumption is not treated as a verified schedule.")
                st.caption("Term end is not polling/counting day. A distant term end is calendar context; a confirmed event date is needed to rank near-term timing. Refresh portfolio evidence if market stress changes; seat percentages alone do not set reserve or borrowing allocations.")
            else:
                st.caption("No upcoming terms in the retained watchlist window. This does not establish that no elections are scheduled.")
            st.markdown(f"[ECI terms & seat allocations]({SOURCE}) · [ECI announcements](https://www.eci.gov.in/) · [How Rajya Sabha representation works]({PARLIAMENT_SOURCE})")
            st.caption("MLAs determine state governments and elect Rajya Sabha members as vacancies arise. Lok Sabha confidence determines the Union government's survival; state results alone do not change that majority. Assembly majorities here assume all seats filled; actual confidence votes can differ with vacancies and absences.")

        if market.get("refresh_errors"):
            st.caption("Market refresh incomplete; retained values may be stale: " + str(market["refresh_errors"]))
        st.caption("Calendar checks monthly; market/regime data checks once daily when the app is used, plus manual refresh. A detected change is a reason to reassess portfolio evidence, not proof of election causation or a bottom. This is an on-screen review flag; no background notifications run while the app is closed.")
