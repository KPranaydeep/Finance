"""Small monthly election and measured-volatility watch for the personal app."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import yfinance as yf
from election_watch import monthly_snapshot, upcoming_terms, market_context, SOURCE, PARLIAMENT_SOURCE, _atomic_save


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def _market_snapshot(end_date):
    result = {}
    for ticker, name in (("^NSEI", "NIFTY 50"), ("^INDIAVIX", "India VIX")):
        try:
            prices = yf.Ticker(ticker).history(period="2y", auto_adjust=True, timeout=12)
            if prices.empty:
                raise ValueError("no daily prices returned")
            prices = prices.loc[[str(day)[:10] < end_date for day in prices.index]]
            if name == "NIFTY 50":
                result[name] = market_context(prices["Close"])
            else:
                series = prices["Close"].dropna()
                result[name] = {"level": float(series.iloc[-1]), "as_of": str(series.index[-1])[:10]}
        except Exception as exc:
            result[name] = {"error": str(exc)[:150]}
    return result


def render_election_watch(scope):
    with st.expander("Election & volatility watch · monthly", expanded=False):
        today = pd.Timestamp.now(tz="Asia/Kolkata").date()
        root = Path.home() / ".portfolio_rebalancer"
        refresh = st.button("Refresh election & market context", key=f"election_watch_{scope}")
        try:
            snapshot = monthly_snapshot(root / "election_watch.json", today, force=refresh)
        except OSError as exc:
            st.warning(f"Election snapshot could not be persisted: {exc}")
            return
        st.caption(f"Official terms verified: {snapshot.get('verified_at', 'Unknown')} · checked once per calendar month while the app is used. {snapshot.get('coverage', '')}")
        if snapshot.get("refresh_error"):
            st.caption(f"ECI refresh unavailable; previous verified snapshot retained. {snapshot['refresh_error']}")
        upcoming = upcoming_terms(snapshot, today)
        if upcoming:
            assembly = sum(row["Assembly seats"] for row in upcoming)
            lok = sum(row["Lok Sabha allocation"] for row in upcoming)
            rajya = sum(row["Rajya Sabha allocation"] for row in upcoming)
            st.markdown(f"**Next 18 months in this watchlist:** {len(upcoming)} states · **{assembly:,} assembly seats** · full-house majorities shown below.")
            st.caption(f"Those states have {lok} Lok Sabha seats and {rajya} Rajya Sabha seats allocated. These are geographic allocations—not seats changing in these assembly elections, party counts, coalition support, or Rajya Sabha seats immediately up for election.")
            for row in upcoming:
                st.markdown(f"**{row['State']}** · term ends **{row['Term ends']}** · {row['Assembly seats']} MLAs · full-house majority {row['Full-house majority']} · LS {row['Lok Sabha allocation']} / RS {row['Rajya Sabha allocation']}")
            nearest = upcoming[0]
            st.caption(f"Nearest known term end: {nearest['State']} in {nearest['Days to term end']} calendar days. Polling and counting dates are unverified here; term end is not an election announcement.")
        else:
            st.caption("No upcoming terms in the retained watchlist window. This does not establish that no elections are scheduled.")
        st.markdown(f"[ECI terms & seat allocations]({SOURCE}) · [ECI announcements](https://www.eci.gov.in/) · [How Rajya Sabha representation works]({PARLIAMENT_SOURCE})")
        st.caption("MLAs determine state governments and elect Rajya Sabha members as vacancies arise. Lok Sabha confidence determines the Union government's survival; state results alone do not change that majority. Assembly majorities here assume all seats filled; actual confidence votes can differ with vacancies and absences.")

        market_path = root / "election_market_context.json"
        try:
            market = json.loads(market_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            market = {}
        if refresh or market.get("attempted_month") != today.strftime("%Y-%m"):
            fetched = _market_snapshot(str(today))
            # Preserve successful data through provider failures, with failure dates.
            market["attempted_month"] = today.strftime("%Y-%m")
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
        st.markdown("**Measured market conditions**")
        if index:
            st.markdown(f"NIFTY 50: **{index['drawdown']:.1%} from its 252-session high** · 30-session realised volatility **{index['volatility']:.1%} annualised** · volatility percentile **{index['volatility_percentile']:.0f}/100** · data {index['as_of']}.")
        if vix:
            st.caption(f"India VIX: {vix['level']:.2f} · data {vix['as_of']}. VIX is an implied-volatility index, not a direction forecast.")
        if market.get("refresh_errors"):
            st.caption("Market refresh incomplete; retained values may be stale: " + str(market["refresh_errors"]))
        st.caption("Context: an approaching election and an observed decline/volatility rise justify checking the portfolio's existing buy scores, reserve and rebalance decision. They do not establish election causation, a bottom, or a profitable entry. This monthly/on-demand snapshot is not a live dip alert; the app must be running for refreshes.")
