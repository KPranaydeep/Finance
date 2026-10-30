"""Reserve evidence and independent refresh; no optimization or orders."""
from pathlib import Path
import pandas as pd
import streamlit as st
from election_schedule import monthly_schedule
from election_reserve import saved_event_reference, combined_reserve


def render_election_reserve(payload, base_fraction, base_value, scope, refresh_history, save):
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    with st.expander("INR election reserve · dates & stress", expanded=False):
        refresh = st.button("Refresh election reserve", key=f"election_reserve_refresh_{scope}",
                            help="Refresh only optimal INR histories and official schedules; no optimizer or trades.")
        try:
            schedule = monthly_schedule(Path.home()/".portfolio_rebalancer"/"election_schedule.json", today, force=refresh)
            if refresh:
                # Calculate successfully before replacing the previous history.
                history = refresh_history(payload)
                candidate = dict(payload, election_reserve_history=history)
                saved_event_reference(candidate, schedule["events"], today)
                payload["election_reserve_history"] = history
                save(payload)
                st.success("INR election history updated and saved; optimizer unchanged.")
            reference = saved_event_reference(payload, schedule["events"], today)
        except Exception as exc:
            st.caption(f"Election reserve update unavailable; portfolio reserve retained. {exc}")
            return float(base_fraction), None
        final = combined_reserve(base_fraction, reference)
        event_fraction = reference.get("event_reserve_fraction")
        st.markdown(f"**Combined reserve: {final:.2%} · ₹{base_value*final:,.0f}** of total invested base")
        st.caption(f"Portfolio buffer {base_fraction:.2%} · active election stress {event_fraction:.2%}" if event_fraction is not None
                   else f"Portfolio buffer {base_fraction:.2%} · no active supported election amount")
        for event in reference.get("active_events", []):
            st.markdown(f"**{event['name']}** · prepare from **{event['preparation_date']}** · counting **{event['counting_date']}**")
            st.caption("Polls: " + ", ".join(event.get("poll_dates", [])))
        future = sorted([e for e in schedule["events"] if e["counting_date"] >= str(today)], key=lambda e:e["counting_date"])
        if not reference.get("active_events"):
            st.caption(f"Next verified counting: {future[0]['counting_date']}." if future
                       else "No upcoming counting date verified in the retained schedule. Term ends do not activate a reserve.")
        status = reference["status"]
        if status in {"missing_inr_history", "incomplete_currency_or_weights", "insufficient_event_history"}:
            st.caption("Election evidence incomplete: " + status.replace("_", " ") + ". Refresh election reserve to load available INR histories.")
        exposure = f"{reference['inr_weight']:.1%}" if reference.get("inr_weight") is not None else "unavailable"
        st.caption(f"Optimal INR allocation: {exposure} of total base · {len(reference.get('windows', []))} non-overlapping historical episodes · prices through {reference.get('data_through') or 'unavailable'}.")
        st.caption("Stress reference = INR allocation × largest observed drawdown of that INR mix across 30 NSE sessions centred on verified past counting dates (15 before, 15 from counting). The same current weights are replayed; this is an in-sample scenario, not an election-causal estimate or a loss limit. Applies from 30 NSE sessions before counting (or ongoing polls) through 30 sessions afterwards. Combined reserve uses max(), never a sum or seat percentage; no trades are changed.")
        if schedule.get("refresh_error"):
            st.caption("Schedule discovery incomplete; retained verified dates. " + schedule["refresh_error"])
        with st.expander("Historical windows & official sources", expanded=False):
            for window in reference.get("windows", []):
                st.caption(f"{window['counting_date']} · INR drawdown {window['loss']:.2%} · {window['start']} → {window['end']}")
            for event in schedule["events"]:
                st.markdown(f"[{event['name']} · {event['counting_date']}]({event['source']})")
        return final, reference
