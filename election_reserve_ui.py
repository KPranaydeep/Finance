"""Reserve evidence and independent refresh; no optimization or orders."""
from pathlib import Path
import pandas as pd
import streamlit as st
from election_schedule import monthly_schedule
from election_reserve import saved_event_reference, reserve_decision_summary
from us_election_risk import us_election_reference, US_EVENTS, allocation_inputs, next_us_event_review


def render_election_reserve(payload, base_fraction, base_value, scope, refresh_history, save, *, hydrate_history=None):
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    with st.expander("Election reserve · India & US", expanded=True):
        refresh = st.button("Refresh election reserve", key=f"election_reserve_refresh_{scope}",
                            help="Refresh only optimal INR histories and official schedules; no optimizer or trades.")
        history_error = None
        attempt_key = "election_reserve_hydrated_" + str(payload.get("run_id") or payload.get("saved_at") or scope)
        if not payload.get("election_reserve_history") and hydrate_history and not st.session_state.get(attempt_key):
            st.session_state[attempt_key] = True
            try:
                history = hydrate_history(payload)
                save(dict(payload, election_reserve_history=history))
                payload["election_reserve_history"] = history
            except Exception as exc:
                history_error = str(exc)
        try:
            schedule = monthly_schedule(Path.home()/".portfolio_rebalancer"/"election_schedule.json", today, force=refresh)
            if refresh:
                # Calculate successfully before replacing the previous history.
                try:
                    history = refresh_history(payload)
                    candidate = dict(payload, election_reserve_history=history)
                    saved_event_reference(candidate, schedule["events"], today)
                    save(candidate)
                    payload["election_reserve_history"] = history
                    st.success("INR election history updated and saved; optimizer unchanged.")
                except Exception as exc:
                    history_error = str(exc)
            reference = saved_event_reference(payload, schedule["events"], today)
        except Exception as exc:
            st.caption(f"Election reserve update unavailable; portfolio reserve retained. {exc}")
            return float(base_fraction), None
        try:
            us = us_election_reference(payload.get('portfolio_evidence_history'), today, schedule['events'])
        except Exception as exc:
            us = dict(status='not_assessed', coverage_complete=False, active_events=[], windows=[],
                      event_reserve_fraction=None, reason='US election risk not assessed: '+str(exc))
        reference['us'] = us
        reference['coverage_complete'] = bool(us.get('coverage_complete')) and not bool(schedule.get('refresh_error'))
        summary = reserve_decision_summary(base_fraction, reference, us)
        reference['reserve_summary'] = summary
        final = summary['fraction']
        upcoming_us = next((e for e in US_EVENTS if e['event_date'] >= str(today)), None)
        if upcoming_us:
            days = (pd.Timestamp(upcoming_us['event_date']).date()-today).days
            st.markdown(f"**US election · {upcoming_us['event_date']} · {days} calendar days away**")
            st.caption('House: 435/435 seats (100%) · Senate: 33/100 regular-cycle seats (33%), plus specials not included in this count. Congressional control is at stake; this is not a presidential election.')
        if us.get('active_events'):
            st.caption('Active US watch: 30 NYSE sessions before/after election day. Results may emerge over multiple days; no fixed certification/counting day is assumed.')
        event_review=next_us_event_review(today)
        if event_review:
            st.markdown(f"**US event review: {event_review} · NYSE session before/at election day**")
        if us.get('event_reserve_fraction') is None:
            st.warning(us['reason'])
        else:
            st.markdown(f"**US whole-portfolio historical stress: {us['event_reserve_fraction']:.2%} of total base** · {len(us['windows'])} complete episodes · data {us.get('data_through')}")
            st.caption(us['reason'])
        weights, metadata = allocation_inputs(payload)
        quoted_usd = sum(w for t,w in weights.items() if metadata.get(t,{}).get('Currency') == 'USD')
        crypto = sum(w for t,w in weights.items() if metadata.get(t,{}).get('Exchange') == 'CCC')
        st.caption(f"USD quoted: {quoted_usd:.1%} · crypto: {crypto:.1%} of total base. US stress replays the complete INR-valued mix, including FX and INR holdings. Issuer domicile/sector sensitivity is not inferred from quote currency; foreign-listing classification remains unverified.")
        st.markdown(f"**Reserve reference: {final:.2%} · ₹{base_value*final:,.0f}** of total invested base · saved optimal mix")
        st.caption(summary['driver'])
        st.caption(summary['comparison'])
        if not reference['coverage_complete']:
            st.caption('Partial event coverage: this is a limited reserve reference, not a complete or optimal portfolio protection amount. Country amounts are never summed.')
        for event in reference.get("active_events", []):
            st.markdown(f"**{event['name']}** · prepare from **{event['preparation_date']}** · counting **{event['counting_date']}**")
            st.caption("Polls: " + ", ".join(event.get("poll_dates", [])))
        future = sorted([e for e in schedule["events"] if e["counting_date"] >= str(today)], key=lambda e:e["counting_date"])
        if not reference.get("active_events"):
            st.caption(f"Next verified counting: {future[0]['counting_date']}." if future
                       else "No upcoming counting date verified in the retained schedule. Term ends do not activate a reserve.")
        status = reference["status"]
        if status in {"missing_inr_history", "incomplete_currency_or_weights", "insufficient_event_history"}:
            st.caption("Election history: " + status.replace("_", " ") + ". Refresh can retrieve missing prices without optimization.")
        exposure = f"{reference['inr_weight']:.1%}" if reference.get("inr_weight") is not None else "unavailable"
        st.caption(f"Optimal INR allocation: {exposure} of total base · {len(reference.get('windows', []))} non-overlapping historical episodes · prices through {reference.get('data_through') or 'unavailable'}.")
        if reference.get("historical_stress_fraction") is not None:
            st.caption(f"Historical event stress: {reference['historical_stress_fraction']:.2%} of total base · applied only in a verified event window.")
        if schedule.get("refresh_error"):
            st.caption("Schedule refresh incomplete · retained verified dates; coverage may be incomplete.")
        with st.expander("Method, historical windows & source status", expanded=False):
            st.caption("Stress reference = INR allocation × largest observed drawdown of that INR mix across 30 NSE sessions centred on verified past counting dates (15 before, 15 from counting). The same current weights are replayed; this is an in-sample scenario, not an election-causal estimate or a loss limit. Applies from 30 NSE sessions before counting (or ongoing polls) through 30 sessions afterwards. Combined reserve uses max(), never a sum or seat percentage; no trades are changed.")
            if history_error:
                st.caption("History refresh: " + history_error + ". Previous saved history retained.")
            if schedule.get("refresh_error"):
                st.caption("Schedule providers: " + schedule["refresh_error"])
            for window in reference.get("windows", []):
                st.caption(f"{window['counting_date']} · INR drawdown {window['loss']:.2%} · {window['start']} → {window['end']}")
            for event in schedule["events"]:
                st.markdown(f"[{event['name']} · {event['counting_date']}]({event['source']})")
            for event in US_EVENTS:
                st.markdown(f"[{event['name']} · {event['event_date']}]({event['source']})")
            for window in us.get('windows', []):
                st.caption(f"US {window['event_date']} · full-mix drawdown {window['loss']:.2%} · {window['start']} → {window['end']}")
            st.caption('US scenarios include the same native-price/dated-FX paths for all holdings, preserving co-movement. Different country scenario maxima are not added. If India/US windows overlap without a validated joint analog, coverage remains partial. Thirty-session windows and historical maxima are stress assumptions; few episodes do not validate an optimal reserve.')
        return final, reference
