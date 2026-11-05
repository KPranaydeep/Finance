"""Reserve evidence and independent refresh; no optimization or orders."""
from pathlib import Path
import pandas as pd
import streamlit as st
from election_schedule import monthly_schedule
from election_watch import monthly_snapshot
from election_reserve import saved_event_reference, reserve_decision_summary
from election_seat_reserve import rolling_seat_reserve, LABELS
from us_election_risk import us_election_reference, us_federal_events, us_event_context, allocation_inputs, next_us_event_review


def render_election_reserve(payload, base_fraction, base_value, scope, refresh_history, save, *, hydrate_history=None, refresh_portfolio=None):
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    with st.expander("Election reserve · India & US", expanded=True):
        refresh = st.button("Refresh election reserve", key=f"election_reserve_refresh_{scope}",
                            help="Refresh portfolio listing calendars, prices, FX and election evidence; no optimizer or trades.")
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
            houses = monthly_snapshot(Path.home()/".portfolio_rebalancer"/"election_watch.json", today, force=refresh)
            schedule = monthly_schedule(Path.home()/".portfolio_rebalancer"/"election_schedule.json", today, force=refresh, assembly_snapshot=houses)
            if refresh:
                # Calculate successfully before replacing the previous history.
                try:
                    if refresh_portfolio:
                        candidate=dict(payload)
                        refresh_portfolio(candidate)
                        save(candidate)
                        payload.update(candidate)
                        st.rerun()
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
        weights, metadata = allocation_inputs(payload)
        seat_policy = rolling_seat_reserve(schedule['events'], today, reference.get('inr_weight', 0.), assembly_snapshot=houses)
        if not schedule.get('coverage_complete') or reference.get('status') == 'incomplete_currency_or_weights':
            seat_policy['coverage_complete'] = False
        reference['seat_policy'] = seat_policy
        try:
            us = us_election_reference(payload.get('portfolio_evidence_history'), today, schedule['events'], current_weights=weights)
        except Exception as exc:
            us = dict(status='not_assessed', coverage_complete=False, active_events=[], windows=[],
                      event_reserve_fraction=None, reason='US election risk not assessed: '+str(exc))
        reference['us'] = us
        reference['coverage_complete'] = bool(us.get('coverage_complete')) and seat_policy['coverage_complete']
        summary = reserve_decision_summary(base_fraction, reference, us, seat_policy)
        reference['reserve_summary'] = summary
        final = summary['fraction']
        st.markdown(f"**Reserve reference: {final:.2%} · ₹{base_value*final:,.0f}** of total invested base")
        st.caption(summary['driver'] + ' · ' + summary['comparison'])
        st.markdown(f"**India · rolling 20 days: {seat_policy['window_start']} → {seat_policy['window_end']}**")
        with st.container(horizontal=True):
            for house, label in LABELS.items():
                fraction = seat_policy['fractions'][house]
                st.metric(label, f"{fraction:.2%}" if fraction is not None else "Unavailable",
                          help=f"{seat_policy['totals'][house]} verified contested seats / {seat_policy['denominators'][house]}; separate house denominator.")
        st.caption(f"Seat-policy reserve: {seat_policy['reserve_fraction']:.2%} of total base = INR allocation {seat_policy['inr_weight']:.1%} × largest house percentage. A chosen liquidity policy, not an estimated drawdown.")
        st.caption(f"House counts dated {seat_policy['denominator_verified_at']} · counting-date window, not assembly term ends.")
        if not seat_policy['coverage_complete']:
            st.caption("Seat schedule/count coverage incomplete; only known verified contests contribute. Unknown does not mean no elections.")
        if seat_policy['missing_counts']:
            st.caption("Counts unavailable: " + "; ".join(seat_policy['missing_counts']))
        timing = us_event_context(today)
        for event in timing['active_events']:
            if event['phase'] == 'post_election':
                st.markdown(f"**US · voting date passed {event['event_date']} · post-election watch through {event['watch_ends']}**")
                st.caption(event['name'] + " voting is no longer upcoming. Result certification and any remaining special/runoff elections are not inferred.")
            else:
                days = (pd.Timestamp(event['event_date']).date()-today).days
                st.markdown(f"**{event['name']} · {event['event_date']} · {days} calendar days away**")
        upcoming_us = timing['next_event']
        if upcoming_us:
            st.caption(f"Next regular US federal election: {upcoming_us['event_date']} · statutory date; separate from specials/runoffs.")
        if us.get('active_events'):
            st.caption('US watch expires automatically after the 20-calendar-day aftermath window. Historical stress uses complete 30-session price episodes, not a seat-percentage reserve.')
        event_review=next_us_event_review(today)
        if event_review:
            st.markdown(f"**US event review: {event_review} · NYSE session before/at election day**")
        if us.get('event_reserve_fraction') is None:
            (st.caption if us.get('status') == 'no_active_us_event' else st.warning)(us['reason'])
            if payload.get('portfolio_evidence_error'):
                st.caption('Evidence refresh could not finish: '+payload['portfolio_evidence_error'])
        else:
            st.markdown(f"**US whole-portfolio historical stress: {us['event_reserve_fraction']:.2%} of total base** · {len(us['windows'])} complete episodes · data {us.get('data_through')}")
            st.caption(us['reason'])
        quoted_usd = sum(w for t,w in weights.items() if metadata.get(t,{}).get('Currency') == 'USD')
        crypto = sum(w for t,w in weights.items() if metadata.get(t,{}).get('Exchange') == 'CCC')
        st.caption(f"USD quoted: {quoted_usd:.1%} · crypto: {crypto:.1%} of total base. US stress replays the complete INR-valued mix, including FX and INR holdings. Issuer domicile/sector sensitivity is not inferred from quote currency; foreign-listing classification remains unverified.")
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
            st.caption("Seat policy: sum verified seats with counting dates from today through today+19 separately for Assembly / LS / RS; divide by the dated national assembly total / 543 / 233, take max(), then multiply by current optimal INR allocation. Multi-phase polling counts each contest once. Term ends and state LS/RS geographic allocations never count as contests. Past results drop out immediately. This is the selected liquidity policy, not a calibrated risk model.")
            st.caption("Combined reserve = max(portfolio buffer, INR-scaled seat policy, active India historical stress, active US whole-portfolio stress). Amounts are not added and no orders are changed. India historical stress replays the current INR mix across complete 30-session episodes centred on prior counting dates; US stress replays the whole INR-valued mix including FX. Historical stress is not an election-causal estimate or a loss limit.")
            if history_error:
                st.caption("History refresh: " + history_error + ". Previous saved history retained.")
            if schedule.get("refresh_error"):
                st.caption("Schedule providers: " + schedule["refresh_error"])
            for window in reference.get("windows", []):
                st.caption(f"{window['counting_date']} · INR drawdown {window['loss']:.2%} · {window['start']} → {window['end']}")
            for event in schedule["events"]:
                st.markdown(f"[{event['name']} · {event['counting_date']}]({event['source']})")
            for event in us_federal_events(today):
                st.markdown(f"[{event['name']} · {event['event_date']}]({event['source']})")
            for window in us.get('windows', []):
                st.caption(f"US {window['event_date']} · full-mix drawdown {window['loss']:.2%} · {window['start']} → {window['end']}")
            st.caption('US scenarios include the same native-price/dated-FX paths for all holdings, preserving co-movement. Different country scenario maxima are not added. If India/US windows overlap without a validated joint analog, coverage remains partial. Thirty-session windows and historical maxima are stress assumptions; few episodes do not validate an optimal reserve.')
        return final, reference
