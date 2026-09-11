"""Read-only monitoring panel; expensive computation runs in the workflow only."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import pandas as pd
import streamlit as st
from . import store


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def load_events(basket_id):
    from public_basket_postgres import get_public_basket_database_url, connect_public_basket_db
    with connect_public_basket_db(get_public_basket_database_url()) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        return store.read(conn, basket_id)


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def load_fresh_preview(basket_id, publication_id):
    from .preview import historical_preview
    from .config import load_policy
    from .service import publications
    from public_basket_postgres import get_public_basket_database_url, connect_public_basket_db
    policy = load_policy()
    with connect_public_basket_db(get_public_basket_database_url()) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        pubs = publications(conn, basket_id)
        publication = next(p for p in pubs if p["publication_id"] == publication_id)
        events = store.read(conn, basket_id)
    return historical_preview(publication, policy, events)


def render_crossings(forecast):
    rows = forecast.get("security_crossings", [])
    if not rows:
        return
    st.caption(f"Net annualized target: {percent(forecast['target_xirr'])} · "
               f"Crossing probability threshold: {percent(forecast['crossing_probability_threshold'])}. "
               "A probability threshold is not statistical confidence or a guaranteed exit date.")
    st.table(pd.DataFrame([{"Security": r["ticker"],
                           "Estimated crossing": r["crossing_date"] or "Not reached in horizon",
                           "Probability by date": percent(r["probability"])} for r in rows]))
    st.caption("Research estimates unless validation passes. A crossing requests a review, not an automatic sale. Short-term XIRR can look large despite a small rupee gain.")


def render_fresh_preview(p):
    d, f = p["decision"], p["forecast"]
    # Retain earlier page-view dates in this session; durable workflow dates
    # are also honored by the assessment engine.
    key = "review_promise_" + p["publication_id"] + "_" + str(p.get("ack_epoch", 0))
    candidate = d.get("next_review")
    prior = st.session_state.get(key)
    date = min(x for x in (candidate, prior) if x) if candidate or prior else None
    st.session_state[key] = date
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    due = bool(d.get("reasons")) or bool(date and date <= today)
    st.metric("Latest suggested review", "Review now" if due else (date or "Next session risk check"))
    if p["provisional"]:
        st.caption("Provisional: " + p["assumption"])
    if d.get("target_crossed_securities"):
        st.warning("Net target already crossed: " + ", ".join(d["target_crossed_securities"]) + ". Review costs and risk before selling.")
    if f.get("next_review") is None:
        st.caption("Target-crossing timing is not validated. Use the risk-review fallback, not the research date as a sell instruction.")
    st.caption(f"Prices through {p['as_of']} · Assessed {p['checked_at']} · Daily data, not live quotes; cache up to five minutes.")
    st.caption("Currency: NSE-listed holdings are priced in INR, including overseas ETFs. Their INR prices already reflect FX exposure; no second USD/INR conversion is applied.")
    if p.get("history_coverage"):
        h = p["history_coverage"]
        st.caption(f"Shared history: {h['start']} to {h['end']} · {h['usable_daily_returns']} valid daily returns · {len(h['missing_sessions'])} incomplete sessions excluded. No price filling.")
    with st.expander("Earliest security target crossings", expanded=True):
        render_crossings(f)


def render_pending(row, now):
    if row is None:
        st.info("No monitoring record exists for this publication yet. Run the enabled model-review workflow; if it already ran, check that the page and workflow use the same basket and database.")
        return
    p = row["payload"]
    if p.get("reason") == "AWAITING_MARKET_ENTRY":
        st.info("Awaiting market entry—not a failure. The publication needs its first completed eligible trading session before the model can freeze an entry and calculate returns.")
        if p.get("ready_at"):
            ready = datetime.fromisoformat(p["ready_at"]).astimezone(ZoneInfo("Asia/Kolkata"))
            st.caption(f"Entry session: {p.get('entry_date')} · Earliest assessment: {ready:%d %b %Y %H:%M IST}, once prices are available. The daily workflow checks automatically when enabled.")
    else:
        st.warning("Cannot assess: " + p.get("reason", "MONITOR_CHECK_FAILED").replace("_", " ").lower())
        if p.get("stage"):
            st.caption("Check stage: " + p["stage"].replace("_", " "))
    if p.get("checked_at"):
        checked = datetime.fromisoformat(p["checked_at"])
        st.caption(f"Last check: {checked.astimezone(ZoneInfo('Asia/Kolkata')):%d %b %Y %H:%M IST}")
        if (now - checked).total_seconds() > 30 * 3600:
            st.warning("This check is stale. Confirm the daily model-review workflow is running.")


def render_events(events, active_ids=None, now=None, latest_publication_id=None):
    now = now or datetime.now(timezone.utc)
    baseline_rows = [r for r in events if r["kind"] == "BASELINE" and
                     (active_ids is None or r["payload"]["publication_id"] in active_ids)]
    pending_shown = False
    if latest_publication_id and not any(r["payload"]["publication_id"] == latest_publication_id for r in baseline_rows):
        pending = next((r for r in reversed(events) if r["kind"] in {"WAITING", "FAILURE"} and
                        r["payload"].get("publication_id", r.get("baseline_id")) == latest_publication_id), None)
        render_pending(pending, now)
        pending_shown = True
        if baseline_rows:
            st.caption("The latest publication has no frozen entry yet. Previously created model investments remain available below.")
    if not baseline_rows:
        if not pending_shown:
            pending = next((r for r in reversed(events) if r["kind"] in {"WAITING", "FAILURE"} and
                            (active_ids is None or r["payload"].get("publication_id", r.get("baseline_id")) in active_ids)), None)
            render_pending(pending, now)
        return
    baselines = sorted([r["payload"] for r in baseline_rows], key=lambda b: b["portfolio_version"], reverse=True)
    selected = st.selectbox("Model investment publication", range(len(baselines)),
                            format_func=lambda i: f'P{baselines[i]["portfolio_version"]:03d} · entry {baselines[i]["entry_date"]}',
                            key="review_publication")
    baseline = baselines[selected]
    bid = baseline["baseline_id"]
    last = store.latest(events, "ASSESSMENT", bid)
    failure = store.latest(events, "FAILURE", bid)
    heartbeat = store.latest(events, "HEARTBEAT", bid)
    st.caption(f"Frozen model capital ₹{baseline['capital']:,.2f} · Publication-based estimate, not your broker account. Selecting a version does not reset any investment.")
    successful_seq = max((last or {}).get("seq", 0), (heartbeat or {}).get("seq", 0))
    if not last or (failure and failure["seq"] > successful_seq):
        st.warning("Cannot assess: " + (failure["payload"]["reason"].replace("_", " ").lower() if failure else "awaiting first assessment"))
        return
    p = last["payload"]
    checked_at = heartbeat["payload"]["at"] if heartbeat else p["checked_at"]
    age = (now - datetime.fromisoformat(checked_at)).total_seconds() / 3600
    if age > 30 or age < -1:
        st.warning("Monitoring heartbeat is stale. Do not rely on the previous review date. Check the scheduled workflow.")
        return
    m, d = p["metrics"], p["decision"]
    status = d["status"].replace("_", " ").capitalize()
    if d["reasons"]:
        st.warning(status + " — review the estimates before acting. No trades have been submitted.")
    else:
        st.info("No configured trigger detected. This is not a guarantee against losses.")
    with st.container(horizontal=True):
        st.metric("Next suggested review", "Now" if d["reasons"] else (d["next_review"] or "Not validated"))
        st.metric("Estimated net XIRR", percent(m["xirr"]))
        st.metric("Estimated net profit", f"₹{m['net_profit']:,.2f}")
        st.metric("Estimated exit proceeds", f"₹{m['net_proceeds']:,.2f}")
    checked = datetime.fromisoformat(checked_at).astimezone(ZoneInfo("Asia/Kolkata"))
    st.caption(f"{m['days_held']} days held · Absolute net return {percent(m['net_total_return'])} · Prices through {p['as_of']} · Checked {checked:%d %b %Y %H:%M IST}")
    st.caption("100% annualized XIRR does not mean your investment has doubled. A review date is not an optimal selling date.")
    sent = store.latest(events, "ALERT_SENT", bid)
    alert_failed = store.latest(events, "ALERT_FAILED", bid)
    if not sent or (alert_failed and alert_failed["seq"] > sent["seq"]) or (heartbeat and not heartbeat["payload"].get("alerts_enabled")):
        st.warning("Alert delivery has not been confirmed for this investment. Do not rely on background notifications yet.")
    with st.expander("Security returns and exit choices"):
        # Static formatted tables avoid the mobile virtualized-row issue.
        frame = pd.DataFrame([{"Security": r["ticker"], "Shares": r["shares"],
                               "Price": f"₹{r['price']:,.2f}", "Net profit": f"₹{r['net_profit']:,.2f}",
                               "Net XIRR": percent(r["xirr"])} for r in m["rows"]])
        st.table(frame)
        options = pd.DataFrame([{"Choice": c["option"], "Cash available": f"₹{c['cash_raised']:,.2f}" if "cash_raised" in c else "N/A",
                                 "Charges + slippage": f"₹{c.get('fees', 0):,.2f}",
                                 "Estimated tax": f"₹{c.get('tax', 0):,.2f}",
                                 "Status": c.get("status", "comparison only").replace("_", " ").lower()} for c in p["comparisons"]])
        st.table(options)
        st.caption("Sale choices minimize estimated fees plus immediate tax within the disclosed whole-share candidate grid—not guaranteed lifetime tax or total risk. Proceeds include existing model cash. Capital recovery may leave a concentrated residual portfolio.")
        st.download_button("Download model comparison (.json)", json.dumps(p["comparisons"], indent=2),
                           file_name=f"model-exit-comparison-P{baseline['portfolio_version']:03d}.json", mime="application/json")
    with st.expander("Review-date evidence and assumptions"):
        forecast = p["forecast"]
        render_crossings(forecast)
        st.write("Forecast status: " + forecast["status"].replace("_", " ").lower())
        if forecast.get("curve"):
            curve = pd.DataFrame(forecast["curve"])
            curve["date"] = pd.to_datetime(curve["date"])
            st.line_chart(curve, x="date", y="any_review_probability", y_label="Estimated probability of a review trigger")
            st.caption("Research candidate: " + forecast["research_candidate"] + ". Not an actionable date unless validation passes.")
        st.write("Rebalance benefit gate: " + d["rebalance"]["status"].replace("_", " ").lower())
        st.caption("The six-percentage-point annual improvement rule is not inferred from short-horizon median returns. Without a validated comparable benefit estimate, no return-seeking rebalance signal is issued.")
        st.json(p["validation"])
        st.json(p["policy"])
        st.caption(p["dividend_assumption"])
        st.caption("Tariff scope: normal funded resident-individual delivery. NSE holdings use the Groww/Zerodha conservative envelope; direct US listings use Tickertape Pro trading charges and HDFC FX-GST assumptions. No annual exemption or loss-offset credit. Not a SEBI certification or a personal tax calculation. MMI is context only.")
        st.json(p["mmi"])
        st.download_button("Download model review evidence (.json)", json.dumps({"baseline": baseline, "assessment": p,
                            "assessment_hash": last["event_hash"]}, indent=2),
                           file_name="public-model-review-evidence.json", mime="application/json")


def render_review_panel(basket_id, active_publications):
    st.subheader("Your next portfolio review")
    if active_publications:
        try:
            with st.spinner("Assessing security targets from available history..."):
                preview = load_fresh_preview(basket_id, active_publications[0]["publication_id"])
            render_fresh_preview(preview)
        except Exception as exc:
            from .service import SAFE_ERRORS
            allowed = SAFE_ERRORS | {"POLICY_APPROVAL_REQUIRED", "TARIFF_REVIEW_REQUIRED",
                                     "UNSUPPORTED_TAX_OR_ACCOUNT_PROFILE", "NSE_CLASSIFICATION_REQUIRED",
                                     "INTEGER_POLICY_REQUIRED", "INVALID_CAPITAL"}
            code = str(exc) if isinstance(exc, ValueError) and str(exc) in allowed else {
                FileNotFoundError: "POLICY_FILE_MISSING",
                ModuleNotFoundError: "PREVIEW_MODULE_MISSING",
                StopIteration: "PUBLICATION_NOT_FOUND",
            }.get(type(exc), "PREVIEW_CHECK_FAILED")
            st.warning("Fresh historical review unavailable: " + code + ". No reliable fresh date is implied.")
            if code == "FOREIGN_REVIEW_COST_MODEL_REQUIRED":
                st.caption("This publication contains direct overseas listings. INR pricing is separate from tax classification. The review engine supports NSE delivery only; overseas brokerage, remittance charges and instrument-specific tax treatment must be integrated before net-XIRR review dates can be shown.")
            if code == "INSTRUMENT_CLASSIFICATION_REQUIRED":
                st.caption("Add explicit instrument_kinds for every published ticker in public_review_policy.json. Stocks and overseas/gold ETFs have different modeled tax treatment; do not default every ticker to equity.")
    try:
        events = load_events(basket_id)
        render_events(events, {p["publication_id"] for p in active_publications},
                      latest_publication_id=active_publications[0]["publication_id"] if active_publications else None)
    except Exception:
        st.warning("Model review inspection is unavailable. No reliable review date can be shown.")
