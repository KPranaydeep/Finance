"""Read-only monitoring panel; expensive computation runs in the workflow only."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import pandas as pd
import streamlit as st
from . import store
from .market import ENTRY_MODEL_VERSION


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
        captured, total = p.get("captured_entries"), p.get("total_entries")
        if captured is not None and total:
            st.info(f"Security entries captured: {captured} of {total}. Each security enters independently according to its own exchange session.")
        else:
            st.info("Awaiting market entry—not a failure. Each security enters at publication when its exchange is trading, or after its own next open plus the configured wait.")
        if p.get("wait_reason") == "ENTRY_DATA_RETRY":
            ticker = p.get("pending_ticker") or "A pending security"
            st.caption(f"{ticker}: the intraday provider did not return usable data. No price was invented; the next workflow run will retry.")
        if p.get("wait_reason") == "FX_DATA_RETRY":
            ticker = p.get("pending_ticker") or "A pending overseas security"
            st.caption(f"{ticker}: a sufficiently recent USD/INR quote was unavailable at its entry trade. No FX rate was invented; the next workflow run will retry.")
        if p.get("ready_at"):
            ready = datetime.fromisoformat(p["ready_at"]).astimezone(ZoneInfo("Asia/Kolkata"))
            st.caption(f"Next pending entry check: {ready:%d %b %Y %H:%M IST}. The workflow checks automatically when enabled.")
        if p.get("pending_tickers"):
            st.caption("Pending securities: " + ", ".join(p["pending_tickers"]))
    else:
        st.warning("Cannot assess: " + p.get("reason", "MONITOR_CHECK_FAILED").replace("_", " ").lower())
        if p.get("stage"):
            st.caption("Check stage: " + p["stage"].replace("_", " "))
    if p.get("checked_at"):
        checked = datetime.fromisoformat(p["checked_at"])
        st.caption(f"Last check: {checked.astimezone(ZoneInfo('Asia/Kolkata')):%d %b %Y %H:%M IST}")
        if (now - checked).total_seconds() > 30 * 3600:
            st.warning("This check is stale. Confirm the daily model-review workflow is running.")


def render_partial_security_reviews(events, publication_id):
    latest = {}
    for row in events:
        if (row["kind"] == "SECURITY_REVIEW_PREVIEW" and
                row["payload"].get("publication_id") == publication_id):
            latest[row["payload"]["ticker"]] = row["payload"]
    if not latest:
        return
    st.markdown("#### Provisional security review estimates")
    st.caption("Available for captured entries while the basket is incomplete. One-share, fully costed security-only estimates; the completed basket baseline will supersede them.")
    st.table(pd.DataFrame([{
        "Security": payload["ticker"],
        "Entry price": f"₹{payload['entry_price_inr']:,.2f}",
        "Data through": payload["as_of"],
        "Estimated target crossing": payload.get("estimated_crossing_date") or "Not reached in horizon",
        "Probability": percent(payload.get("crossing_probability")),
    } for payload in sorted(latest.values(), key=lambda item: item["ticker"])]))
    st.caption("Research estimates—not sell dates or recommendations. Fixed costs are conservative at one share; results may change when final basket quantities are known.")


def render_events(events, active_ids=None, now=None, latest_publication_id=None,
                  suppress_latest_failure=False):
    now = now or datetime.now(timezone.utc)
    baseline_by_publication = {}
    for row in events:
        if (row["kind"] == "BASELINE" and
                (active_ids is None or row["payload"]["publication_id"] in active_ids)):
            baseline_by_publication[row["payload"]["publication_id"]] = row
    baseline_rows = list(baseline_by_publication.values())
    pending_shown = False
    if latest_publication_id and not any(r["payload"]["publication_id"] == latest_publication_id for r in baseline_rows):
        pending = next((r for r in reversed(events) if r["kind"] in {"WAITING", "FAILURE"} and
                        (not suppress_latest_failure or r["kind"] != "FAILURE") and
                        (r["kind"] != "WAITING" or r["payload"].get("entry_model_version") == ENTRY_MODEL_VERSION) and
                        r["payload"].get("publication_id", r.get("baseline_id")) == latest_publication_id), None)
        if pending is not None or not suppress_latest_failure:
            render_pending(pending, now)
        pending_shown = True
        if baseline_rows:
            st.caption("The latest publication has no frozen entry yet. Previously created model investments remain available below.")
            render_partial_security_reviews(events, latest_publication_id)
    if not baseline_rows:
        if not pending_shown:
            pending = next((r for r in reversed(events) if r["kind"] in {"WAITING", "FAILURE"} and
                            (not suppress_latest_failure or r["kind"] != "FAILURE") and
                            (r["kind"] != "WAITING" or r["payload"].get("entry_model_version") == ENTRY_MODEL_VERSION) and
                            (active_ids is None or r["payload"].get("publication_id", r.get("baseline_id")) in active_ids)), None)
            render_pending(pending, now)
        if latest_publication_id:
            render_partial_security_reviews(events, latest_publication_id)
        return
    baselines = sorted([r["payload"] for r in baseline_rows], key=lambda b: b["portfolio_version"], reverse=True)
    selected = st.selectbox("Model investment publication", range(len(baselines)),
                            format_func=lambda i: f'P{baselines[i]["portfolio_version"]:03d} · entry {baselines[i]["entry_date"]}',
                            key="review_publication")
    baseline = baselines[selected]
    bid = baseline["baseline_id"]
    last = store.latest(events, "ASSESSMENT", bid)
    failure = store.latest(events, "FAILURE", bid)
    waiting = store.latest(events, "WAITING", bid)
    heartbeat = store.latest(events, "HEARTBEAT", bid)
    st.caption(f"Frozen model capital ₹{baseline['capital']:,.2f} · Publication-based estimate, not your broker account. Selecting a version does not reset any investment.")
    with st.expander("Security entry evidence"):
        st.table(pd.DataFrame([{
            "Security": lot["ticker"],
            "Requested entry": lot.get("entry_at") or lot["entry_date"],
            "Price timestamp": lot.get("entry_quote_at") or "Legacy daily open",
            "Entry price": f"₹{lot['price']:,.2f}",
            "Basis": lot.get("entry_basis", "Legacy shared session open").replace("_", " ").lower(),
        } for lot in baseline["lots"]]))
    successful_seq = max((last or {}).get("seq", 0), (heartbeat or {}).get("seq", 0))
    if not last or (failure and failure["seq"] > successful_seq):
        if waiting and waiting["payload"].get("entry_frozen"):
            message = "Entry established from verified opening prices; awaiting the first completed global assessment session."
            ready_at = waiting["payload"].get("ready_at")
            if ready_at:
                ready = datetime.fromisoformat(ready_at).astimezone(ZoneInfo("Asia/Kolkata"))
                message += f" Earliest assessment: {ready:%d %b %Y %H:%M IST}."
            st.info(message)
        else:
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
    fresh_waiting = False
    if active_publications:
        try:
            with st.spinner("Assessing security targets from available history..."):
                preview = load_fresh_preview(basket_id, active_publications[0]["publication_id"])
            render_fresh_preview(preview)
        except Exception as exc:
            from .service import SAFE_ERRORS
            allowed = SAFE_ERRORS | {"POLICY_APPROVAL_REQUIRED", "TARIFF_REVIEW_REQUIRED",
                                     "UNSUPPORTED_TAX_OR_ACCOUNT_PROFILE", "NSE_CLASSIFICATION_REQUIRED",
                                     "INTEGER_POLICY_REQUIRED", "INVALID_CAPITAL",
                                     "INVALID_POLICY_ENTRY_QUOTE_INTERVAL"}
            code = str(exc) if isinstance(exc, ValueError) and str(exc) in allowed else {
                FileNotFoundError: "POLICY_FILE_MISSING",
                ModuleNotFoundError: "PREVIEW_MODULE_MISSING",
                StopIteration: "PUBLICATION_NOT_FOUND",
            }.get(type(exc), "PREVIEW_CHECK_FAILED")
            if code == "AWAITING_MARKET_ENTRY":
                fresh_waiting = True
                st.info("Opening-price entry is waiting for every represented market to open, or has been established while the first completed-session assessment is still pending.")
                if getattr(exc, "wait_reason", None) == "ENTRY_DATA_RETRY":
                    st.caption("Intraday data is temporarily unavailable for " +
                               str(getattr(exc, "pending_ticker", "a pending security")) +
                               ". No price was invented; the workflow will retry.")
                if getattr(exc, "wait_reason", None) == "FX_DATA_RETRY":
                    st.caption("A sufficiently recent USD/INR quote is temporarily unavailable for " +
                               str(getattr(exc, "pending_ticker", "a pending overseas security")) +
                               ". No FX rate was invented; the workflow will retry.")
            else:
                st.warning("Fresh historical review unavailable: " + code + ". No reliable fresh date is implied.")
            if code == "FOREIGN_REVIEW_COST_MODEL_REQUIRED":
                st.caption("This publication contains direct overseas listings. INR pricing is separate from tax classification. The review engine supports NSE delivery only; overseas brokerage, remittance charges and instrument-specific tax treatment must be integrated before net-XIRR review dates can be shown.")
            if code == "INSTRUMENT_CLASSIFICATION_REQUIRED":
                st.caption("Automatic classification could not safely identify one or more published instruments. The owner policy does not require a ticker list; retry after the NSE/Yahoo metadata source is available or add support for the unrecognized instrument category in code.")
    try:
        events = load_events(basket_id)
        render_events(events, {p["publication_id"] for p in active_publications},
                      latest_publication_id=active_publications[0]["publication_id"] if active_publications else None,
                      suppress_latest_failure=fresh_waiting)
    except Exception:
        st.warning("Model review inspection is unavailable. No reliable review date can be shown.")
