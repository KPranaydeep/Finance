"""Read-only monitoring panel; expensive computation runs in the workflow only."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import math
import pandas as pd
import streamlit as st
from . import store
from .forecast import TIMING_MODEL
from .market import ENTRY_MODEL_VERSION


@st.cache_data(ttl=300, max_entries=16, show_spinner=False)
def load_events(basket_id):
    from public_basket_postgres import get_public_basket_database_url, connect_public_basket_db
    with connect_public_basket_db(get_public_basket_database_url()) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        return store.read(conn, basket_id)


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


REVIEW_SORT_OPTIONS = (
    "Probability by date (high to low)",
    "Review date (earliest first)",
    "Target weight (high to low)",
    "Security (A to Z)",
)


def sort_security_estimates(rows, sort_by):
    """Return review estimates in a predictable, numerically correct order."""
    def probability(row):
        value = row.get("probability", row.get("crossing_probability"))
        return float(value) if value is not None else -1.0

    def target_weight(row):
        value = row.get("target_weight")
        return float(value) if value is not None else -1.0

    def review_date(row):
        return (row.get("review_date") or row.get("crossing_date")
                or row.get("estimated_crossing_date") or "9999-12-31")

    def ticker(row):
        return str(row.get("ticker") or "")

    if sort_by == "Review date (earliest first)":
        key = lambda row: (review_date(row), -probability(row),
                           -target_weight(row), ticker(row))
    elif sort_by == "Target weight (high to low)":
        key = lambda row: (-target_weight(row), -probability(row),
                           review_date(row), ticker(row))
    elif sort_by == "Security (A to Z)":
        key = lambda row: (ticker(row),)
    else:
        # The default is the most useful planning view: credible candidates
        # first, with the earlier date and larger allocation breaking ties.
        key = lambda row: (-probability(row), review_date(row),
                           -target_weight(row), ticker(row))
    return sorted(rows, key=key)


def is_current_preview(row):
    """Reject durable previews produced by an older timing methodology."""
    if not row:
        return False
    forecast = row.get("payload", {}).get("forecast", {})
    return forecast.get("timing_model") == TIMING_MODEL


def has_durable_preview(events, publication_id):
    """Whether the latest publication already has a usable stored preview."""
    baseline_ids = {
        row["payload"].get("baseline_id", row.get("baseline_id"))
        for row in (events or [])
        if row["kind"] == "BASELINE"
        and row["payload"].get("publication_id") == publication_id
    }
    return any(
        row["kind"] == "PREVIEW" and row.get("baseline_id") in baseline_ids
        and is_current_preview(row)
        for row in (events or [])
    )


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


def render_crossings(forecast, sort_key="security_crossings_sort"):
    rows = forecast.get("security_crossings", [])
    reached = [row for row in rows if row.get("crossing_date")]
    if not reached:
        st.caption("No security reaches the configured probability threshold within the forecast horizon.")
        return
    observation_sessions = int(forecast.get("minimum_forecast_review_sessions", 1))
    minimum_return = forecast.get("minimum_net_return")
    return_floor = (percent(minimum_return) if minimum_return is not None
                    else "legacy break-even only")
    st.caption(f"Profit gate: minimum net return of {return_floor} after modeled round-trip friction · "
               f"Net annualized target: {percent(forecast['target_xirr'])} · "
               f"Crossing probability threshold: {percent(forecast['crossing_probability_threshold'])}. "
               f"Forecasting begins only after {observation_sessions} complete post-entry observation session"
               f"{'s' if observation_sessions != 1 else ''}; entry sessions are excluded. "
               "This does not delay daily monitoring or authorize a trade. A probability threshold is not statistical confidence or a guaranteed exit date.")
    weighted = forecast.get("allocation_weighted_review")
    if weighted:
        st.info(
            "Allocation-weighted planning review: " + weighted["review_date"]
            + " · Follow-up: " + (weighted.get("review_followup_date") or "N/A")
            + " · Contributing target weight: "
            + percent(weighted.get("contributing_target_weight"))
        )
        st.caption(
            "Calculated once as Σ(security review date × probability by date × published target weight) "
            "÷ Σ(probability by date × published target weight), using only non-zero dated crossings. "
            "The result is mapped to the nearest valid consecutive two-day market window."
        )
    sort_by = st.selectbox(
        "Sort security estimates",
        REVIEW_SORT_OPTIONS,
        key=sort_key,
        help=("Probability is sorted numerically, not as formatted text. "
              "Missing probabilities and undated estimates remain last."),
    )
    reached = sort_security_estimates(reached, sort_by)
    st.table(pd.DataFrame([{"Security": r["ticker"],
                           "Target weight": percent(r.get("target_weight")),
                           "Estimated crossing": r["crossing_date"] or "Not reached in horizon",
                           "Review date": r.get("review_date") or "N/A",
                           "Review + 1": r.get("review_followup_date") or "N/A",
                           "Probability by date": percent(r["probability"])} for r in reached]))
    omitted = len(rows) - len(reached)
    if omitted:
        st.caption(f"{omitted} securities without a threshold crossing in the configured horizon are omitted.")
    st.caption("Research estimates unless validation passes. A crossing requests a review, not an automatic sale. Short-term XIRR can look large despite a small rupee gain.")


def review_promise_key(p):
    """Scope remembered dates to the policy that produced them."""
    policy_identity = (
        p.get("policy_digest")
        or p.get("policy", {}).get("policy_version")
        or p.get("policy_version")
        or p.get("forecast", {}).get("policy_hash")
        or "legacy"
    )
    return (
        "review_promise_" + p["publication_id"] + "_"
        + str(p.get("ack_epoch", 0)) + "_" + str(policy_identity)
    )


def review_display_state(p, today=None):
    """Return one mutually exclusive public review state."""
    decision = p.get("decision", {})
    candidate = decision.get("next_review")
    reasons = decision.get("reasons") or []
    planning_only = bool(p.get("planning_estimate"))
    today = today or datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()

    if planning_only:
        return {
            "state": "Planning estimate",
            "date_label": "Planning review",
            "date_value": candidate or "Not yet estimated",
            "due": False,
        }
    if decision.get("review_acknowledged"):
        return {
            "state": "Review completed",
            "date_label": "Next review" if candidate else "Monitoring",
            "date_value": candidate or "Continues automatically",
            "due": False,
        }
    if reasons:
        triggered_as_of = p.get("as_of") or str(p.get("checked_at") or "")[:10]
        return {
            "state": "Review now",
            "date_label": "Triggered as of",
            "date_value": triggered_as_of or "Latest assessment",
            "due": True,
        }
    if candidate and candidate <= today:
        return {
            "state": "Review now",
            "date_label": "Review due since",
            "date_value": candidate,
            "due": True,
        }
    return {
        "state": "No action now",
        "date_label": "Next review",
        "date_value": candidate or "Not yet validated",
        "due": False,
    }


def review_card_summary(payload):
    """Normalize a current-publication review payload for share-card use."""
    if not payload:
        return None
    display = review_display_state(payload)
    date_value = display.get("date_value")
    if not date_value or date_value in {
        "Not yet estimated", "Not yet validated", "Continues automatically",
    }:
        return None
    try:
        review_date = pd.Timestamp(date_value)
    except (TypeError, ValueError):
        return None
    if pd.isna(review_date):
        return None
    metrics = payload.get("metrics") or {}
    net_return = metrics.get("net_total_return")
    if net_return is not None:
        try:
            net_return = float(net_return)
            if not math.isfinite(net_return):
                net_return = None
        except (TypeError, ValueError):
            net_return = None
    return {
        "review_date": review_date.date().isoformat(),
        "review_state": (
            "Planning estimate"
            if payload.get("planning_estimate")
            else ("Review now" if display.get("due") else "Observed")
        ),
        "net_return": net_return,
    }


def _durable_review_card_summary(events, publication_id, now=None):
    """Use a recent observed assessment when a fresh preview is unavailable."""
    if not events or not publication_id:
        return None
    now = now or datetime.now(timezone.utc)
    baselines = [
        row for row in events
        if row.get("kind") == "BASELINE"
        and row.get("payload", {}).get("publication_id") == publication_id
    ]
    if not baselines:
        return None
    baseline = max(baselines, key=lambda row: row.get("seq", 0))["payload"]
    assessment = store.latest(events, "ASSESSMENT", baseline["baseline_id"])
    if not assessment:
        return None
    payload = assessment["payload"]
    checked_at = payload.get("checked_at")
    if not checked_at:
        return None
    checked = datetime.fromisoformat(checked_at)
    if checked.tzinfo is None:
        return None
    age_hours = (now - checked).total_seconds() / 3600
    if age_hours > 30 or age_hours < -1:
        return None
    return review_card_summary(payload)


def render_fresh_preview(p):
    d, f = p["decision"], p["forecast"]
    planning_only = bool(p.get("planning_estimate"))
    display = review_display_state(p)
    metrics = p.get("metrics") or {}
    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("Current state", display["state"])
            st.metric(display["date_label"], display["date_value"])
            if metrics:
                st.metric("Net return", percent(metrics.get("net_total_return")))
        if planning_only and p.get("observation_ready_at"):
            ready = datetime.fromisoformat(p["observation_ready_at"]).astimezone(
                ZoneInfo("Asia/Kolkata"))
            st.caption(
                f"Observed monitoring begins after {ready:%d %b %Y, %H:%M IST}. "
                "This planning date is not a sell instruction."
            )
        elif d.get("review_acknowledged"):
            acknowledged_at = d.get("acknowledged_at")
            acknowledged_label = ""
            if acknowledged_at:
                acknowledged = datetime.fromisoformat(acknowledged_at).astimezone(
                    ZoneInfo("Asia/Kolkata")
                )
                acknowledged_label = f" on {acknowledged:%d %b %Y, %H:%M IST}"
            st.caption(
                "Owner review completed" + acknowledged_label
                + ". No trade or portfolio change was recorded; monitoring continues."
            )
        elif d.get("reasons"):
            st.caption(
                "A configured review trigger is active. Review risk, taxes and "
                "trading costs before taking any action."
            )
        else:
            st.caption("No configured review trigger is active.")

    followup = f.get("next_common_review_session")
    if d.get("target_crossed_securities"):
        message = (
            "Net target already crossed: "
            + ", ".join(d["target_crossed_securities"])
            + "."
        )
        if d.get("review_acknowledged"):
            st.info(message + " This unchanged trigger was acknowledged; no sale was recorded.")
        else:
            st.warning(message + " Review costs and risk before selling.")

    timing = p.get("valuation_timing")
    timing_rows = timing.get("rows", []) if timing else []
    all_completed_closes = bool(timing_rows) and all(
        row.get("price_source") == "LATEST_COMPLETED_POST_ENTRY_CLOSE"
        for row in timing_rows
    )
    timing_label = (
        "Synchronized"
        if timing and timing.get("all_prices_synchronized") and all_completed_closes
        else "Mixed-time provisional"
    )

    with st.expander("Research and audit details", expanded=False):
        forecast_date = f.get("next_review")
        if planning_only and not forecast_date:
            forecast_date = d.get("next_review")
        if not display["due"] and forecast_date and followup:
            st.caption(
                f"Review window: {forecast_date}, then {followup} if follow-up is needed. "
                "The second date is the next verified common trading session for every market represented in this basket."
            )
        if p.get("provisional"):
            st.caption("Provisional: " + p["assumption"])
        if f.get("next_review") is None:
            st.caption("Target-crossing timing is not validated. Use the risk-review fallback, not the research date as a sell instruction.")
        st.caption(f"Prices through {p['as_of']} · Assessed {p['checked_at']} · Daily data, not live quotes; cache up to five minutes.")
        st.caption("Currency: NSE-listed holdings are priced in INR, including overseas ETFs. Their INR prices already reflect FX exposure; no second USD/INR conversion is applied.")
        if p.get("history_coverage"):
            h = p["history_coverage"]
            st.caption(f"Shared history: {h['start']} to {h['end']} · {h['usable_daily_returns']} valid daily returns · {len(h['missing_sessions'])} incomplete sessions excluded. No price filling.")
        st.download_button(
            "Download current review evidence",
            json.dumps(p, indent=2, default=str),
            file_name="public-model-review-evidence.json",
            mime="application/json",
        )

    if timing:
        with st.expander("Valuation timing · " + timing_label, expanded=False):
            st.caption("Audit detail: every price was observable by the assessment time. Frozen entry prices are never overwritten.")
            st.table(pd.DataFrame([{
                "Security": row["ticker"],
                "Price basis": row["price_source"].replace("_", " ").lower(),
                "Observed at": row["price_observed_at"],
                "Chronology valid": "Yes" if row["chronology_valid"] else "No",
            } for row in timing_rows]))

    with st.expander("Security target-crossing estimates", expanded=False):
        render_crossings(f, "fresh_security_crossings_sort")
        if p.get("metrics", {}).get("rows"):
            st.markdown("**Observed security returns**")
            st.table(pd.DataFrame([{
                "Security": row["ticker"],
                "Shares": row["shares"],
                "Price": f"₹{row['price']:,.2f}",
                "Net profit": f"₹{row['net_profit']:,.2f}",
                "Net XIRR": percent(row.get("xirr")),
            } for row in p["metrics"]["rows"]]))


def render_pending(row, now):
    if row is None:
        st.info("No monitoring record exists for this publication yet. Run the enabled model-review workflow; if it already ran, check that the page and workflow use the same basket and database.")
        return
    p = row["payload"]
    if p.get("reason") == "AWAITING_MARKET_ENTRY":
        captured, total = p.get("captured_entries"), p.get("total_entries")
        if p.get("wait_reason") == "FORECAST_OBSERVATION_WAIT":
            sessions = int(p.get("observation_sessions") or 1)
            st.info(
                f"Entry is verified. Waiting for every represented exchange to complete "
                f"{sessions} post-entry observation session{'s' if sessions != 1 else ''}; "
                "entry sessions do not count."
            )
        elif captured is not None and total:
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
    st.caption("Available from captured entries until the first complete basket assessment. One-share, fully costed security-only estimates; complete basket results will supersede them.")
    sort_by = st.selectbox(
        "Sort provisional security estimates",
        REVIEW_SORT_OPTIONS,
        key=f"partial_security_crossings_sort_{publication_id}",
        help=("Probability is sorted numerically, not as formatted text. "
              "Missing probabilities and undated estimates remain last."),
    )
    estimates = sort_security_estimates(list(latest.values()), sort_by)
    st.table(pd.DataFrame([{
        "Security": payload["ticker"],
        "Entry price": f"₹{payload['entry_price_inr']:,.2f}",
        "Target weight": percent(payload.get("target_weight")),
        "Data through": payload["as_of"],
        "Estimated target crossing": payload.get("estimated_crossing_date") or "Not reached in horizon",
        "Review date": payload.get("review_date") or "N/A",
        "Review + 1": payload.get("review_followup_date") or "N/A",
        "Probability by date": percent(payload.get("crossing_probability")),
    } for payload in estimates]))
    st.caption("Research estimates—not sell dates or recommendations. Fixed costs are conservative at one share; results may change when final basket quantities are known.")


def render_events(events, active_ids=None, now=None, latest_publication_id=None,
                  suppress_latest_failure=False, suppress_durable_preview=False):
    now = now or datetime.now(timezone.utc)
    # A fresh current-policy preview is the authoritative page result. Older
    # durable assessments remain in the append-only evidence ledger but must
    # not create a second, contradictory public review panel.
    if suppress_durable_preview:
        return
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
    provisional = store.latest(events, "PREVIEW", bid)
    if not is_current_preview(provisional):
        provisional = None
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
        if suppress_durable_preview:
            return
        if provisional and not suppress_durable_preview:
            render_fresh_preview(provisional["payload"])
        if provisional or suppress_durable_preview:
            st.info("Entry is verified. The review date above uses chronology-safe provisional marks; synchronized observed performance will replace it when every required close is available.")
        elif waiting and waiting["payload"].get("entry_frozen"):
            message = "Entry established from verified opening prices; awaiting the first completed global assessment session."
            ready_at = waiting["payload"].get("ready_at")
            if ready_at:
                ready = datetime.fromisoformat(ready_at).astimezone(ZoneInfo("Asia/Kolkata"))
                message += f" Earliest assessment: {ready:%d %b %Y %H:%M IST}."
            st.info(message)
        else:
            st.warning("Cannot assess: " + (failure["payload"]["reason"].replace("_", " ").lower() if failure else "awaiting first assessment"))
        if not provisional and not suppress_durable_preview:
            render_partial_security_reviews(events, baseline["publication_id"])
        return
    p = last["payload"]
    checked_at = heartbeat["payload"]["at"] if heartbeat else p["checked_at"]
    age = (now - datetime.fromisoformat(checked_at)).total_seconds() / 3600
    if age > 30 or age < -1:
        st.warning("Monitoring heartbeat is stale. Do not rely on the previous review date. Check the scheduled workflow.")
        return
    m, d = p["metrics"], p["decision"]
    status = d["status"].replace("_", " ").capitalize()
    review_required = d.get(
        "review_required",
        bool(d["reasons"]) and not d.get("review_acknowledged"),
    )
    if review_required:
        st.warning(status + " — review the estimates before acting. No trades have been submitted.")
    elif d.get("review_acknowledged"):
        st.info("Owner review completed. No trade or portfolio change was recorded; monitoring continues.")
    else:
        st.info("No configured trigger detected. This is not a guarantee against losses.")
    with st.container(horizontal=True):
        review_value = (
            "Now" if review_required
            else d.get("next_review") or (
                "Review completed" if d.get("review_acknowledged") else "Not validated"
            )
        )
        st.metric("Next suggested review", review_value)
        st.metric("Estimated net XIRR", percent(m["xirr"]))
        st.metric("Estimated net profit", f"₹{m['net_profit']:,.2f}")
        st.metric("Estimated exit proceeds", f"₹{m['net_proceeds']:,.2f}")
    forecast_review = p.get("forecast", {}).get("next_review")
    followup = p.get("forecast", {}).get("next_common_review_session")
    if not review_required and forecast_review and followup:
        st.caption(
            f"Review window: {forecast_review}, then {followup} if follow-up is needed. "
            "The second date is the next verified common trading session across the basket's represented markets."
        )
    checked = datetime.fromisoformat(checked_at).astimezone(ZoneInfo("Asia/Kolkata"))
    st.caption(f"{m['days_held']} days held · Absolute net return {percent(m['net_total_return'])} · Prices through {p['as_of']} · Checked {checked:%d %b %Y %H:%M IST}")
    st.caption("100% annualized XIRR does not mean your investment has doubled. A review date is not an optimal selling date.")
    sent = store.latest(events, "ALERT_SENT", bid)
    alert_failed = store.latest(events, "ALERT_FAILED", bid)
    if not sent or (alert_failed and alert_failed["seq"] > sent["seq"]) or (heartbeat and not heartbeat["payload"].get("alerts_enabled")):
        st.warning("Alert delivery has not been confirmed for this investment. Do not rely on background notifications yet.")
    with st.expander("Security returns and exit choices", expanded=False):
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
    with st.expander("Review-date evidence and assumptions", expanded=False):
        forecast = p["forecast"]
        render_crossings(forecast, f"assessment_security_crossings_sort_{bid}")
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
    fresh_preview_shown = False
    card_summary = None
    try:
        events = load_events(basket_id)
    except Exception:
        events = None
    if active_publications:
        try:
            with st.spinner("Assessing security targets from available history..."):
                preview = load_fresh_preview(basket_id, active_publications[0]["publication_id"])
            render_fresh_preview(preview)
            fresh_preview_shown = True
            card_summary = review_card_summary(preview)
        except Exception as exc:
            from .service import SAFE_ERRORS
            allowed = SAFE_ERRORS | {"POLICY_APPROVAL_REQUIRED", "TARIFF_REVIEW_REQUIRED",
                                     "UNSUPPORTED_TAX_OR_ACCOUNT_PROFILE", "NSE_CLASSIFICATION_REQUIRED",
                                     "INTEGER_POLICY_REQUIRED", "INVALID_CAPITAL",
                                     "INVALID_POLICY_ENTRY_QUOTE_INTERVAL",
                                     "INVALID_POLICY_PROFIT_REVIEW_RULE",
                                     "INVALID_POLICY_MINIMUM_FORECAST_REVIEW_SESSIONS"}
            code = str(exc) if isinstance(exc, ValueError) and str(exc) in allowed else {
                FileNotFoundError: "POLICY_FILE_MISSING",
                ModuleNotFoundError: "PREVIEW_MODULE_MISSING",
                StopIteration: "PUBLICATION_NOT_FOUND",
            }.get(type(exc), "PREVIEW_CHECK_FAILED")
            if code == "AWAITING_MARKET_ENTRY":
                fresh_waiting = True
                if getattr(exc, "wait_reason", None) == "FORECAST_OBSERVATION_WAIT":
                    sessions = int(getattr(exc, "observation_sessions", 1))
                    st.info(
                        f"Entry is verified. Waiting for every represented exchange to complete "
                        f"{sessions} post-entry observation session{'s' if sessions != 1 else ''}; "
                        "entry sessions do not count."
                    )
                else:
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
                latest_publication_id = active_publications[0]["publication_id"]
                if not has_durable_preview(events, latest_publication_id):
                    st.warning("Fresh historical review unavailable: " + code + ". No reliable fresh date is implied.")
            if code == "FOREIGN_REVIEW_COST_MODEL_REQUIRED":
                st.caption("This publication contains direct overseas listings. INR pricing is separate from tax classification. The review engine supports NSE delivery only; overseas brokerage, remittance charges and instrument-specific tax treatment must be integrated before net-XIRR review dates can be shown.")
            if code == "INSTRUMENT_CLASSIFICATION_REQUIRED":
                st.caption("Automatic classification could not safely identify one or more published instruments. The owner policy does not require a ticker list; retry after the NSE/Yahoo metadata source is available or add support for the unrecognized instrument category in code.")
    try:
        if events is None:
            events = load_events(basket_id)
        if card_summary is None and active_publications:
            card_summary = _durable_review_card_summary(
                events, active_publications[0]["publication_id"]
            )
        render_events(events, {p["publication_id"] for p in active_publications},
                      latest_publication_id=active_publications[0]["publication_id"] if active_publications else None,
                      suppress_latest_failure=fresh_waiting,
                      suppress_durable_preview=fresh_preview_shown or fresh_waiting)
    except Exception:
        st.warning("Model review inspection is unavailable. No reliable review date can be shown.")
    return card_summary
