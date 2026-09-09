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


def render_events(events, active_ids=None, now=None):
    now = now or datetime.now(timezone.utc)
    baseline_rows = [r for r in events if r["kind"] == "BASELINE" and
                     (active_ids is None or r["payload"]["publication_id"] in active_ids)]
    if not baseline_rows:
        st.info("Model monitoring is not ready. The owner must approve the cost/tax/risk policy and run the review workflow. No next review date is available.")
        failures = [r for r in events if r["kind"] == "FAILURE"]
        if failures:
            st.caption("Latest check: " + failures[-1]["payload"]["reason"].replace("_", " ").lower())
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
        st.caption("Tariff scope: normal funded resident-individual NSE delivery, Groww/Zerodha component-wise conservative envelope. No annual exemption or loss-offset credit. Not a SEBI certification or a personal tax calculation. MMI is context only.")
        st.json(p["mmi"])
        st.download_button("Download model review evidence (.json)", json.dumps({"baseline": baseline, "assessment": p,
                            "assessment_hash": last["event_hash"]}, indent=2),
                           file_name="public-model-review-evidence.json", mime="application/json")


def render_review_panel(basket_id, active_publications):
    st.subheader("Your next portfolio review")
    try:
        events = load_events(basket_id)
        render_events(events, {p["publication_id"] for p in active_publications})
    except Exception:
        st.warning("Model review inspection is unavailable. No reliable review date can be shown.")

