import json
from datetime import date
from pathlib import Path

import streamlit as st

import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb

st.set_page_config(page_title="Manual: Publish Weekly Signal", layout="wide")
st.title("Manual: Build & Publish Weekly Public Signal (one-off)")

st.warning(
    "This page can write authoritative records to the public ledger. "
    "Enable only temporarily and ensure your Streamlit deployment is private or access-controlled."
)

# ---------------------------------------------------------------------
# Database config
# ---------------------------------------------------------------------
database_url = pb.get_public_basket_database_url()
if not database_url:
    st.error(
        "PUBLIC_BASKET_DATABASE_URL is not configured for this app. "
        "Set it in Streamlit Secrets or environment variables before using this page."
    )
    st.stop()

# ---------------------------------------------------------------------
# Input selection: either upload or use repo file
# ---------------------------------------------------------------------
st.subheader("Select frozen input JSON")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload the operator-prepared JSON input (recommended)", type="json")
    use_repo_path = st.checkbox("Use the repository sample input file: sample_public_basket_input.json", False)

input_payload = None
input_path = None

if uploaded is not None:
    try:
        input_payload = json.load(uploaded)
    except Exception as exc:
        st.error(f"Could not parse uploaded JSON: {exc}")
        st.stop()
else:
    if use_repo_path:
        repo_file = Path("sample_public_basket_input.json")
        if repo_file.exists():
            try:
                input_payload = json.loads(repo_file.read_text(encoding="utf-8"))
                input_path = str(repo_file)
                st.info(f"Loaded repository input: {repo_file}")
            except Exception as exc:
                st.error(f"Could not read sample_public_basket_input.json: {exc}")
                st.stop()
        else:
            st.error("sample_public_basket_input.json not found in the deployed app repository.")
            st.stop()
    else:
        st.info("Upload a frozen input JSON (or check the repo file checkbox) to proceed.")
        st.stop()

# ---------------------------------------------------------------------
# Show basic info from input
# ---------------------------------------------------------------------
st.subheader("Input preview")
scheduled_iso = input_payload.get("scheduled_session_date")
st.write({"scheduled_session_date": scheduled_iso, "positions": len(input_payload.get("portfolio", []))})

try:
    scheduled_date = date.fromisoformat(str(scheduled_iso))
except Exception:
    st.error("The input's scheduled_session_date is missing or invalid ISO date.")
    st.stop()

# ---------------------------------------------------------------------
# Show rebalance gate
# ---------------------------------------------------------------------
conn = pb.connect_public_basket_db(database_url)
try:
    gate = pb.rebalance_gate(conn, pb.DEFAULT_BASKET_ID, scheduled_date)
finally:
    conn.close()

st.subheader("Scheduler gate")
st.write(gate)

force_any_day = False
if gate.get("status") not in ("DUE", "ALREADY_EVALUATED"):
    st.warning(
        "The scheduler gate is not DUE for the scheduled_session_date above. "
        "You can force a one-time run by checking the box below (temporarily enables single-run-any-day)."
    )
    force_any_day = st.checkbox("Force single one-off run (override gate for this publish)", value=False)

# ---------------------------------------------------------------------
# Dry-run: build the signal with the adapter (no DB writes)
# ---------------------------------------------------------------------
st.subheader("Dry-run optimizer (no DB writes)")

try:
    # If forcing, temporarily set the module-level flag so rebalance_gate inside record_weekly_signal will accept today's run.
    if force_any_day:
        pb.PUBLIC_BASKET_ALLOW_ANY_DAY = True

    signal = adapter.build_public_signal(scheduled_session_date=scheduled_date)
except Exception as exc:
    st.error("Dry-run failed: " + str(exc))
    st.exception(exc)
    st.stop()

st.success(f"Dry-run succeeded — decision_status: {signal.get('decision_status')}")
st.write("Optimizer summary")
st.json(
    {
        "strategy_version": signal.get("strategy_version"),
        "decision_status": signal.get("decision_status"),
        "optimizer_rows": len(signal.get("optimizer_output", {}).get("target_allocation", [])),
    }
)

st.subheader("Preview signal_output (first 200 rows)")
st.write(signal.get("signal_output"))

# ---------------------------------------------------------------------
# Confirm & publish
# ---------------------------------------------------------------------
st.subheader("Publish official signal to public ledger")
publish = st.button("Publish this signal (writes to Postgres)")

if publish:
    conn = pb.connect_public_basket_db(database_url)
    try:
        try:
            # If forcing, ensure module flag is set (again) before calling record_weekly_signal
            if force_any_day:
                pb.PUBLIC_BASKET_ALLOW_ANY_DAY = True

            signal_run_id = pb.record_weekly_signal(
                conn=conn,
                basket_id=pb.DEFAULT_BASKET_ID,
                today=scheduled_date,
                strategy_version=signal["strategy_version"],
                git_commit_sha=None,
                settings=signal["settings"],
                portfolio_before=signal["portfolio_before"],
                optimizer_output=signal["optimizer_output"],
                signal_output=signal["signal_output"],
                decision_status=signal["decision_status"],
            )
        except Exception as exc:
            st.error("Publishing failed: " + str(exc))
            st.exception(exc)
            raise

        st.success(f"Published signal_run_id={signal_run_id}")
        st.info("A weekly_rebalance_cycles row is created; subsequent runs this week will be ALREADY_EVALUATED.")
    finally:
        conn.close()

# ---------------------------------------------------------------------
# Cleanup / guidance
# ---------------------------------------------------------------------
st.markdown(
    """
After you publish successfully:
- Remove this page from the repository to return the app to read-only mode.
- Verify the performance page shows data (it reads daily_nav). If daily_nav is still empty you may need to run your NAV calculation pipeline that inserts daily_nav rows.
"""
)
