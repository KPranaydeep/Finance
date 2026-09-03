# pages/YY_Generate_and_Run_From_CSV.py
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

# Re-use the project's generator + adapter + postgres modules
import generate_public_basket_input as gen
import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb

st.set_page_config(page_title="Generate input from CSV + Run", layout="wide")
st.title("Generate frozen input from CSV, check tickers, run optimizer, and publish (one-off)")

st.warning(
    "This page will fetch live prices and can write authoritative records to the public ledger. "
    "Enable only temporarily and ensure your deployment is private or access-controlled."
)

# DB check
database_url = pb.get_public_basket_database_url()
if not database_url:
    st.error("PUBLIC_BASKET_DATABASE_URL is not configured for this app. Set it in Streamlit Secrets or environment.")
    st.stop()

# CSV existence
csv_path = Path("universal_portfolio_backup.csv")
if not csv_path.exists():
    st.error("universal_portfolio_backup.csv not found in the deployed repository. Upload it or push it to the repo.")
    st.stop()

# Scheduled session date input (must match gate unless you force)
scheduled_iso = st.text_input("Scheduled session date (YYYY-MM-DD)", value=str(date.today()))
try:
    scheduled_date = date.fromisoformat(scheduled_iso)
except Exception:
    st.error("Enter a valid ISO date (YYYY-MM-DD).")
    st.stop()

# Step 1: generate payload
st.markdown("## Step 1 — Generate frozen input JSON from CSV")
if st.button("Generate input from CSV (fetches live prices & FX)"):
    with st.spinner("Building frozen input (fetching prices & FX)..."):
        try:
            df = gen.load_holdings(str(csv_path))
            payload = gen.build_payload(df, scheduled_session_date=scheduled_iso)
        except Exception as exc:
            st.error("Failed to generate input: " + str(exc))
            st.exception(exc)
            st.stop()

        # Save to a temp file and show snippet
        tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-", suffix=".json", delete=False)
        tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
        tmp.flush()
        tmp.close()
        st.success(f"Wrote frozen input to {tmp.name}")
        st.code(json.dumps({k: payload[k] for k in ("scheduled_session_date", "market_data_cutoff", "input_created_at", "portfolio")}, indent=2)[:4000])

        # Expose the path and payload to the page state
        st.session_state["generated_input_path"] = tmp.name
        st.session_state["generated_payload"] = payload

generated_path = st.session_state.get("generated_input_path")
payload = st.session_state.get("generated_payload")

if not generated_path or not payload:
    st.info("Generate the input from CSV first (above).")
    st.stop()

# Ensure adapter will read the file
os.environ[adapter.INPUT_PATH_ENV] = generated_path

# Show rebalance gate for the scheduled date
conn = pb.connect_public_basket_db(database_url)
try:
    gate = pb.rebalance_gate(conn, pb.DEFAULT_BASKET_ID, scheduled_date)
finally:
    conn.close()

st.subheader("Scheduler gate for scheduled_session_date")
st.write(gate)

force_any_day = False
if gate.get("status") not in ("DUE", "ALREADY_EVALUATED"):
    st.warning("The gate is not DUE. If you still want to run once this week, check Force one-off below.")
    force_any_day = st.checkbox("Force single one-off run (override gate for this publish)", value=False)

# Step 2: ticker health check
st.markdown("## Step 2 — Check tickers' price history")
tickers = [row["Yahoo Ticker"].strip() for row in payload["portfolio"]]

st.write(f"{len(tickers):,} tickers in generated payload")
if st.button("Check price history for these tickers (yfinance)"):
    missing = []
    insufficient = []
    info_rows = []

    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers, 1):
        try:
            hist = yf.Ticker(t).history(period="180d", auto_adjust=False)
            non_null_days = hist["Close"].dropna().shape[0] if not hist.empty else 0
        except Exception:
            non_null_days = 0
        info_rows.append((t, non_null_days))
        if non_null_days == 0:
            missing.append(t)
        elif non_null_days < 10:
            insufficient.append(t)
        progress.progress(int(i / total * 100))

    df_info = pd.DataFrame(info_rows, columns=["ticker", "recent_close_days"])
    st.dataframe(df_info)

    if missing:
        st.error(f"{len(missing)} tickers have no history: {missing[:50]}")
    else:
        st.success("No completely missing tickers found.")

    if insufficient:
        st.warning(f"{len(insufficient)} tickers have very little history (fewer than 10 close days): {insufficient[:50]}")

    st.session_state["missing_tickers"] = missing
    st.session_state["insufficient_tickers"] = insufficient

# If missing tickers exist, let user remove them
missing = st.session_state.get("missing_tickers", [])
if missing:
    st.markdown("### Missing tickers detected")
    st.write(missing)
    if st.button("Remove missing tickers and update input, then run dry-run"):
        new_portfolio = [r for r in payload["portfolio"] if r["Yahoo Ticker"].strip() not in set(missing)]
        if not new_portfolio:
            st.error("Removing missing tickers would leave an empty portfolio. Aborting.")
            st.stop()
        payload["portfolio"] = new_portfolio

        # write updated payload to temp file and set env
        tmp2 = tempfile.NamedTemporaryFile(prefix="public-basket-input-", suffix=".json", delete=False)
        tmp2.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
        tmp2.flush()
        tmp2.close()
        st.session_state["generated_input_path"] = tmp2.name
        st.session_state["generated_payload"] = payload
        os.environ[adapter.INPUT_PATH_ENV] = tmp2.name
        st.success(f"Updated input written to {tmp2.name}. Now running dry-run...")

        # run dry-run
        try:
            if force_any_day:
                pb.PUBLIC_BASKET_ALLOW_ANY_DAY = True
            signal = adapter.build_public_signal(scheduled_session_date=scheduled_date)
        except Exception as exc:
            st.error("Dry-run failed after removing missing tickers: " + str(exc))
            st.exception(exc)
            st.stop()

        st.success(f"Dry-run succeeded — decision_status: {signal.get('decision_status')}")
        st.json(
            {
                "strategy_version": signal.get("strategy_version"),
                "decision_status": signal.get("decision_status"),
                "optimizer_rows": len(signal.get("optimizer_output", {}).get("target_allocation", [])),
            }
        )
        st.subheader("Preview signal_output (first 200 rows)")
        st.write(signal.get("signal_output"))
        st.session_state["last_signal"] = signal

# Step 3: run dry-run (if user prefers to run without removing tickers)
st.markdown("## Step 3 — Dry-run optimizer (no DB writes)")
if st.button("Run dry-run now (use current input path)"):
    try:
        if force_any_day:
            pb.PUBLIC_BASKET_ALLOW_ANY_DAY = True
        signal = adapter.build_public_signal(scheduled_session_date=scheduled_date)
    except Exception as exc:
        st.error("Dry-run failed: " + str(exc))
        st.exception(exc)
        st.stop()

    st.success(f"Dry-run succeeded — decision_status: {signal.get('decision_status')}")
    st.json(
        {
            "strategy_version": signal.get("strategy_version"),
            "decision_status": signal.get("decision_status"),
            "optimizer_rows": len(signal.get("optimizer_output", {}).get("target_allocation", [])),
        }
    )
    st.subheader("Preview signal_output (first 200 rows)")
    st.write(signal.get("signal_output"))
    st.session_state["last_signal"] = signal

# Step 4: Publish
st.markdown("## Step 4 — Publish official signal to public ledger (writes to Postgres)")
if st.button("Publish signal now"):
    signal = st.session_state.get("last_signal")
    if not signal:
        st.error("Run Dry-run first.")
        st.stop()

    conn = pb.connect_public_basket_db(database_url)
    try:
        if force_any_day:
            pb.PUBLIC_BASKET_ALLOW_ANY_DAY = True
        try:
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

st.markdown(
    """
After you finish:
- Delete this page from the repository to return the app to read-only mode.
- If you published NO_CHANGE the ledger will have recorded the evaluation but no NAVs were written. To show performance you need daily_nav rows (run your NAV pipeline or insert test rows).
"""
)
