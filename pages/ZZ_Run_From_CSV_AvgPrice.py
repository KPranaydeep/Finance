import json
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import generate_public_basket_input as gen
import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb

st.set_page_config(page_title="Run optimizer from CSV (AvgPrice)", layout="wide")
st.title("Run optimizer from universal_portfolio_backup.csv using Average Price (one-off)")

st.warning(
    "This page uses the CSV 'Average Price' as the price source (NOT live market prices). "
    "Use only for testing or if Average Price is acceptable. Remove this page after use."
)

# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------
database_url = pb.get_public_basket_database_url()
if not database_url:
    st.error("PUBLIC_BASKET_DATABASE_URL is not configured. Add it to Streamlit Secrets or env.")
    st.stop()

csv_path = Path("universal_portfolio_backup.csv")
if not csv_path.exists():
    st.error("universal_portfolio_backup.csv not found in the deployed repo. Upload or push it first.")
    st.stop()

scheduled_iso = st.text_input("Scheduled session date (YYYY-MM-DD)", value=str(date.today()))
try:
    scheduled_date = date.fromisoformat(scheduled_iso)
except Exception:
    st.error("Enter a valid ISO date (YYYY-MM-DD).")
    st.stop()

# ---------------------------------------------------------------------
# Load CSV (handle BOM)
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)

df = load_csv(str(csv_path))

# Column detection
if "Yahoo Ticker" not in df.columns:
    st.error("CSV missing 'Yahoo Ticker' column. Please ensure the CSV has that column.")
    st.stop()

# Normalize tickers
df["Yahoo Ticker"] = df["Yahoo Ticker"].astype(str).str.strip()

# Use 'Average Price' as numeric, try several column names
avg_price_col = None
for candidate in ("Average Price", "Average_Price", "AveragePrice", "Average Price."):
    if candidate in df.columns:
        avg_price_col = candidate
        break

if avg_price_col is None:
    st.warning("CSV does not have 'Average Price' column — prices will be missing for many rows.")
    df["AveragePrice"] = np.nan
    avg_price_col = "AveragePrice"

df[avg_price_col] = pd.to_numeric(df[avg_price_col], errors="coerce")

st.subheader("CSV summary")
st.write(f"Rows: {len(df):,}")
st.write(df[["Yahoo Ticker", avg_price_col]].head(20))

missing_price_rows = df[df[avg_price_col].isna()]
if not missing_price_rows.empty:
    st.warning(f"{len(missing_price_rows):,} tickers have no Average Price in the CSV.")
    with st.expander("Missing price tickers (provide fallback price or drop)"):
        for i, row in missing_price_rows.iterrows():
            t = row["Yahoo Ticker"]
            fallback = st.text_input(f"Fallback price for {t} (leave blank to drop)", key=f"fp_{i}")
            if fallback and fallback.strip():
                try:
                    df.at[i, avg_price_col] = float(fallback.strip())
                except Exception:
                    st.error(f"Invalid number for ticker {t}: {fallback}")

# Option to drop rows with no price
if st.checkbox("Drop rows without a numeric price (recommended for quick run)", value=True):
    before = len(df)
    df = df[df[avg_price_col].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        st.info(f"Dropped {dropped} rows with no price")

if df.empty:
    st.error("No rows remain after dropping missing-price rows.")
    st.stop()

# ---------------------------------------------------------------------
# Build frozen input: construct minimal payload for adapter
# ---------------------------------------------------------------------
st.markdown("## Build frozen input JSON (using CSV prices)")

if st.button("Build frozen input from CSV prices"):
    try:
        # Build a minimal portfolio list matching expected input structure
        portfolio = []
        for _, r in df.iterrows():
            # If Quantity column is named differently, try to coerce common names
            qty = None
            for qname in ("Quantity", "quantity", "Qty", "QTY"):
                if qname in r.index and pd.notna(r.get(qname)):
                    try:
                        qty = float(r.get(qname))
                        break
                    except Exception:
                        qty = 0.0
            if qty is None:
                qty = 0.0

            portfolio.append(
                {
                    "symbol": r.get("Symbol") or "",
                    "yahoo_ticker": r["Yahoo Ticker"],
                    "quantity": float(qty),
                    "price": float(r[avg_price_col]),
                }
            )

        # Create a simple payload structure compatible with the adapter's frozen input reader.
        payload = {
            "market": "NSE_EQ",
            "scheduled_session_date": scheduled_iso,
            "market_data_cutoff": scheduled_iso,
            "input_created_at": str(pd.Timestamp.now(tz="UTC")),
            # adapter expects a "portfolio" list; columns used by adapter vary — this minimal set often suffices for dry-run.
            "portfolio": [
                {
                    "symbol": p["symbol"],
                    "yahoo_ticker": p["yahoo_ticker"],
                    "quantity": p["quantity"],
                    "price": p["price"],
                }
                for p in portfolio
            ],
        }

        # Write the payload to a controlled file and set env var so adapter.reading works
        tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-csv-", suffix=".json", delete=False)
        tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
        tmp.flush()
        tmp.close()
        st.success(f"Wrote frozen input to {tmp.name}")
        st.session_state["csv_generated_input_path"] = tmp.name
        st.session_state["csv_generated_payload"] = payload

        # ensure adapter reads the file
        os.environ[adapter.INPUT_PATH_ENV] = tmp.name

    except Exception as exc:
        st.error("Failed to build frozen input: " + str(exc))
        st.exception(exc)

# ---------------------------------------------------------------------
# Dry-run optimizer (no DB writes)
# ---------------------------------------------------------------------
if st.session_state.get("csv_generated_input_path"):
    st.markdown("## Dry-run optimizer (no DB writes)")

    if st.button("Run dry-run optimizer using CSV prices"):
        try:
            # build_public_signal reads the controlled input file via adapter.INPUT_PATH_ENV
            signal = adapter.build_public_signal(scheduled_session_date=scheduled_date)
        except Exception as exc:
            st.error("Dry-run failed: " + str(exc))
            st.exception(exc)
        else:
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
            st.session_state["csv_last_signal"] = signal

# ---------------------------------------------------------------------
# Publish (writes to Postgres)
# ---------------------------------------------------------------------
st.markdown("## Publish official signal to public ledger (writes to Postgres)")
if st.button("Publish signal now (writes to Postgres)"):
    signal = st.session_state.get("csv_last_signal")
    if not signal:
        st.error("Run Dry-run first.")
        st.stop()
    conn = pb.connect_public_basket_db(database_url)
    try:
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
    finally:
        conn.close()

st.markdown(
    """
Notes:
- This uses CSV Average Price (or fallbacks you provided). These prices may differ from live market prices.
- This is intended as a temporary/testing tool so you can run the optimizer without Yahoo data. Remove after use.
- The performance page displays NAV from daily_nav rows; publishing a signal does not create NAVs. You still need to run the NAV pipeline or insert daily_nav rows for the performance page to show charts.
"""
)
