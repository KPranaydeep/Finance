from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import yfinance as yf

import generate_public_basket_input as gen
import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb

st.set_page_config(page_title="Run optimizer from CSV (AvgPrice + health checks)", layout="wide")
st.title("Run optimizer from universal_portfolio_backup.csv — Avg Price + ticker health checks")

st.warning(
    "This page uses CSV Average Price and checks ticker history via yfinance. "
    "It can drop or remap tickers before running the optimizer. Use temporarily."
)

# --------------------------
# Settings (tweak if necessary)
# --------------------------
DEFAULT_YF_CHUNK = 40
DEFAULT_YF_RETRIES = 4
DEFAULT_YF_DELAY = 1.0

yf_chunk_size = st.number_input("yfinance chunk size (lower = less rate limit)", min_value=1, max_value=200, value=DEFAULT_YF_CHUNK)
yf_retries = st.number_input("yfinance retries per chunk", min_value=1, max_value=8, value=DEFAULT_YF_RETRIES)
yf_base_delay = st.number_input("yfinance base retry delay (seconds)", min_value=0.2, max_value=10.0, value=float(DEFAULT_YF_DELAY))

# --------------------------
# Basic checks
# --------------------------
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

# --------------------------
# Load CSV
# --------------------------
@st.cache_data(ttl=300)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)

df = load_csv(str(csv_path))

if "Yahoo Ticker" not in df.columns:
    st.error("CSV missing 'Yahoo Ticker' column.")
    st.stop()

df["Yahoo Ticker"] = df["Yahoo Ticker"].astype(str).str.strip()

# Prefer column naming detection for Average Price
avg_price_col = None
for candidate in ("Average Price", "Average_Price", "AveragePrice", "Average Price."):
    if candidate in df.columns:
        avg_price_col = candidate
        break
if avg_price_col is None:
    df["AveragePrice"] = np.nan
    avg_price_col = "AveragePrice"
df[avg_price_col] = pd.to_numeric(df[avg_price_col], errors="coerce")

st.subheader("CSV summary")
st.write(f"Rows: {len(df):,}")
st.write(df[["Yahoo Ticker", avg_price_col]].head(20))

# allow user to fill missing average prices or drop (same as before)
missing_price_rows = df[df[avg_price_col].isna()]
if not missing_price_rows.empty:
    st.warning(f"{len(missing_price_rows):,} tickers have no Average Price in the CSV.")
    with st.expander("Provide fallback price per ticker or leave blank to drop"):
        for i, row in missing_price_rows.iterrows():
            t = row["Yahoo Ticker"]
            fallback = st.text_input(f"Fallback price for {t}", key=f"fp_{i}")
            if fallback and fallback.strip():
                try:
                    df.at[i, avg_price_col] = float(fallback.strip())
                except Exception:
                    st.error(f"Invalid number for {t}: {fallback}")

if st.checkbox("Drop rows without a numeric price (recommended)", value=True):
    before = len(df)
    df = df[df[avg_price_col].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        st.info(f"Dropped {dropped} rows with no price")

if df.empty:
    st.error("No rows remain after dropping missing-price rows.")
    st.stop()

# --------------------------
# Helper: normalize ticker
# --------------------------
def normalize_ticker(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("\ufeff", "").replace("\u200b", "")
    t = t.upper().replace(" ", "")
    # if no suffix and no dot, append .NS (reasonable default for NSE)
    if "." not in t:
        t = t + ".NS"
    return t

# --------------------------
# Helper: yf download with retries
# --------------------------
def _download_with_retries(tickers: List[str], period: str = "180d") -> Tuple[pd.DataFrame, Exception]:
    last_exc = None
    for attempt in range(1, int(yf_retries) + 1):
        try:
            dfp = yf.download(tickers, period=period, group_by="ticker", threads=False, progress=False, timeout=30)
            return dfp, None
        except Exception as exc:
            last_exc = exc
            delay = yf_base_delay * (2 ** (attempt - 1))
            time.sleep(delay)
    return None, last_exc

# --------------------------
# Build payload helper (same structure adapter expects)
# --------------------------
def build_payload_from_df(df_in: pd.DataFrame, scheduled_iso: str):
    # extract qty
    def extract_qty(row):
        for qname in ("Quantity", "quantity", "Qty", "QTY"):
            if qname in row.index and pd.notna(row.get(qname)):
                try:
                    return float(row.get(qname))
                except Exception:
                    return 0.0
        return 0.0

    rows = []
    total_value = 0.0
    for _, r in df_in.iterrows():
        qty = extract_qty(r)
        price = float(r[avg_price_col])
        mv = qty * price
        total_value += mv
        rows.append({"Symbol": r.get("Symbol") or "",
                     "Yahoo Ticker": r["Yahoo Ticker"],
                     "Quantity": qty,
                     "Average Price": price,
                     "Currency": "INR",
                     "FX to INR": 1.0,
                     "Weight": None,
                     })
    if total_value > 0:
        for r in rows:
            r["Weight"] = float((r["Quantity"] * r["Average Price"]) / total_value)
    else:
        n = max(1, len(rows))
        for r in rows:
            r["Weight"] = float(1.0 / n)
    payload = {
        "market": "NSE_EQ",
        "scheduled_session_date": scheduled_iso,
        "market_data_cutoff": scheduled_iso,
        "input_created_at": str(pd.Timestamp.now(tz="UTC")),
        "portfolio": rows,
    }
    return payload

# --------------------------
# Step: Build initial payload from CSV (button)
# --------------------------
if st.button("Build frozen input from CSV prices"):
    try:
        payload = build_payload_from_df(df, scheduled_iso)
        tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-csv-", suffix=".json", delete=False)
        tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
        tmp.flush()
        tmp.close()
        st.success(f"Wrote frozen input to {tmp.name}")
        st.session_state["csv_generated_input_path"] = tmp.name
        st.session_state["csv_generated_payload"] = payload
        os.environ[adapter.INPUT_PATH_ENV] = tmp.name
    except Exception as exc:
        st.error("Failed to build frozen input: " + str(exc))
        st.exception(exc)

payload = st.session_state.get("csv_generated_payload")
generated_path = st.session_state.get("csv_generated_input_path")

# --------------------------
# Ticker history check
# --------------------------
st.markdown("## Check ticker history (yfinance)")
if payload:
    tickers = [normalize_ticker(r["Yahoo Ticker"]) for r in payload["portfolio"]]
    st.write(f"{len(tickers):,} tickers in payload (normalized)")

    if st.button("Check ticker history"):
        # batch and collect missing
        missing = []
        info = []
        total = len(tickers)
        for i in range(0, total, int(yf_chunk_size)):
            chunk = tickers[i : i + int(yf_chunk_size)]
            st.info(f"Downloading chunk {i//int(yf_chunk_size)+1} / {((total-1)//int(yf_chunk_size))+1} ({len(chunk)} tickers)...")
            dfp, exc = _download_with_retries(chunk, period="180d")
            for t in chunk:
                got = False
                if dfp is not None:
                    try:
                        # handle multi / single ticker shapes
                        if len(chunk) == 1:
                            series = dfp["Close"].dropna() if "Close" in dfp else pd.Series()
                        else:
                            if ("Close", t) in dfp.columns:
                                series = dfp[("Close", t)].dropna()
                            elif "Close" in dfp.columns and isinstance(dfp["Close"], pd.DataFrame) and t in dfp["Close"].columns:
                                series = dfp["Close"][t].dropna()
                            else:
                                series = pd.Series()
                        if not series.empty:
                            info.append((t, int(series.shape[0])))
                            got = True
                    except Exception:
                        got = False
                if not got:
                    missing.append(t)
                    info.append((t, 0))
            st.progress(min(100, int((i + len(chunk)) / max(1, total) * 100)))
        st.session_state["yf_check_info"] = info
        st.session_state["yf_missing"] = missing
        # Instead of experimental_rerun (may be unavailable), toggle a session_state flag and stop
        st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
        st.stop()

    info = st.session_state.get("yf_check_info")
    missing = st.session_state.get("yf_missing", [])
    if info:
        df_info = pd.DataFrame(info, columns=["ticker", "recent_close_days"]).sort_values("recent_close_days", ascending=False)
        st.dataframe(df_info)
    if missing:
        st.warning(f"{len(missing)} tickers appear to have no history: show first 200 ->")
        st.write(missing[:200])

        # Offer replacements for missing tickers
        st.markdown("### Provide replacements or drop missing tickers")
        replacements = {}
        for t in missing[:200]:
            replacements[t] = st.text_input(f"Replacement for {t}", key=f"rep_{t}", placeholder=t)

        if st.button("Apply replacements and re-check"):
            # apply replacements in payload
            for bad, rep in replacements.items():
                if rep and rep.strip():
                    for row in payload["portfolio"]:
                        if normalize_ticker(row["Yahoo Ticker"]) == bad:
                            row["Yahoo Ticker"] = rep.strip()
            # rewrite payload and recheck next run
            tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-csv-", suffix=".json", delete=False)
            tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
            tmp.flush()
            tmp.close()
            st.session_state["csv_generated_input_path"] = tmp.name
            st.session_state["csv_generated_payload"] = payload
            os.environ[adapter.INPUT_PATH_ENV] = tmp.name
            st.success("Applied replacements; please press 'Check ticker history' again.")
            st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
            st.stop()

        if st.button("Drop missing tickers and rebuild payload (I accept data loss)"):
            kept = [r for r in payload["portfolio"] if normalize_ticker(r["Yahoo Ticker"]) not in set(missing)]
            if not kept:
                st.error("Dropping all tickers would leave an empty portfolio. Aborted.")
            else:
                payload["portfolio"] = kept
                tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-csv-", suffix=".json", delete=False)
                tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
                tmp.flush()
                tmp.close()
                st.session_state["csv_generated_input_path"] = tmp.name
                st.session_state["csv_generated_payload"] = payload
                os.environ[adapter.INPUT_PATH_ENV] = tmp.name
                st.success(f"Rebuilt input without missing tickers: {tmp.name}")
                st.session_state["_rerun_toggle"] = not st.session_state.get("_rerun_toggle", False)
                st.stop()

# --------------------------
# Show current payload path and allow dry-run
# --------------------------
if generated_path:
    st.markdown(f"Generated frozen input: `{generated_path}`")
    if st.button("Run dry-run optimizer using CSV prices"):
        try:
            signal = adapter.build_public_signal(scheduled_session_date=scheduled_date)
        except Exception as exc:
            st.error("Dry-run failed: " + str(exc))
            st.exception(exc)
        else:
            st.success(f"Dry-run succeeded — decision_status: {signal.get('decision_status')}")
            st.json({
                "strategy_version": signal.get("strategy_version"),
                "decision_status": signal.get("decision_status"),
                "optimizer_rows": len(signal.get("optimizer_output", {}).get("target_allocation", [])),
            })
            st.subheader("Preview signal_output (first 200 rows)")
            st.write(signal.get("signal_output"))
            st.session_state["csv_last_signal"] = signal

# --------------------------
# Publish
# --------------------------
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

st.markdown("""
Notes:
- Many CSV tickers are non-standard names or ETFs with alternate Yahoo tickers. Best practice: fix the Yahoo Ticker column upstream.
- Use "Apply replacements" to correct tickers, or "Drop missing tickers" to proceed with a reduced universe.
- If you hit yfinance rate limits, reduce chunk size and try again after a pause.
""")
