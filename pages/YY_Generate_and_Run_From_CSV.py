import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st
import yfinance as yf

import generate_public_basket_input as gen
import public_basket_optimizer_adapter as adapter
import public_basket_postgres as pb

st.set_page_config(page_title="Generate input from CSV + Run (robust)", layout="wide")
st.title("Generate frozen input from CSV, check & fix tickers, run optimizer, publish (one-off)")

st.warning(
    "This page will fetch live prices and can write authoritative records to the public ledger. "
    "Enable only temporarily and ensure your deployment is private or access-controlled."
)

# -------------------------
# Config
# -------------------------
YF_CHUNK_SIZE = st.session_state.get("yf_chunk_size", 30)
YF_MAX_RETRIES = st.session_state.get("yf_retries", 5)
YF_RETRY_BASE_DELAY = 1.0

# -------------------------
# DB / CSV checks
# -------------------------
database_url = pb.get_public_basket_database_url()
if not database_url:
    st.error("PUBLIC_BASKET_DATABASE_URL is not configured. Add it to Streamlit Secrets or env.")
    st.stop()

csv_path = Path("universal_portfolio_backup.csv")
if not csv_path.exists():
    st.error("universal_portfolio_backup.csv not found in the deployed repository. Upload or push it to the repo.")
    st.stop()

# scheduled date
scheduled_iso = st.text_input("Scheduled session date (YYYY-MM-DD)", value=str(date.today()))
try:
    scheduled_date = date.fromisoformat(scheduled_iso)
except Exception:
    st.error("Enter a valid ISO date (YYYY-MM-DD).")
    st.stop()

# -------------------------
# Helpers: batching + retries + normalizations
# -------------------------
def _download_chunk_with_retries(tickers: List[str], period: str = "5d"):
    last_exc = None
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            df = yf.download(tickers, period=period, group_by="ticker", threads=True, progress=False, timeout=30)
            return df
        except Exception as exc:
            last_exc = exc
            delay = YF_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
    raise last_exc

def normalize_ticker(t: str) -> str:
    # common fixes
    t = t.strip()
    # remove BOM or invisible chars
    t = t.replace("\ufeff", "").replace("\u200b", "")
    t = t.replace(" ", "").upper()
    # replace ampersand
    t = t.replace("&", "AND")
    # replace weird hyphen/emdash
    t = t.replace("—", "-").replace("–", "-")
    # ensure .NS suffix for NSE if absent and ticker isn't obviously non-equity
    if not t.endswith(".NS") and "." not in t:
        t = t + ".NS"
    return t

def batch_fetch_latest_prices(tickers: List[str]) -> (Dict[str, float], List[str]):
    prices = {}
    missing = []
    tickers = list(dict.fromkeys(tickers))  # dedupe while preserving order
    for i in range(0, len(tickers), YF_CHUNK_SIZE):
        chunk = tickers[i : i + YF_CHUNK_SIZE]
        try:
            df = _download_chunk_with_retries(chunk, period="5d")
        except Exception as exc:
            # if the whole chunk failed, treat all as missing for now
            st.warning(f"yf.download chunk failed: {exc}. Will try per-ticker fallback.")
            df = None

        for t in chunk:
            got = False
            # first attempt: read from df if available
            if df is not None:
                try:
                    # df shape differs when single vs multi ticker
                    if len(chunk) == 1:
                        series = df["Close"].dropna() if "Close" in df else pd.Series()
                    else:
                        if ("Close", t) in df.columns:
                            series = df[("Close", t)].dropna()
                        elif "Close" in df.columns and isinstance(df["Close"], pd.DataFrame) and t in df["Close"].columns:
                            series = df["Close"][t].dropna()
                        else:
                            series = pd.Series()
                    if not series.empty:
                        prices[t] = float(series.iloc[-1])
                        got = True
                except Exception:
                    got = False

            # fallback: try per-ticker single download with retries
            if not got:
                try:
                    s = _download_chunk_with_retries([t], period="5d")
                    if "Close" in s and not s["Close"].dropna().empty:
                        prices[t] = float(s["Close"].dropna().iloc[-1])
                        got = True
                    elif len(s.columns) > 0:
                        # single-ticker df may be single-column
                        series = None
                        if isinstance(s, pd.DataFrame):
                            # choose any numeric column
                            for col in s.columns:
                                if s[col].dropna().empty is False:
                                    series = s[col].dropna()
                                    break
                        if series is not None:
                            prices[t] = float(series.iloc[-1])
                            got = True
                except Exception:
                    got = False

            if not got:
                missing.append(t)
    return prices, missing

# -------------------------
# Step 1: generate payload (uses generator which will call our batch fetch)
# -------------------------
st.markdown("## Step 1 — Generate frozen input JSON from CSV (robust downloader)")

if st.button("Generate input from CSV (fetch prices; may take minutes)"):
    with st.spinner("Building frozen input (fetching prices & FX)..."):
        try:
            df = gen.load_holdings(str(csv_path))
            # normalize tickers in the holdings first to reduce missing items
            for r in df.to_dict("records"):
                pass
            # build payload but intercept price fetch: we will pre-fetch prices and give to builder if it supports it.
            # Generate payload normally and then attempt to fill prices; if generator fails, fall back to calling its fetch.
            payload = gen.build_payload(df, scheduled_session_date=scheduled_iso)
            # Now attempt to fetch live prices for the tickers referenced in payload
            tickers = [row["Yahoo Ticker"].strip() for row in payload["portfolio"]]
            # apply normalization heuristics first
            norm_map = {}
            normalized = []
            for t in tickers:
                nt = normalize_ticker(t)
                norm_map[nt] = t  # preserve original
                normalized.append(nt)
            prices, missing = batch_fetch_latest_prices(normalized)
            # Map prices keys back to original requested tickers if necessary
            # If any normalized ticker yields a price, accept it.
            # If missing non-empty, show to user below.
            if missing:
                st.warning(f"Could not fetch live prices for {len(missing)} tickers (after normalization). See list below.")
                st.session_state["generated_payload_tmp"] = payload
                st.session_state["yf_missing"] = missing
                st.session_state["yf_prices"] = prices
                st.info("Use the Ticker fixes section below to map or drop missing tickers, then re-run dry-run.")
            else:
                # write payload to temp file and store in session
                tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-", suffix=".json", delete=False)
                tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
                tmp.flush()
                tmp.close()
                st.success(f"Wrote frozen input to {tmp.name}")
                st.session_state["generated_input_path"] = tmp.name
                st.session_state["generated_payload"] = payload
                st.session_state["yf_prices"] = prices
        except Exception as exc:
            st.error(f"Failed to generate input: {exc}")
            st.exception(exc)

payload = st.session_state.get("generated_payload") or st.session_state.get("generated_payload_tmp")
generated_path = st.session_state.get("generated_input_path")
missing = st.session_state.get("yf_missing", [])

# show missing tickers UI
if payload is None:
    st.info("Generate the input from CSV first (above).")
    st.stop()

st.subheader("Detected tickers")
tickers_all = [row["Yahoo Ticker"].strip() for row in payload["portfolio"]]
st.write(f"{len(tickers_all):,} tickers in generated payload")

if missing:
    st.markdown("### Missing tickers (no price found after normalization)")
    st.write(missing[:300])
    st.info("You can: (A) let the page attempt automatic normalizations, (B) provide a replacement Yahoo ticker for each missing item, or (C) drop them and continue (unsafe if you want those holdings).")

    if st.button("Attempt additional automatic normalizations"):
        # try extra aggressive normalizations
        remap = {}
        new_try = []
        for t in missing:
            t0 = t
            # aggressive attempts
            candidates = [
                t.replace(".NS", ""),  # try no suffix
                t.replace(".NS", "") + ".NS",
                t.replace("-", ".NS"),
                t.replace("-", ""),
                t.replace("'", ""),
                t.replace(".", "").upper() + ".NS",
            ]
            tried = False
            for c in candidates:
                nt = normalize_ticker(c)
                try:
                    prices2, missing2 = batch_fetch_latest_prices([nt])
                    if nt in prices2:
                        remap[t0] = nt
                        tried = True
                        break
                except Exception:
                    continue
            if not tried:
                new_try.append(t0)
        # apply remaps
        if remap:
            st.success(f"Auto remapped {len(remap)} tickers")
            # replace in payload portfolio
            for i, row in enumerate(payload["portfolio"]):
                orig = row["Yahoo Ticker"].strip()
                # if user originally had an entry that normalized to a key in remap, change it
                for bad, good in remap.items():
                    if normalize_ticker(orig) == bad:
                        payload["portfolio"][i]["Yahoo Ticker"] = good
        # re-attempt batch fetch after remap
        normalized = [normalize_ticker(r["Yahoo Ticker"].strip()) for r in payload["portfolio"]]
        prices, missing = batch_fetch_latest_prices(normalized)
        st.session_state["yf_missing"] = missing
        st.session_state["generated_payload_tmp"] = payload
        st.session_state["yf_prices"] = prices
        st.experimental_rerun()

    st.markdown("#### Provide replacements for missing tickers")
    replacements = {}
    for t in missing:
        replacements[t] = st.text_input(f"Replacement Yahoo ticker for {t}", key=f"rep_{t}", placeholder=normalize_ticker(t))

    if st.button("Apply replacements and re-check"):
        # apply user mappings
        for bad, rep in replacements.items():
            if rep and rep.strip():
                for i, row in enumerate(payload["portfolio"]):
                    if normalize_ticker(row["Yahoo Ticker"].strip()) == bad:
                        payload["portfolio"][i]["Yahoo Ticker"] = rep.strip()
        # re-run batch fetch
        normalized = [normalize_ticker(r["Yahoo Ticker"].strip()) for r in payload["portfolio"]]
        prices, missing = batch_fetch_latest_prices(normalized)
        st.session_state["yf_missing"] = missing
        st.session_state["generated_payload_tmp"] = payload
        st.session_state["yf_prices"] = prices
        st.experimental_rerun()

    if st.button("Drop remaining missing tickers and continue (I accept data loss)"):
        kept = [r for r in payload["portfolio"] if normalize_ticker(r["Yahoo Ticker"].strip()) not in set(missing)]
        if not kept:
            st.error("Dropping all tickers would leave an empty portfolio. Aborted.")
        else:
            payload["portfolio"] = kept
            tmp2 = tempfile.NamedTemporaryFile(prefix="public-basket-input-", suffix=".json", delete=False)
            tmp2.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
            tmp2.flush()
            tmp2.close()
            st.session_state["generated_input_path"] = tmp2.name
            st.session_state["generated_payload"] = payload
            st.session_state.pop("yf_missing", None)
            st.success(f"Updated input written to {tmp2.name}. You can now run dry-run.")
            st.experimental_rerun()

# If no missing or user resolved, write payload to temp file if not already written
if not st.session_state.get("generated_input_path") and not st.session_state.get("yf_missing"):
    tmp = tempfile.NamedTemporaryFile(prefix="public-basket-input-", suffix=".json", delete=False)
    tmp.write(json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8"))
    tmp.flush()
    tmp.close()
    st.session_state["generated_input_path"] = tmp.name
    st.session_state["generated_payload"] = payload
    st.success(f"Wrote frozen input to {tmp.name}")

# ensure adapter reads the file
if st.session_state.get("generated_input_path"):
    os.environ[adapter.INPUT_PATH_ENV] = st.session_state["generated_input_path"]

# show gate
conn = pb.connect_public_basket_db(database_url)
try:
    gate = pb.rebalance_gate(conn, pb.DEFAULT_BASKET_ID, scheduled_date)
finally:
    conn.close()
st.subheader("Scheduler gate for scheduled_session_date")
st.write(gate)

force_any_day = False
if gate.get("status") not in ("DUE", "ALREADY_EVALUATED"):
    force_any_day = st.checkbox("Force single one-off run (override gate for this publish)", value=False)

# Dry-run
st.markdown("## Step 3 — Dry-run optimizer (no DB writes)")
if st.button("Run dry-run now"):
    try:
        if force_any_day:
            pb.PUBLIC_BASKET_ALLOW_ANY_DAY = True
        signal = adapter.build_public_signal(scheduled_session_date=scheduled_date)
    except Exception as exc:
        st.error("Dry-run failed: " + str(exc))
        st.exception(exc)
        st.stop()
    st.success(f"Dry-run succeeded — decision_status: {signal.get('decision_status')}")
    st.json({
        "strategy_version": signal.get("strategy_version"),
        "decision_status": signal.get("decision_status"),
        "optimizer_rows": len(signal.get("optimizer_output", {}).get("target_allocation", [])),
    })
    st.subheader("Preview signal_output (first 200 rows)")
    st.write(signal.get("signal_output"))
    st.session_state["last_signal"] = signal

# Publish
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

st.markdown("""
After you finish:
- Delete this page from the repository to return the app to read-only mode.
- If you published NO_CHANGE the ledger will record the evaluation but no NAVs were written. To show performance you need daily_nav rows later.
""")
