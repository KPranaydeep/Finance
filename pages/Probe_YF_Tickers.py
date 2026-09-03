# pages/Probe_YF_Tickers.py
import time
import io
from typing import List
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Probe tickers (yfinance)", layout="wide")
st.title("Probe tickers with yfinance (run on Streamlit Cloud)")

st.markdown(
    "Use this temporary page to test which tickers have usable recent history from Yahoo Finance. "
    "Paste a list of tickers or upload a CSV (column 'Ticker' or 'Yahoo Ticker' or first column)."
)

st.sidebar.header("Probe settings")
delay = st.sidebar.number_input("Delay between probes (seconds)", min_value=0.1, max_value=5.0, value=0.6, step=0.1)
retries = st.sidebar.number_input("Retries per ticker on failure", min_value=0, max_value=8, value=2, step=1)
timeout = st.sidebar.number_input("yf.download timeout (seconds)", min_value=5, max_value=120, value=30, step=1)

uploaded = st.file_uploader("Upload CSV of tickers (optional)", type=["csv"], key="probe_upload")
paste = st.text_area("Or paste tickers (one per line)", height=200, key="probe_paste")

run_btn = st.button("Run probe", type="primary")

def extract_tickers_from_upload(fh) -> List[str]:
    try:
        df = pd.read_csv(fh, dtype=str)
    except Exception:
        # fallback: read lines
        fh.seek(0)
        lines = fh.read().decode("utf-8-sig").splitlines()
        return [l.strip() for l in lines if l.strip()]
    for col in ("Ticker", "Yahoo Ticker"):
        if col in df.columns:
            return df[col].dropna().astype(str).str.strip().tolist()
    # otherwise take first column
    first = df.columns[0]
    return df[first].dropna().astype(str).str.strip().tolist()

def probe_one(ticker: str, timeout: int = 30):
    last_exc = None
    for attempt in range(1, int(retries) + 2):
        try:
            df = yf.download(ticker, period="10d", threads=False, progress=False, timeout=timeout, auto_adjust=True)
            if df is None or df.empty:
                return False, None
            numeric = df.apply(pd.to_numeric, errors="coerce")
            ok = bool(numeric.notna().any().any())
            return ok, None
        except Exception as exc:
            last_exc = exc
            # small backoff between retries
            time.sleep(0.5)
            continue
    return False, str(last_exc) if last_exc is not None else "unknown"

if run_btn:
    tickers = []
    if uploaded is not None:
        tickers = extract_tickers_from_upload(uploaded)
    elif paste and paste.strip():
        tickers = [line.strip() for line in paste.splitlines() if line.strip()]
    else:
        st.warning("Provide tickers by upload or paste first.")
        st.stop()

    tickers = list(dict.fromkeys([t.strip() for t in tickers if t and t.strip()]))
    if not tickers:
        st.warning("No tickers found in input.")
        st.stop()

    st.info(f"Probing {len(tickers)} tickers with delay={delay}s and retries={retries}...")
    progress = st.progress(0)
    status_box = st.empty()

    good = []
    bad = []
    rows = []
    for i, t in enumerate(tickers, start=1):
        status_box.text(f"Probing {i}/{len(tickers)}: {t}")
        ok, err = probe_one(t, timeout=int(timeout))
        if ok:
            good.append(t)
            rows.append({"Ticker": t, "Status": "OK", "Error": ""})
        else:
            bad.append(t)
            rows.append({"Ticker": t, "Status": "NO", "Error": err or ""})
        progress.progress(int(i / len(tickers) * 100))
        time.sleep(float(delay))

    df_out = pd.DataFrame(rows)
    st.success(f"Done. OK: {len(good)}  |  NO: {len(bad)}")
    st.dataframe(df_out, use_container_width=True)

    # build CSV bytes
    buf_good = io.StringIO()
    pd.DataFrame({"Ticker": good}).to_csv(buf_good, index=False)
    buf_bad = io.StringIO()
    pd.DataFrame({"Ticker": bad}).to_csv(buf_bad, index=False)
    st.download_button("Download good_tickers.csv", data=buf_good.getvalue().encode("utf-8"), file_name="good_tickers.csv", mime="text/csv")
    st.download_button("Download bad_tickers.csv", data=buf_bad.getvalue().encode("utf-8"), file_name="bad_tickers.csv", mime="text/csv")

    # store summary in session_state so you can revisit
    st.session_state["probe_last_results"] = df_out.to_dict(orient="records")

else:
    prev = st.session_state.get("probe_last_results")
    if prev:
        st.info("Previous run results (from session state)")
        st.dataframe(pd.DataFrame(prev), use_container_width=True)
