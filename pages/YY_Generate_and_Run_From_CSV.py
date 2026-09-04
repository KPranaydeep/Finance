from __future__ import annotations

import io
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

import generate_public_basket_input as generator

DEFAULT_CSV = Path("universal_portfolio_backup.csv")
PAGE_VERSION = "event-input-generator-r8"
MINIMUM_PRICE_COVERAGE = 0.80

st.set_page_config(page_title="Prepare Public Basket Input", page_icon="📦", layout="wide")
st.title("📦 Prepare Public Basket Input")
st.caption(f"{PAGE_VERSION} · Two-stage read-only preparation · No PostgreSQL writes")
st.info("Stage 1 shows liquidity exclusions. Stage 2 creates JSON after confirmation. The optimizer runs later in Public Basket Publisher.")

uploaded_csv = st.file_uploader("Holdings CSV (optional)", type=["csv"])
source_label = uploaded_csv.name if uploaded_csv is not None else str(DEFAULT_CSV)
st.write("Selected source:", f"`{source_label}`")
if uploaded_csv is None and not DEFAULT_CSV.is_file():
    st.error("universal_portfolio_backup.csv is missing. Upload a holdings CSV above.")
    st.stop()

exclude_percent = st.slider(
    "Exclude the least-liquid positions before optimization", 0, 90, 80, 5,
    format="%d%%", help="Ranks positions by 3-month median daily traded value in INR."
)
max_optimizer_positions = st.number_input(
    "Maximum positions sent to the optimizer",
    min_value=10,
    max_value=600,
    value=400,
    step=10,
    help="If the percentage filter retains more positions, only the most liquid are kept.",
)
zero_investment_start = st.checkbox(
    "This public basket starts with zero investment",
    value=True,
    help=(
        "The personal CSV supplies the analysis universe only. "
        "Public-basket quantities are written as zero."
    ),
)

st.subheader("Stage 1 — Build read-only liquidity preview")
if st.button("Fetch data and build preview", type="primary"):
    try:
        st.session_state.pop("prepared_public_basket_json", None)
        if uploaded_csv is not None:
            frame = generator.validate_holdings(pd.read_csv(io.BytesIO(uploaded_csv.getvalue())))
        else:
            frame = generator.load_holdings(str(DEFAULT_CSV))

        tickers = frame["Yahoo Ticker"].tolist()
        bar = st.progress(0, text="Fetching current prices…")
        prices, price_missing = generator.fetch_latest_prices_with_missing(
            tickers,
            progress=lambda done, total: bar.progress(done / total, text=f"Price batch {done} of {total}"),
        )
        bar.empty()
        price_coverage = len(prices) / len(set(tickers)) if tickers else 0.0
        if price_coverage < MINIMUM_PRICE_COVERAGE:
            raise RuntimeError(f"Price coverage is {price_coverage:.1%}; at least {MINIMUM_PRICE_COVERAGE:.0%} is required.")

        priced = frame[frame["Yahoo Ticker"].isin(prices)].copy()
        priced.attrs = frame.attrs.copy()
        fx_rates = generator.fetch_fx_rates(set(priced["Currency"]))
        bar = st.progress(0, text="Measuring liquidity…")
        liquidity, liquidity_missing = generator.fetch_liquidity_metrics(
            priced["Yahoo Ticker"].tolist(),
            progress=lambda done, total: bar.progress(done / total, text=f"Liquidity batch {done} of {total}"),
        )
        bar.empty()

        ranked = priced[priced["Yahoo Ticker"].isin(liquidity)].copy()
        ranked["Latest Price"] = ranked["Yahoo Ticker"].map(prices)
        ranked["Median Daily Traded Value (INR)"] = ranked.apply(
            lambda row: float(liquidity[row["Yahoo Ticker"]]["median_daily_traded_value"])
            * fx_rates[str(row["Currency"]).upper()], axis=1
        )
        ranked["Observed Sessions"] = ranked["Yahoo Ticker"].map(
            lambda ticker: int(liquidity[ticker]["observed_sessions"])
        )
        ranked = ranked.sort_values("Median Daily Traded Value (INR)")
        remove_count = min(math.floor(len(ranked) * exclude_percent / 100), max(len(ranked) - 1, 0))
        percent_excluded = ranked.iloc[:remove_count].copy()
        after_percent = ranked.iloc[remove_count:].copy()
        cap_remove_count = max(len(after_percent) - int(max_optimizer_positions), 0)
        cap_excluded = after_percent.iloc[:cap_remove_count].copy()
        retained = after_percent.iloc[cap_remove_count:].copy()
        retained.attrs = frame.attrs.copy()

        exclusions = []
        reasons = [(t, "price_unavailable") for t in price_missing]
        reasons += [(t, "liquidity_data_unavailable") for t in liquidity_missing]
        reasons += [(str(r["Yahoo Ticker"]), "bottom_liquidity_percentile") for _, r in percent_excluded.iterrows()]
        reasons += [(str(r["Yahoo Ticker"]), "liquidity_rank_cap") for _, r in cap_excluded.iterrows()]
        for ticker, reason in reasons:
            for _, row in frame[frame["Yahoo Ticker"].eq(ticker)].iterrows():
                exclusions.append({"symbol": str(row["Symbol"]), "yahoo_ticker": ticker, "reason": reason})

        st.session_state.liquidity_preview = {
            "retained": retained, "excluded": pd.DataFrame(exclusions), "prices": prices,
            "fx_rates": fx_rates, "exclude_percent": exclude_percent,
            "max_optimizer_positions": int(max_optimizer_positions), "original_count": len(frame),
            "zero_investment_start": zero_investment_start,
        }
    except Exception as exc:
        st.session_state.pop("liquidity_preview", None)
        st.error(f"Could not build the preview: {exc}")

preview = st.session_state.get("liquidity_preview")
if preview:
    retained, excluded = preview["retained"], preview["excluded"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original positions", preview["original_count"])
    c2.metric("Retained for optimizer", len(retained))
    c3.metric("Excluded", len(excluded))
    c4.metric("Retained", f"{len(retained) / preview['original_count']:.1%}")
    st.success(
        f"Read-only preview ready. Bottom {preview['exclude_percent']}% by liquidity is excluded, "
        f"with a maximum of {preview['max_optimizer_positions']} optimizer positions."
    )
    st.markdown("**Retained positions — most liquid first**")
    st.dataframe(retained.sort_values("Median Daily Traded Value (INR)", ascending=False), use_container_width=True, hide_index=True)
    st.markdown("**Excluded positions and reasons**")
    st.dataframe(excluded, use_container_width=True, hide_index=True)

    st.subheader("Stage 2 — Confirm and create JSON")
    confirmed = st.checkbox(f"I reviewed the preview and approve sending {len(retained)} positions to the optimizer.")
    if st.button("Create public basket input JSON", disabled=not confirmed, type="primary"):
        payload = generator.build_payload(
            retained, prices=preview["prices"], fx_rates=preview["fx_rates"],
            excluded_positions=excluded.to_dict(orient="records"),
            zero_investment_start=preview["zero_investment_start"],
        )
        payload["liquidity_filter"] = {
            "method": "bottom_percent_by_3_month_median_daily_traded_value_inr",
            "excluded_percent": preview["exclude_percent"],
            "maximum_optimizer_positions": preview["max_optimizer_positions"],
        }
        st.session_state.prepared_public_basket_json = json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        st.success("JSON created. No optimizer or database write has occurred.")

prepared = st.session_state.get("prepared_public_basket_json")
if prepared:
    st.download_button("Download public_basket_input.json", prepared, "public_basket_input.json", "application/json", type="primary")
    st.markdown("Next: upload the JSON to **Public Basket Publisher**, build its read-only optimizer preview, review it, and publish only after digest confirmation.")
