from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import generate_public_basket_input as generator


DEFAULT_CSV = Path("universal_portfolio_backup.csv")
PAGE_VERSION = "event-input-generator-r2"
PAGE_VERSION = "event-input-generator-r4"
MINIMUM_TICKER_COVERAGE = 0.80


st.set_page_config(
    page_title="Prepare Public Basket Input",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Prepare Public Basket Input")
st.caption(f"{PAGE_VERSION} · Generates JSON only · No PostgreSQL writes")
st.caption(
    f"JSON generation threshold: {MINIMUM_TICKER_COVERAGE:.0%} of open-position tickers"
)
st.info(
    "This page fetches current prices and creates the input JSON used by Public "
    "Basket Publisher. It does not run the optimizer and cannot publish ledger records."
)

uploaded_csv = st.file_uploader(
    "Holdings CSV (optional)",
    type=["csv"],
    help=(
        "Leave empty to use universal_portfolio_backup.csv from the repository, "
        "or upload a newer holdings CSV."
    ),
)

source_label = uploaded_csv.name if uploaded_csv is not None else str(DEFAULT_CSV)
st.write("Selected source:", f"`{source_label}`")

if uploaded_csv is None and not DEFAULT_CSV.is_file():
    st.error("universal_portfolio_backup.csv is missing. Upload a holdings CSV above.")
    st.stop()

if st.button("Fetch prices and prepare JSON", type="primary"):
    try:
        if uploaded_csv is not None:
            raw_csv = uploaded_csv.getvalue()
            frame = generator.validate_holdings(pd.read_csv(io.BytesIO(raw_csv)))
        else:
            frame = generator.load_holdings(str(DEFAULT_CSV))

        tickers = frame["Yahoo Ticker"].astype(str).str.strip().str.upper().tolist()
        applied_aliases = frame.attrs.get("ticker_aliases_applied", [])
        if applied_aliases:
            st.info(
                "Corrected verified legacy ticker aliases before fetching prices."
            )
            st.dataframe(
                pd.DataFrame(applied_aliases).rename(
                    columns={"from": "CSV ticker", "to": "ETF ticker"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        progress_bar = st.progress(0, text="Preparing price batches…")

        def update_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Price batch {done} of {total}")

        with st.spinner("Fetching current prices from Yahoo Finance…"):
            prices, missing = generator.fetch_latest_prices_with_missing(
                tickers,
                progress=update_progress,
            )

        progress_bar.empty()
        if missing:
        total_tickers = len(set(tickers))
        coverage = len(prices) / total_tickers if total_tickers else 0.0
        if missing and coverage < MINIMUM_TICKER_COVERAGE:
            st.session_state.pop("prepared_public_basket_json", None)
            st.session_state.pop("prepared_public_basket_summary", None)
            st.error(
                f"Prices were found for {len(prices):,} of {len(set(tickers)):,} tickers. "
                f"The JSON was not created because {len(missing):,} open positions remain unresolved."
                f"Prices were found for {len(prices):,} of {total_tickers:,} tickers "
                f"({coverage:.1%}). The JSON requires at least "
                f"{MINIMUM_TICKER_COVERAGE:.0%} coverage."
            )
            missing_frame = pd.DataFrame({"Yahoo Ticker": missing})
            st.dataframe(missing_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download unresolved tickers CSV",
                data=missing_frame.to_csv(index=False).encode("utf-8"),
                file_name="public_basket_unresolved_tickers.csv",
                mime="text/csv",
            )
        else:
            excluded_positions: list[dict[str, str]] = []
            build_frame = frame
            if missing:
                missing_set = set(missing)
                excluded = frame[frame["Yahoo Ticker"].isin(missing_set)]
                excluded_positions = [
                    {
                        "symbol": str(row["Symbol"]).strip(),
                        "yahoo_ticker": str(row["Yahoo Ticker"]).strip().upper(),
                        "reason": "price_unavailable",
                    }
                    for _, row in excluded.iterrows()
                ]
                build_frame = frame[~frame["Yahoo Ticker"].isin(missing_set)].copy()
                build_frame.attrs = frame.attrs.copy()
                st.warning(
                    f"Coverage is {coverage:.1%}, above the "
                    f"{MINIMUM_TICKER_COVERAGE:.0%} minimum. "
                    f"The {len(excluded_positions):,} unresolved positions are excluded "
                    "and the included weights will be recalculated to 100%."
                )
                st.dataframe(
                    pd.DataFrame(excluded_positions),
                    use_container_width=True,
                    hide_index=True,
                )
            with st.spinner("Building immutable input snapshot…"):
                payload = generator.build_payload(frame, prices=prices)
                payload = generator.build_payload(
                    build_frame,
                    prices=prices,
                    excluded_positions=excluded_positions,
                )
            encoded = json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            st.session_state.prepared_public_basket_json = encoded
            st.session_state.prepared_public_basket_summary = {
                "positions": len(payload["portfolio"]),
                "excluded_positions": len(payload["excluded_positions"]),
                "market_data_cutoff": payload["market_data_cutoff"],
                "source": source_label,
                "ticker_aliases_applied": payload.get("ticker_aliases_applied", []),
            }
            st.success("The public-basket input JSON is ready.")
    except Exception as exc:
        st.session_state.pop("prepared_public_basket_json", None)
        st.error(f"Could not prepare the input JSON: {exc}")

prepared = st.session_state.get("prepared_public_basket_json")
summary = st.session_state.get("prepared_public_basket_summary")
if prepared and summary:
    st.subheader("Prepared snapshot")
    c1, c2 = st.columns(2)
    c1.metric("Open positions", summary["positions"])
    c2.write("Market data cutoff (UTC)")
    c2.code(summary["market_data_cutoff"], language="text")
    c1, c2, c3 = st.columns(3)
    c1.metric("Included positions", summary["positions"])
    c2.metric("Excluded positions", summary.get("excluded_positions", 0))
    c3.write("Market data cutoff (UTC)")
    c3.code(summary["market_data_cutoff"], language="text")
    st.download_button(
        "Download public basket input JSON",
        data=prepared,
        file_name="public_basket_input.json",
        mime="application/json",
        type="primary",
    )
    st.markdown(
        "Next: open **Public Basket Publisher**, upload `public_basket_input.json`, "
        "and build the read-only preview. Do not publish until the preview is reviewed."
    )
