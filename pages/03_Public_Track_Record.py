"""Public, publication-linked security track records with owner exports on demand."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from public_basket_postgres import DEFAULT_BASKET_ID
from public_card_feed import build_card_feed, load_public_record
from public_track_record import (
    BENCHMARK_LABEL,
    WORLD_LABEL,
    build_card_batch,
    evidence_csv,
    exited_symbol_rows,
    exited_symbols_csv,
    exited_symbols_text,
    load_security_evidence,
    percent,
    percentage_points,
    portfolio_cover_card,
    share_text,
    whatsapp_card,
)


IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Public portfolio track record",
    page_icon=":material/query_stats:",
    layout="wide",
)

st.title("Portfolio track record")
st.caption(
    "Publication-linked evidence for every security held by the public model portfolio. "
    "Read-only · no trades · no database writes"
)

try:
    record = load_public_record(DEFAULT_BASKET_ID)
    current = record.get("current")
    if not record.get("basket") or not current:
        raise RuntimeError("No active public portfolio publication is available.")
    feed = build_card_feed(record, current)
except Exception:
    LOGGER.exception("Public track-record feed could not be loaded")
    st.error("The verified public track record is temporarily unavailable.")
    st.stop()

active = sorted(
    (
        item
        for item in feed["securities"]
        if str(item.get("status", "active")).lower() == "active"
    ),
    key=lambda item: (-float(item.get("target_weight") or 0), item["ticker"]),
)
exited = sorted(
    (
        item
        for item in feed["securities"]
        if str(item.get("status", "active")).lower() == "removed"
    ),
    key=lambda item: (str(item.get("exit_date") or ""), item["ticker"]),
    reverse=True,
)

with st.container(horizontal=True):
    st.metric("Portfolio version", feed["portfolio_version"])
    st.metric("Current holdings", len(active))
    st.metric("Recorded exits", len(exited))
    st.metric("Published", str(feed["publication_date"]))

st.subheader("Evidence deck")
st.write(
    "Start with the portfolio summary, then move through one publication-linked "
    "security card at a time."
)

if not active:
    st.info("The latest publication has no active securities.")
    st.stop()

scope = "Current holdings"
if exited:
    scope = st.segmented_control(
        "Track-record set",
        ["Current holdings", "Exited holdings"],
        default="Current holdings",
        selection_mode="single",
    )
selection_rows = active if scope == "Current holdings" else exited
slide_number = st.pagination(
    len(selection_rows) + 1,
    default=1,
    max_visible_pages=5,
    width="stretch",
    key=f"track_record_slides_{scope}",
)
st.caption(
    f"Slide {slide_number} of {len(selection_rows) + 1} · "
    + ("Portfolio summary" if slide_number == 1 else selection_rows[slide_number - 2]["ticker"])
)

if slide_number == 1:
    cover = portfolio_cover_card(feed, selection_rows, scope_label=scope)
    st.image(cover, width="stretch")
    st.caption(
        "The cover is built directly from the immutable publication record and "
        "requires no market-data request."
    )
else:
    selected = selection_rows[slide_number - 2]
    selected_ticker = selected["ticker"]
    now = datetime.now(IST)
    refresh_bucket = now.replace(second=0, microsecond=0)
    refresh_bucket = refresh_bucket.replace(minute=refresh_bucket.minute // 5 * 5)
    try:
        with st.skeleton(height=360):
            metrics, chart = load_security_evidence(
                selected_ticker,
                str(selected["entry_date"]),
                exit_date=(
                    str(selected["exit_date"]) if selected.get("exit_date") else None
                ),
                refresh_bucket=refresh_bucket.isoformat(),
            )
        share_card = whatsapp_card(selected_ticker, metrics, chart)
        st.image(share_card, width="stretch")
        st.caption(
            f"{selected_ticker} · first publication {selected['entry_date']} · "
            f"evidence through {metrics['as_of']}"
        )

        detail_section = st.expander(
            "Interactive evidence",
            expanded=False,
            icon=":material/analytics:",
            on_change="rerun",
        )
        if detail_section.open:
            with detail_section:
                symbol = metrics.get("price_symbol", "₹")
                with st.container(horizontal=True):
                    st.metric(
                        "₹100 became",
                        f"₹{100 * (1 + metrics['ticker_return']):,.2f}",
                    )
                    st.metric("Return in INR", percent(metrics["ticker_return"]))
                    st.metric(
                        "Versus Nifty 50",
                        percentage_points(metrics["excess_return"]),
                    )
                    st.metric(
                        "Versus VT world",
                        percentage_points(metrics["excess_world_return"]),
                    )
                st.caption(
                    f"{metrics['calendar_days']} calendar days / "
                    f"{metrics['ticker_sessions']} market sessions"
                )
                display_chart = chart.rename(columns={"Ticker": selected_ticker})
                st.line_chart(
                    display_chart,
                    y_label="Growth of ₹100",
                    color=["#9f4339", "#315f78", "#b27a18"],
                    height=460,
                )
                price_left, price_right = st.columns(2)
                price_left.metric(
                    "Entry close", f"{symbol}{metrics['entry_close']:,.2f}"
                )
                price_right.metric(
                    str(metrics.get("endpoint_label", "Latest price")),
                    f"{symbol}{metrics['endpoint_price']:,.2f}",
                )
                st.caption(
                    f"Latest price observed: {metrics['endpoint_as_of']} · "
                    "Overseas returns are translated to INR using same-date USD/INR."
                )

        share_section = st.expander(
            "Download and share this slide",
            expanded=False,
            icon=":material/share:",
            on_change="rerun",
        )
        if share_section.open:
            with share_section:
                with st.container(horizontal=True):
                    st.download_button(
                        "Download image",
                        share_card,
                        file_name=(
                            f"{selected_ticker.replace('^', '')}-track-record-"
                            f"{metrics['as_of']}.png"
                        ),
                        mime="image/png",
                        type="primary",
                        on_click="ignore",
                    )
                    st.download_button(
                        "Download evidence",
                        evidence_csv(selected_ticker, metrics, chart),
                        file_name=(
                            f"{selected_ticker.replace('^', '')}-track-record-"
                            f"{metrics['as_of']}.csv"
                        ),
                        mime="text/csv",
                        on_click="ignore",
                    )
                    st.download_button(
                        "Download caption",
                        share_text(selected_ticker, metrics, chart),
                        file_name=(
                            f"{selected_ticker.replace('^', '')}-share-caption.txt"
                        ),
                        mime="text/plain",
                        on_click="ignore",
                    )
    except Exception:
        LOGGER.exception("Selected security evidence could not be built")
        st.warning(
            "Market data for this slide is temporarily unavailable. Move to another "
            "slide or retry later; the immutable publication record is unchanged."
        )

composition_section = st.expander(
    "Portfolio composition",
    expanded=False,
    icon=":material/list:",
    on_change="rerun",
)
if composition_section.open:
    with composition_section:
        holdings_frame = pd.DataFrame(
            [
                {
                    "Security": item["ticker"],
                    "Target weight": float(item["target_weight"]),
                    "Tracked since": item["entry_date"],
                }
                for item in active
            ]
        )
        holdings_frame["Tracked since"] = pd.to_datetime(
            holdings_frame["Tracked since"], errors="coerce"
        ).dt.date
        st.dataframe(
            holdings_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Target weight": st.column_config.NumberColumn(format="percent"),
                "Tracked since": st.column_config.DateColumn(format="DD MMM YYYY"),
            },
        )
        if exited:
            st.markdown("**Exited securities**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Security": item["ticker"],
                            "Tracked since": item["entry_date"],
                            "Exit publication": item.get("exit_date"),
                        }
                        for item in exited
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "An exit is historical evidence, not a current sell instruction."
            )


owner_section = st.expander(
    "Owner exports and maintenance",
    expanded=False,
    icon=":material/build:",
    on_change="rerun",
)
if owner_section.open:
    with owner_section:
        st.caption(
            "Operational downloads are intentionally separated from the public reading path. "
            "They expose no write controls."
        )
        feed_bytes = json.dumps(feed, indent=2, sort_keys=True, default=str).encode("utf-8")
        st.download_button(
            "Download card-feed JSON",
            feed_bytes,
            file_name=f"{DEFAULT_BASKET_ID.lower()}-card-feed.json",
            mime="application/json",
            on_click="ignore",
        )
        recent_exits = exited_symbol_rows(feed, lookback_days=90)
        annual_exits = exited_symbol_rows(feed, lookback_days=365)
        if recent_exits:
            st.code(exited_symbols_text(recent_exits).decode("utf-8").strip())
            with st.container(horizontal=True):
                st.download_button(
                    "90-day removal list",
                    exited_symbols_text(recent_exits),
                    file_name="remove-from-universal-last-90-days.txt",
                    mime="text/plain",
                    on_click="ignore",
                )
                st.download_button(
                    "One-year exit review",
                    exited_symbols_csv(annual_exits),
                    file_name="exited-last-365-days-review.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
        else:
            st.caption("No dated exits fall within the rolling 90-day cooldown.")

        if st.button("Generate complete card archive", icon=":material/archive:"):
            try:
                with st.status("Generating cards from current market history…", expanded=True) as status:
                    st.session_state["track_record_archive"] = build_card_batch(feed)
                    status.update(label="Card archive ready", state="complete", expanded=False)
            except Exception:
                LOGGER.exception("Complete public card archive generation failed")
                st.error("The complete archive could not be generated from current market data.")
        archive = st.session_state.get("track_record_archive")
        if archive:
            st.download_button(
                "Download complete archive",
                archive,
                file_name=f"{DEFAULT_BASKET_ID.lower()}-cards.zip",
                mime="application/zip",
                type="primary",
                on_click="ignore",
            )

method_section = st.expander(
    "How this evidence is calculated",
    expanded=False,
    icon=":material/info:",
)
with method_section:
    st.markdown(
        "- The start date comes from the first active immutable portfolio publication containing the security.\n"
        "- Adjusted closes account for splits and distributions where Yahoo Finance supplies them.\n"
        "- Overseas security and VT returns are converted to INR using same-date USD/INR data.\n"
        "- Active holdings use the latest available price snapshot; exited holdings end at their exit publication.\n"
        "- Results are historical evidence, not a forecast, recommendation, or guarantee."
    )

st.caption(
    f"{feed['basket_id']} · {feed['portfolio_version']} · publication {feed['publication_id']} · "
    f"{BENCHMARK_LABEL} and {WORLD_LABEL} comparisons"
)
