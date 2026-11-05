"""Private allocation pause controls, independent of executable trade rows."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_deferrals import (
    active_trade_deferrals,
    defer_trades_for_quarter,
    optimal_weight_pause_rows,
)


def _save_weight_pauses(rows, owner, key_prefix, get_connection):
    """Save before the fragment redraws, using only submitted checkbox edits."""
    edits = st.session_state.get(f"{key_prefix}_pause_editor", {}).get("edited_rows", {})
    indices = [int(index) for index, edit in edits.items()
               if str(index).isdigit() and int(index) < len(rows)
               and edit.get("Pause 90 days") is True]
    with get_connection() as conn:
        active = active_trade_deferrals(conn, owner)
        selected = rows.iloc[indices].copy()
        selected = selected.loc[~selected["Yahoo Ticker"].isin(active["yahoo_ticker"])]
        if selected.empty:
            st.session_state[f"{key_prefix}_pause_notice"] = (
                "warning", "Select an unpaused security; existing pause expiries are unchanged."
            )
            return
        selected["Action"] = "Allocation"
        saved = defer_trades_for_quarter(conn, owner, selected.to_dict(orient="records"))
    tickers = ", ".join(row["Yahoo Ticker"] for row in saved)
    st.session_state[f"{key_prefix}_pause_notice"] = (
        "success", f"Paused for 90 days: {tickers}. Saved optimal weights are unchanged."
    )


@st.fragment
def render_active_security_pauses(owner, get_connection, key_prefix):
    """Always-readable persistent status, independent of optimizer results."""
    with get_connection() as conn:
        active = active_trade_deferrals(conn, owner)
    if st.button("Refresh pause status", key=f"{key_prefix}_refresh_pauses"):
        with get_connection() as conn:
            active = active_trade_deferrals(conn, owner)
    if active.empty:
        st.caption("No active security pauses.")
        return
    display = active.rename(columns={"yahoo_ticker": "Yahoo Ticker"})
    display["Paused until"] = pd.to_datetime(display["expires_at"], utc=True).dt.tz_convert(
        "Asia/Kolkata").dt.strftime("%d %b %Y, %H:%M IST")
    st.markdown(f"**{len(active)} securities paused** · excluded until the dates below.")
    st.dataframe(display[["Yahoo Ticker", "Paused until"]], hide_index=True, width="stretch")


@st.fragment
def render_optimal_weight_pause_editor(weights, owner, key_prefix, get_connection):
    """Show every positive target and save pauses to the existing owner registry."""
    rows = optimal_weight_pause_rows(weights)
    if rows.empty:
        st.info("No positive optimal weights are available to pause.")
        return
    with get_connection() as conn:
        deferrals = active_trade_deferrals(conn, owner)
    expiry = dict(zip(deferrals["yahoo_ticker"], deferrals["expires_at"]))
    rows["Paused until"] = pd.to_datetime(
        rows["Yahoo Ticker"].map(expiry), utc=True, errors="coerce"
    ).dt.tz_convert("Asia/Kolkata").dt.strftime("%d %b %Y, %H:%M IST").fillna("Not paused")
    rows.insert(0, "Pause 90 days", False)
    notice = st.session_state.pop(f"{key_prefix}_pause_notice", None)
    if notice:
        getattr(st, notice[0])(notice[1])
    st.caption(
        "Pause any allocated security, including holds with no executable trade. "
        "The pause blocks trades immediately and excludes it from future optimizer "
        "runs for 90 days; the saved weights remain a record of the original analysis."
    )
    with st.form(f"{key_prefix}_pause_form", border=False):
        st.data_editor(
            rows, key=f"{key_prefix}_pause_editor", width="stretch", hide_index=True,
            disabled=[column for column in rows if column != "Pause 90 days"],
            column_config={
                "Pause 90 days": st.column_config.CheckboxColumn(
                    "Pause 90 days", default=False, pinned=True,
                    help="Save a security-level pause, whether or not a trade exists.",
                ),
                "Optimal Weight": st.column_config.NumberColumn(format="percent"),
                "Paused until": st.column_config.TextColumn(),
            },
        )
        st.form_submit_button(
            "Pause selected securities for 90 days", icon=":material/pause_circle:",
            on_click=_save_weight_pauses,
            args=(rows, owner, key_prefix, get_connection),
        )
