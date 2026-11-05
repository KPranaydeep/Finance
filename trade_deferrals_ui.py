"""Private allocation pause controls, independent of executable trade rows."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_deferrals import (
    active_trade_deferrals,
    defer_trades_for_quarter,
    optimal_weight_pause_rows,
)


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
    st.caption(
        "Pause any allocated security, including holds with no executable trade. "
        "The pause blocks trades immediately and excludes it from future optimizer "
        "runs for 90 days; the saved weights remain a record of the original analysis."
    )
    with st.form(f"{key_prefix}_pause_form", border=False):
        edited = st.data_editor(
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
        submitted = st.form_submit_button(
            "Pause selected securities for 90 days", icon=":material/pause_circle:"
        )
    if submitted:
        selected = edited.loc[
            edited["Pause 90 days"].fillna(False)
            & ~edited["Yahoo Ticker"].isin(expiry)
        ].copy()
        if selected.empty:
            st.warning("Select an unpaused security; existing pause expiries are unchanged.")
            return
        selected["Action"] = "Allocation"
        with get_connection() as conn:
            defer_trades_for_quarter(conn, owner, selected.to_dict(orient="records"))
        st.rerun()
