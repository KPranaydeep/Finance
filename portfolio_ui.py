"""Shared native Streamlit presentation; no data loading or financial logic."""
from __future__ import annotations

import streamlit as st
import pandas as pd


def render_portfolio_heading(title, description, *, context=None, publication=None):
    if context:
        st.caption(context)
    st.title(title)
    st.write(description)
    if publication:
        st.badge(publication, icon=":material/verified:", color="gray")


def render_metric_strip(items):
    """Responsive native cards with visible notes, including unavailable states."""
    items = list(items)
    if not items:
        return
    # Native columns stack on small screens instead of squeezing or clipping
    # dates, currency amounts and unavailable-state explanations.
    for column, (label, value, note) in zip(st.columns(len(items), gap="small"), items):
        with column.container(border=True):
            st.metric(label, value)
            if note:
                st.caption(note)


def render_execution_validation(report, proposed_orders=()):
    """Present final feasibility separately from research and target statistics."""
    report = report or {}
    with st.container(border=True):
        st.markdown("**Final allocation check**")
        if report.get("passed"):
            st.success("Plan passed the recorded funding, position and available risk checks.")
            st.caption("Sale proceeds must be available before buys. This is modeled feasibility, not broker execution approval.")
        elif report:
            st.warning("Plan withheld: the final allocation did not pass every required check. Saved analysis is retained.")
            for reason in dict.fromkeys(report.get("violations") or ["Validation evidence unavailable"]):
                st.write("• " + str(reason))
        else:
            st.info("This older plan has no final allocation check. Use Update plan to validate it without rerunning optimization.")
        if proposed_orders:
            with st.expander("Withheld proposal — review only"):
                st.caption("This is a draft for review, excluded from the executable plan.")
                draft = pd.DataFrame(proposed_orders).rename(columns={
                    "Executable Quantity": "Proposed quantity", "Executable Value": "Proposed value",
                })
                st.dataframe(draft, hide_index=True, width="stretch")
