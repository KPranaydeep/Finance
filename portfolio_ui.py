"""Shared native Streamlit presentation; no data loading or financial logic."""
from __future__ import annotations

import streamlit as st


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
