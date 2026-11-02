"""Named, searchable navigation without loading every security's market history."""

from datetime import date

import streamlit as st


OVERVIEW = "__portfolio_overview__"


def _date_label(value):
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%d %b %Y")
    except ValueError:
        return "date unavailable"


def evidence_option_label(row):
    ticker = str(row["ticker"])
    weight = float(row.get("target_weight") or 0)
    if str(row.get("status", "active")).lower() == "removed":
        return f"{ticker} · last target {weight:.0%} · exited {_date_label(row.get('exit_date'))}"
    return f"{ticker} · target {weight:.0%} · since {_date_label(row.get('entry_date'))}"


def _select(key, value):
    st.session_state[key] = value


def render_evidence_picker(rows, scope):
    """Return the selected row, or None for the portfolio overview."""
    by_ticker = {str(row["ticker"]): row for row in rows}
    options = [OVERVIEW, *by_ticker]
    key = f"track_record_security_{scope}"
    if st.session_state.get(key) not in options:
        st.session_state[key] = OVERVIEW
    chosen = st.selectbox(
        "Open evidence", options, key=key, persist_state="session",
        format_func=lambda value: "Portfolio overview" if value == OVERVIEW
        else evidence_option_label(by_ticker[value]),
        help="Type a ticker to jump directly. Weights and dates come from the publication record.",
    )
    index = options.index(chosen)
    previous = options[max(0, index - 1)]
    following = options[min(len(options) - 1, index + 1)]

    def short_name(value):
        return "Portfolio overview" if value == OVERVIEW else value

    with st.container(horizontal=True):
        st.button(
            "Previous" if index == 0 else f"Previous: {short_name(previous)}",
            icon=":material/arrow_back:", disabled=index == 0,
            key=f"{key}_previous", on_click=_select, args=(key, previous),
        )
        st.button(
            "Next" if index == len(options) - 1 else f"Next: {short_name(following)}",
            icon=":material/arrow_forward:", disabled=index == len(options) - 1,
            key=f"{key}_next", on_click=_select, args=(key, following),
        )
    return by_ticker.get(chosen)
