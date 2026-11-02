"""Named, searchable navigation without loading every security's market history."""

from datetime import date

import streamlit as st
from public_evidence_values import lifecycle_key, value_of_100


OVERVIEW = "__portfolio_overview__"


def _date_label(value):
    try:
        return date.fromisoformat(str(value)[:10]).strftime("%d %b %Y")
    except ValueError:
        return "date unavailable"


def evidence_option_label(row):
    ticker = str(row["ticker"])
    if "evidence_value_100" in row:
        value = row["evidence_value_100"]
        ticker += f" · ₹100 → ₹{value:,.2f}" if value is not None else " · ₹100 → —"
    weight = float(row.get("target_weight") or 0)
    if str(row.get("status", "active")).lower() == "removed":
        return f"{ticker} · last allocation {weight:.0%} · exited {_date_label(row.get('exit_date'))}"
    return f"{ticker} · allocation {weight:.0%} · since {_date_label(row.get('entry_date'))}"


def rank_evidence_rows(rows, summaries):
    """Sort recorded lifecycle returns; missing evidence is never zero."""
    evidence = {lifecycle_key(row): row for row in summaries}
    ranked = []
    for row in rows:
        value = value_of_100(evidence.get(lifecycle_key(row), {}))
        ranked.append({**row, "evidence_value_100": value,
                       "evidence_return": value / 100 - 1 if value is not None else None})
    return sorted(ranked, key=lambda row: (
        row["evidence_value_100"] is None,
        -row["evidence_value_100"] if row["evidence_value_100"] is not None else 0,
        str(row["ticker"]),
    ))


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
        help="Search by ticker. ₹100 → ₹… uses the same INR outcome as the evidence card, sorted highest first. Holding periods differ.",
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
