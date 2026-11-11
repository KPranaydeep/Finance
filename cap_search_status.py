"""Read-only cap-search provenance beside the primary search action."""
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

from shortlist_search_checkpoint import load_latest_checkpoint


@lru_cache(maxsize=128)
def _session_closes(start, end):
    return mcal.get_calendar("NSE").schedule(start, end)["market_close"]


def scan_age_text(checkpoint, *, now=None):
    """Count completed NSE closes since discovery, never since a reuse save."""
    if not checkpoint or not checkpoint.get("results"):
        return "No saved cap scan yet."
    rows = checkpoint["results"]
    partial = checkpoint.get("status") != "complete"
    context = next((row["Search skipper context"] for row in reversed(rows)
                    if isinstance(row.get("Search skipper context"), dict)), {})
    stamp = (context.get("last_discovery_at_utc") if not partial else None) or checkpoint.get("updated_at")
    label = "Partial scan checkpoint" if partial else "Last cap scan"
    try:
        scanned = pd.Timestamp(stamp)
        current = pd.Timestamp(now or datetime.now(timezone.utc))
        if pd.isna(scanned) or scanned.tzinfo is None or current.tzinfo is None or scanned > current:
            raise ValueError("Dated scan evidence unavailable")
        local = scanned.tz_convert(ZoneInfo("Asia/Kolkata"))
        closes = _session_closes(str(local.date()), str(current.tz_convert(ZoneInfo("Asia/Kolkata")).date()))
        age = int(((closes > scanned) & (closes <= current)).sum())
        text = f"{label}: **{age} NSE trading session{'s' if age != 1 else ''} old** · {local:%d %b %Y, %H:%M} IST."
    except (ValueError, TypeError, KeyError, OverflowError):
        text = f"{label}: trading-session age unavailable."
    if partial:
        text += f" {len(rows):,} pairs saved; resume available."
    if checkpoint.get("analysis_cutoff"):
        text += f" Price history through {str(checkpoint['analysis_cutoff'])[:10]}."
    return text


def render_scan_age(placeholder, owner, connect, *, load=load_latest_checkpoint, now=None):
    try:
        with connect() as conn:
            checkpoint = load(conn, owner)
        text = scan_age_text(checkpoint, now=now)
    except Exception:
        # A read/calendar failure cannot turn missing evidence into "fresh".
        text = "Saved cap scan age unavailable; checkpoint was not changed."
    placeholder.caption(text)


def render_cap_search_action(owner, connect, *, enabled, load=load_latest_checkpoint, now=None):
    import streamlit as st
    clicked = st.button(
        "Run / resume cap search", width="stretch", key="run_shortlist_cap_search_btn",
        disabled=not enabled,
        help="Uses the sidebar search settings, including recent-search reuse. Age counts completed NSE sessions since discovery (or the partial checkpoint), not calendar days; other markets can differ.",
    )
    age_slot = st.empty()
    render_scan_age(age_slot, owner, connect, load=load, now=now)
    if not enabled:
        st.caption("Enable Advanced shortlist-cap search in the sidebar to search.")
    return clicked, age_slot
