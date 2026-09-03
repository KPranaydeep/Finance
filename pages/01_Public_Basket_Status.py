from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from nse_market_calendar import (
    SOURCE_AS_OF,
    SOURCE_REF,
)
from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    DEFAULT_BASKET_NAME,
    DEFAULT_STRATEGY_VERSION,
    connect_public_basket_db,
    create_public_basket,
    get_basket_record,
    get_public_basket_database_url,
    init_public_basket_schema,
    list_week_sessions,
    public_database_healthcheck,
    rebalance_gate,
    seed_nse_2026_calendar,
)


st.set_page_config(
    page_title="Public Basket Status",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Public Basket Scheduler")

st.caption(
    "Read-only infrastructure status. "
    "This page does not run the portfolio optimizer, "
    "publish a recommendation, or execute a trade."
)


# ---------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------

database_url = get_public_basket_database_url()

if not database_url:
    st.error(
        "Durable public-basket PostgreSQL is not configured."
    )

    st.markdown(
        """
Add this to your **Streamlit App Secrets**:

```toml
[public_basket]
database_url = "postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require"
```

Alternatively, set the environment variable:

```text
PUBLIC_BASKET_DATABASE_URL
```
"""
    )

    st.stop()


# ---------------------------------------------------------------------
# Current scheduler date
# ---------------------------------------------------------------------

india_timezone = ZoneInfo("Asia/Kolkata")
now_india = datetime.now(india_timezone)
today_india = now_india.date()


# ---------------------------------------------------------------------
# Database status
# ---------------------------------------------------------------------

try:
    conn = connect_public_basket_db(database_url)

    init_public_basket_schema(conn)

    basket_created = create_public_basket(
        conn=conn,
        basket_id=DEFAULT_BASKET_ID,
        basket_name=DEFAULT_BASKET_NAME,
        strategy_version=DEFAULT_STRATEGY_VERSION,
    )

    calendar_snapshot_id = seed_nse_2026_calendar(conn)

    health = public_database_healthcheck(conn)

    basket = get_basket_record(
        conn=conn,
        basket_id=DEFAULT_BASKET_ID,
    )

    gate = rebalance_gate(
        conn=conn,
        basket_id=DEFAULT_BASKET_ID,
        today=today_india,
    )

    week_sessions = list_week_sessions(
        conn=conn,
        basket_id=DEFAULT_BASKET_ID,
        reference_date=today_india,
    )

except Exception as exc:
    st.error(
        "The public-basket database could not be initialized or queried."
    )
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

st.success("PostgreSQL connection is healthy.")

column_1, column_2, column_3 = st.columns(3)

column_1.metric(
    "Scheduler date",
    today_india.isoformat(),
)

column_2.metric(
    "Gate status",
    gate.get("status", "UNKNOWN"),
)

column_3.metric(
    "Calendar source date",
    SOURCE_AS_OF,
)


# ---------------------------------------------------------------------
# Basket information
# ---------------------------------------------------------------------

st.subheader("Public basket")

if basket:
    st.write(
        {
            "basket_id": basket["basket_id"],
            "basket_name": basket["basket_name"],
            "strategy_version": basket["strategy_version"],
            "calendar_market": basket["calendar_market"],
            "rebalance_rule": basket["rebalance_rule"],
            "created_now": basket_created,
        }
    )
else:
    st.warning("The public basket record was not found.")


# ---------------------------------------------------------------------
# Rebalance gate
# ---------------------------------------------------------------------

st.subheader("Weekly rebalance gate")

gate_status = gate.get("status", "UNKNOWN")

if gate_status == "DUE":
    st.warning(
        "The public basket is due for its scheduled evaluation today."
    )
elif gate_status == "ALREADY_EVALUATED":
    st.success(
        "This week's public-basket cycle has already been evaluated."
    )
elif gate_status == "NOT_DUE":
    st.info(
        "This week's scheduled trading session has not arrived yet."
    )
elif gate_status == "MISSED":
    st.error(
        "This week's scheduled session has passed without an "
        "official evaluation. The scheduler must not catch up later."
    )
elif gate_status == "NO_OPEN_SESSION":
    st.info("There is no open NSE session during this week.")
elif gate_status == "CALENDAR_INCOMPLETE":
    st.error(
        "No verified market-calendar snapshot covers this week."
    )
else:
    st.warning(f"Unexpected gate status: {gate_status}")

st.write(gate)


# ---------------------------------------------------------------------
# Week calendar
# ---------------------------------------------------------------------

st.subheader("NSE sessions for this week")

if week_sessions:
    sessions_frame = pd.DataFrame(week_sessions)

    if "session_date" in sessions_frame.columns:
        sessions_frame["session_date"] = (
            sessions_frame["session_date"].astype(str)
        )

    st.dataframe(
        sessions_frame,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No session records are available for the current week."
    )


# ---------------------------------------------------------------------
# Technical details
# ---------------------------------------------------------------------

with st.expander("Technical details"):
    st.write(
        {
            "database_name": health.get("database_name"),
            "database_time": str(health.get("database_time")),
            "calendar_snapshot_id": calendar_snapshot_id,
            "calendar_source": SOURCE_REF,
            "calendar_source_as_of": SOURCE_AS_OF,
            "timezone": "Asia/Kolkata",
        }
    )


conn.close()
