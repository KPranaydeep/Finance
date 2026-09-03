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
    "This page does NOT run the portfolio optimizer, "
    "publish a recommendation, or execute a trade."
)


# =====================================================================
# DATABASE CONFIG
# =====================================================================

database_url = (
    get_public_basket_database_url()
)


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
