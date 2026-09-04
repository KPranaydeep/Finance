from __future__ import annotations

import inspect
import json
import logging
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import streamlit as st

from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    DEFAULT_BASKET_NAME,
    DEFAULT_STRATEGY_VERSION,
    connect_public_basket_db,
    create_public_basket,
    get_basket_record,
    get_public_basket_counts,
    get_public_basket_database_url,
    init_public_basket_schema,
    list_rebalance_events,
    list_signal_runs,
    list_trade_executions,
    list_trade_orders,
    public_database_healthcheck,
)


INDIA_TZ = ZoneInfo("Asia/Kolkata")
RECENT_LIMIT = 25
LOGGER = logging.getLogger(__name__)


def display_value(value: Any) -> Any:
    """Convert database values into compact, display-safe values."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=INDIA_TZ)
        return value.astimezone(INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return value


def display_row(row: Any) -> dict[str, Any]:
    values = dict(row)
    return {key: display_value(value) for key, value in values.items()}


def display_rows(rows: Any) -> list[dict[str, Any]]:
    return [display_row(row) for row in (rows or [])]


def call_ledger_reader(
    reader: Callable[..., Any],
    conn: Any,
    *,
    basket_id: str,
    limit: int | None = None,
    rebalance_id: str | None = None,
    order_id: str | None = None,
) -> Any:
    """Call a ledger reader using only arguments supported by its signature."""
    available = {
        "basket_id": basket_id,
        "limit": limit,
        "rebalance_id": rebalance_id,
        "order_id": order_id,
    }
    parameters = inspect.signature(reader).parameters
    for name, parameter in parameters.items():
        if name == "conn" or parameter.default is not inspect.Parameter.empty:
            continue
        if name in available and available[name] is None:
            return []
    kwargs = {
        name: available[name]
        for name in available
        if name in parameters and available[name] is not None
    }
    return reader(conn, **kwargs)


def reader_accepts(reader: Callable[..., Any], parameter_name: str) -> bool:
    return parameter_name in inspect.signature(reader).parameters


def show_table(title: str, rows: list[dict[str, Any]], empty_message: str) -> None:
    st.subheader(title)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info(empty_message)


st.set_page_config(
    page_title="Public Basket Status",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Public Basket Status")
st.caption(
    "Inspection of the durable event ledger with idempotent schema and basket "
    "initialization. This page never runs the optimizer, creates portfolio events, "
    "places orders, or executes trades."
)

database_url = get_public_basket_database_url()

if not database_url:
    st.error("Durable public-basket PostgreSQL is not configured.")
    st.markdown(
        """
Add the connection to **Streamlit App Secrets**:

```toml
[public_basket]
database_url = "postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require"
```

Alternatively, set `PUBLIC_BASKET_DATABASE_URL` in the app environment.
"""
    )
    st.stop()

connection = None

try:
    connection = connect_public_basket_db(database_url)
    init_public_basket_schema(connection)
    basket_created = create_public_basket(
        conn=connection,
        basket_id=DEFAULT_BASKET_ID,
        basket_name=DEFAULT_BASKET_NAME,
        strategy_version=DEFAULT_STRATEGY_VERSION,
    )

    health = public_database_healthcheck(connection)
    basket = get_basket_record(connection, DEFAULT_BASKET_ID)
    counts = call_ledger_reader(
        get_public_basket_counts,
        connection,
        basket_id=DEFAULT_BASKET_ID,
    )
    signal_runs_raw = list(call_ledger_reader(
        list_signal_runs,
        connection,
        basket_id=DEFAULT_BASKET_ID,
        limit=RECENT_LIMIT,
    ) or [])
    rebalance_events_raw = list(call_ledger_reader(
        list_rebalance_events,
        connection,
        basket_id=DEFAULT_BASKET_ID,
        limit=RECENT_LIMIT,
    ) or [])

    rebalance_events = display_rows(rebalance_events_raw)
    latest_rebalance_id = (
        dict(rebalance_events_raw[0]).get("rebalance_id")
        if rebalance_events_raw
        else None
    )
    trade_orders_raw = list(call_ledger_reader(
        list_trade_orders,
        connection,
        basket_id=DEFAULT_BASKET_ID,
        rebalance_id=latest_rebalance_id,
        limit=RECENT_LIMIT,
    ) or [])
    latest_order_ids = [
        dict(row).get("order_id")
        for row in trade_orders_raw
        if dict(row).get("order_id")
    ]

    if reader_accepts(list_trade_executions, "order_id"):
        execution_rows: list[Any] = []
        for order_id in latest_order_ids:
            execution_rows.extend(
                call_ledger_reader(
                    list_trade_executions,
                    connection,
                    basket_id=DEFAULT_BASKET_ID,
                    order_id=order_id,
                    limit=RECENT_LIMIT,
                )
                or []
            )

        executions_by_id = {
            dict(row).get("execution_id", f"row-{index}"): row
            for index, row in enumerate(execution_rows)
        }
        trade_executions_raw = sorted(
            executions_by_id.values(),
            key=lambda row: str(dict(row).get("executed_at", "")),
            reverse=True,
        )[:RECENT_LIMIT]
    else:
        trade_executions_raw = list(
            call_ledger_reader(
                list_trade_executions,
                connection,
                basket_id=DEFAULT_BASKET_ID,
                limit=RECENT_LIMIT,
            )
            or []
        )

except Exception:
    LOGGER.exception("Public-basket status query failed")
    st.error("The public-basket database could not be initialized or queried.")
    st.info("Please try again later or ask the application operator to check the logs.")
    st.stop()
finally:
    if connection is not None:
        connection.close()


now_india = datetime.now(INDIA_TZ)
health_values = dict(health or {})
basket_values = display_row(basket) if basket else {}
count_values = dict(counts or {})

st.success("PostgreSQL connection is healthy.")

summary_1, summary_2, summary_3 = st.columns(3)
summary_1.metric("Basket", basket_values.get("basket_name", DEFAULT_BASKET_NAME))
summary_2.metric("Status", basket_values.get("status", "ACTIVE"))
summary_3.metric("Checked", now_india.strftime("%d %b %Y, %H:%M %Z"))

st.subheader("Basket metadata")
if basket_values:
    st.dataframe([basket_values], use_container_width=True, hide_index=True)
    if basket_created:
        st.caption("The basket metadata record was initialized during this page load.")
else:
    st.warning("The public basket record was not found.")

st.subheader("Ledger counts")
if count_values:
    count_columns = st.columns(min(len(count_values), 4))
    for index, (label, value) in enumerate(count_values.items()):
        readable_label = label.replace("_", " ").title()
        count_columns[index % len(count_columns)].metric(readable_label, value)
else:
    st.info("No ledger counts are available.")

show_table(
    "Recent signal runs",
    display_rows(signal_runs_raw),
    "No signal runs have been recorded.",
)
show_table(
    "Recent rebalance events",
    rebalance_events,
    "No rebalance events have been recorded.",
)

with st.expander("Latest orders and executions", expanded=False):
    if latest_rebalance_id:
        st.caption(f"Orders for latest rebalance: {latest_rebalance_id}")
    show_table(
        "Trade orders",
        display_rows(trade_orders_raw),
        "No trade orders are available.",
    )
    if latest_order_ids:
        st.caption(
            f"Executions across {len(latest_order_ids)} order(s) in the latest rebalance."
        )
    show_table(
        "Trade executions",
        display_rows(trade_executions_raw),
        "No trade executions are available.",
    )

with st.expander("Database details", expanded=False):
    database_time = health_values.get("database_time")
    st.json(
        {
            "database_name": health_values.get("database_name"),
            "database_time": display_value(database_time),
            "basket_id": DEFAULT_BASKET_ID,
            "display_timezone": "Asia/Kolkata",
            "recent_row_limit": RECENT_LIMIT,
        }
    )
