from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    connect_public_basket_db,
    get_public_basket_database_url,
)


INDIA_TZ = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger(__name__)


st.set_page_config(
    page_title="Public Portfolio Performance",
    page_icon="📈",
    layout="wide",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def display_value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=INDIA_TZ)
        return value.astimezone(INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return value


def display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: display_value(value) for key, value in row.items()}
        for row in rows
    ]


def detect_schema(conn: Any) -> str:
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'public_baskets'
        """
    ).fetchall()
    columns = {row["column_name"] for row in rows}
    if {"base_currency", "status"}.issubset(columns) and "rebalance_rule" not in columns:
        return "V2"
    if {"calendar_market", "rebalance_rule"}.issubset(columns):
        return "V1"
    return "UNKNOWN"


@st.cache_data(ttl=300, show_spinner=False)
def load_public_history(basket_id: str) -> dict[str, Any]:
    database_url = get_public_basket_database_url()
    if not database_url:
        raise RuntimeError("Public PostgreSQL is not configured")

    conn = connect_public_basket_db(database_url)
    try:
        schema = detect_schema(conn)
        if schema == "UNKNOWN":
            raise RuntimeError("Unrecognized public-basket schema")
        if schema == "V1":
            basket = conn.execute(
                """
                SELECT basket_id, basket_name, strategy_version, schema_version, created_at
                FROM public_baskets
                WHERE basket_id = %s
                """,
                (basket_id,),
            ).fetchone()
            return {
                "schema": "V1",
                "basket": dict(basket) if basket else None,
                "signals": [],
                "rebalances": [],
                "orders": [],
                "executions": [],
                "nav": [],
                "audit": [],
            }

        basket = conn.execute(
            """
            SELECT basket_id, basket_name, base_currency, strategy_version,
                   schema_version, created_at, status
            FROM public_baskets
            WHERE basket_id = %s
            """,
            (basket_id,),
        ).fetchone()

        signals = conn.execute(
            """
            SELECT signal_run_id, basket_id, generated_at, data_as_of,
                   strategy_version, git_commit_sha, input_snapshot_sha256,
                   settings_json, portfolio_before_json, optimizer_output_json,
                   signal_output_json, decision_status, payload_sha256, created_at
            FROM signal_runs
            WHERE basket_id = %s
            ORDER BY generated_at DESC
            """,
            (basket_id,),
        ).fetchall()

        rebalances = conn.execute(
            """
            SELECT rebalance_id, basket_id, signal_run_id, created_at, effective_at,
                   status, rationale, payload_json, payload_sha256
            FROM rebalance_events
            WHERE basket_id = %s
            ORDER BY created_at DESC
            """,
            (basket_id,),
        ).fetchall()

        orders = conn.execute(
            """
            SELECT o.order_id, o.rebalance_id, o.created_at, o.symbol,
                   o.yahoo_ticker, o.isin, o.side, o.current_weight,
                   o.target_weight, o.theoretical_quantity, o.requested_quantity,
                   o.reference_price, o.execution_rule, o.order_status,
                   o.payload_json, o.payload_sha256
            FROM trade_orders AS o
            JOIN rebalance_events AS r ON r.rebalance_id = o.rebalance_id
            WHERE r.basket_id = %s
            ORDER BY o.created_at DESC, o.symbol
            """,
            (basket_id,),
        ).fetchall()

        executions = conn.execute(
            """
            SELECT x.execution_id, x.order_id, o.symbol, o.side, x.executed_at,
                   x.quantity, x.market_price, x.execution_price, x.fees_inr,
                   x.taxes_inr, x.slippage_bps, x.cash_change_inr,
                   x.payload_json, x.payload_sha256
            FROM trade_executions AS x
            JOIN trade_orders AS o ON o.order_id = x.order_id
            JOIN rebalance_events AS r ON r.rebalance_id = o.rebalance_id
            WHERE r.basket_id = %s
            ORDER BY x.executed_at DESC
            """,
            (basket_id,),
        ).fetchall()

        nav = conn.execute(
            """
            SELECT DISTINCT ON (nav_date) nav_date, calculation_version, nav,
                   portfolio_value, cash_value, total_value, daily_return,
                   drawdown, input_sha256, calculated_at
            FROM daily_nav
            WHERE basket_id = %s
            ORDER BY nav_date, calculation_version DESC
            """,
            (basket_id,),
        ).fetchall()

        audit = conn.execute(
            """
            SELECT audit_id, event_at, entity_type, entity_id, event_type,
                   payload_json, previous_hash, event_hash
            FROM public_basket_audit_log
            ORDER BY audit_id
            """
        ).fetchall()
    finally:
        conn.close()

    return {
        "schema": "V2",
        "basket": dict(basket) if basket else None,
        "signals": [dict(row) for row in signals],
        "rebalances": [dict(row) for row in rebalances],
        "orders": [dict(row) for row in orders],
        "executions": [dict(row) for row in executions],
        "nav": [dict(row) for row in nav],
        "audit": [dict(row) for row in audit],
    }


def verify_audit_chain(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    previous_hash = ""
    for row in rows:
        stored_previous = row.get("previous_hash") or ""
        if stored_previous != previous_hash:
            return False, f"Broken link at audit record {row['audit_id']}"
        payload = row.get("payload_json")
        if isinstance(payload, str):
            payload = json.loads(payload)
        expected_hash = sha256_text(previous_hash + "|" + canonical_json(payload))
        if expected_hash != row.get("event_hash"):
            return False, f"Hash mismatch at audit record {row['audit_id']}"
        previous_hash = row["event_hash"]
    if not rows:
        return False, "No audit records have been published yet"
    return True, f"All {len(rows):,} audit records link correctly"


def performance_summary(nav_frame: pd.DataFrame) -> dict[str, Any]:
    frame = nav_frame.copy()
    frame["nav_date"] = pd.to_datetime(frame["nav_date"], errors="coerce")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")
    frame = frame.drop_duplicates("nav_date", keep="last")
    frame = frame[frame["nav"] > 0]
    if frame.empty:
        return {}

    start_nav = float(frame.iloc[0]["nav"])
    end_nav = float(frame.iloc[-1]["nav"])
    elapsed_days = int((frame.iloc[-1]["nav_date"] - frame.iloc[0]["nav_date"]).days)
    daily_returns = frame["nav"].pct_change().dropna()
    running_peak = frame["nav"].cummax()
    drawdown = frame["nav"] / running_peak - 1.0
    return {
        "start_date": frame.iloc[0]["nav_date"].date(),
        "end_date": frame.iloc[-1]["nav_date"].date(),
        "observations": int(len(frame)),
        "start_nav": start_nav,
        "end_nav": end_nav,
        "total_return": float(end_nav / start_nav - 1.0),
        "annual_return": (
            float((end_nav / start_nav) ** (365.25 / elapsed_days) - 1.0)
            if elapsed_days > 0 else float("nan")
        ),
        "annual_volatility": (
            float(daily_returns.std(ddof=1) * np.sqrt(252))
            if len(daily_returns) >= 2 else float("nan")
        ),
        "max_drawdown": float(drawdown.min()),
    }


def evidence_bundle(history: dict[str, Any]) -> bytes:
    return json.dumps(history, sort_keys=True, indent=2, default=str).encode("utf-8")


def show_recent(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    st.subheader(title)
    if not rows:
        st.info(f"No {title.lower()} have been published.")
        return
    selected = [{column: row.get(column) for column in columns} for row in rows[:25]]
    st.dataframe(display_rows(selected), use_container_width=True, hide_index=True)


st.title("Public Portfolio Performance")
st.caption(
    "A public record of optimizer signals, rebalance decisions, model orders, "
    "recorded executions, and portfolio performance. No uploads are required."
)

if not get_public_basket_database_url():
    st.info("Public history will appear after durable PostgreSQL is configured.")
    st.stop()

try:
    with st.spinner("Loading verified public records…"):
        history = load_public_history(DEFAULT_BASKET_ID)
except Exception:
    LOGGER.exception("Unable to load public portfolio history")
    st.error(
        "The verified public record is temporarily unavailable. No figures are "
        "shown because partial or unverified results could be misleading."
    )
    st.stop()

if history["schema"] == "V1":
    st.warning(
        "The public ledger is undergoing a version-2 migration. Performance and "
        "event details will appear after the migration is verified."
    )
    st.stop()

basket = history["basket"]
if basket is None:
    st.info("The public portfolio has not been initialized yet.")
    st.stop()

nav_frame = pd.DataFrame(history["nav"])
summary = performance_summary(nav_frame) if not nav_frame.empty else {}
audit_ok, audit_message = verify_audit_chain(history["audit"])

st.subheader("At a glance")
if not summary:
    st.info("There is not yet enough published NAV history to calculate performance.")
else:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Growth since public start", f"{summary['total_return']:.2%}")
    metric_2.metric("Latest portfolio index", f"{summary['end_nav']:,.2f}")
    metric_3.metric("Largest fall", f"{summary['max_drawdown']:.2%}")
    metric_4.metric("Published days", f"{summary['observations']:,}")

    if np.isfinite(summary["annual_return"]):
        st.caption(
            f"Annualized growth: {summary['annual_return']:.2%}. This is not a forecast."
        )
    chart = nav_frame.copy()
    chart["nav_date"] = pd.to_datetime(chart["nav_date"], errors="coerce")
    chart["nav"] = pd.to_numeric(chart["nav"], errors="coerce")
    chart = chart.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")
    st.line_chart(chart.set_index("nav_date")[["nav"]], y_label="Portfolio index")

st.subheader("Published ledger")
count_1, count_2, count_3, count_4 = st.columns(4)
count_1.metric("Signal runs", len(history["signals"]))
count_2.metric("Rebalance events", len(history["rebalances"]))
count_3.metric("Orders", len(history["orders"]))
count_4.metric("Executions", len(history["executions"]))

show_recent(
    "Recent signal runs",
    history["signals"],
    ["signal_run_id", "generated_at", "data_as_of", "decision_status", "strategy_version", "payload_sha256"],
)
show_recent(
    "Recent rebalance events",
    history["rebalances"],
    ["rebalance_id", "signal_run_id", "created_at", "effective_at", "status", "rationale", "payload_sha256"],
)

with st.expander("Orders and executions", expanded=False):
    show_recent(
        "Model orders",
        history["orders"],
        ["order_id", "created_at", "symbol", "side", "requested_quantity", "reference_price", "order_status"],
    )
    show_recent(
        "Recorded executions",
        history["executions"],
        ["execution_id", "order_id", "executed_at", "symbol", "side", "quantity", "execution_price", "fees_inr", "taxes_inr"],
    )

st.subheader("Verification")
if audit_ok:
    st.success(audit_message)
else:
    st.warning(audit_message)
st.write(
    "Every ledger record carries a SHA-256 fingerprint, and audit records are "
    "linked in sequence so a broken public history can be detected."
)

st.download_button(
    "Download the public evidence bundle",
    data=evidence_bundle(history),
    file_name=f"{DEFAULT_BASKET_ID.lower()}-public-evidence-v2.json",
    mime="application/json",
    use_container_width=True,
)

st.subheader("Important limits")
st.warning(
    "This is a model portfolio, not personal investment advice. Returns can be "
    "negative, and past performance does not predict future results."
)
st.info(
    "Benchmark comparison is intentionally omitted until a versioned benchmark "
    "series and its calculation rules are stored in the public ledger."
)

with st.expander("Portfolio identity"):
    st.write(
        {
            "basket_id": basket["basket_id"],
            "basket_name": basket["basket_name"],
            "base_currency": basket["base_currency"],
            "status": basket["status"],
            "strategy_version": basket["strategy_version"],
            "schema_version": basket["schema_version"],
            "display_timezone": "Asia/Kolkata",
        }
    )
