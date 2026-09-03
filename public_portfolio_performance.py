from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    connect_public_basket_db,
    get_public_basket_database_url,
)


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
        default=str,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@st.cache_data(ttl=300, show_spinner=False)
def load_public_history(database_url: str, basket_id: str) -> dict[str, Any]:
    """Read public records. This function never creates or changes data."""
    conn = connect_public_basket_db(database_url)
    try:
        basket = conn.execute(
            """
            SELECT *
            FROM public_baskets
            WHERE basket_id = %s
            """,
            (basket_id,),
        ).fetchone()

        signals = conn.execute(
            """
            SELECT
                signal_run_id,
                week_start_date,
                scheduled_session_date,
                generated_at,
                strategy_version,
                git_commit_sha,
                payload_sha256
            FROM signal_runs
            WHERE basket_id = %s
            ORDER BY week_start_date DESC
            """,
            (basket_id,),
        ).fetchall()

        rebalances = conn.execute(
            """
            SELECT
                rebalance_id,
                signal_run_id,
                created_at,
                effective_session_date,
                model_status,
                rationale,
                payload_sha256
            FROM rebalance_events
            WHERE basket_id = %s
            ORDER BY effective_session_date DESC
            """,
            (basket_id,),
        ).fetchall()

        orders = conn.execute(
            """
            SELECT
                o.order_id,
                o.rebalance_id,
                o.symbol,
                o.isin,
                o.action,
                o.current_weight,
                o.target_weight,
                o.signal_quantity,
                o.executable_quantity,
                o.signal_price,
                o.execution_rule,
                o.payload_sha256
            FROM trade_orders AS o
            JOIN rebalance_events AS r
              ON r.rebalance_id = o.rebalance_id
            WHERE r.basket_id = %s
            ORDER BY r.effective_session_date DESC, o.symbol
            """,
            (basket_id,),
        ).fetchall()

        executions = conn.execute(
            """
            SELECT
                x.execution_id,
                x.order_id,
                o.symbol,
                o.action,
                x.executed_at,
                x.quantity,
                x.market_price,
                x.slippage_bps,
                x.execution_price,
                x.fees_inr,
                x.cash_change_inr,
                x.payload_sha256
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
            SELECT DISTINCT ON (nav_date)
                nav_date,
                calculation_version,
                nav,
                portfolio_value,
                cash_value,
                input_sha256,
                calculated_at
            FROM daily_nav
            WHERE basket_id = %s
            ORDER BY nav_date, calculation_version DESC
            """,
            (basket_id,),
        ).fetchall()

        audit = conn.execute(
            """
            SELECT
                audit_id,
                event_at,
                entity_type,
                entity_id,
                event_type,
                payload_json,
                previous_hash,
                event_hash
            FROM public_basket_audit_log
            ORDER BY audit_id
            """
        ).fetchall()
    finally:
        conn.close()

    return {
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

        expected_hash = sha256_text(
            previous_hash + "|" + canonical_json(payload)
        )
        if expected_hash != row.get("event_hash"):
            return False, f"Hash mismatch at audit record {row['audit_id']}"

        previous_hash = row["event_hash"]

    if not rows:
        return False, "No audit records have been published yet"
    return True, f"All {len(rows):,} audit records link correctly"


def performance_summary(nav_frame: pd.DataFrame) -> dict[str, float | int | date]:
    frame = nav_frame.copy()
    frame["nav_date"] = pd.to_datetime(frame["nav_date"])
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")
    frame = frame.drop_duplicates("nav_date", keep="last")
    frame = frame[frame["nav"] > 0]

    if frame.empty:
        return {}

    start_nav = float(frame.iloc[0]["nav"])
    end_nav = float(frame.iloc[-1]["nav"])
    elapsed_days = int((frame.iloc[-1]["nav_date"] - frame.iloc[0]["nav_date"]).days)
    total_return = end_nav / start_nav - 1.0

    running_peak = frame["nav"].cummax()
    drawdown = frame["nav"] / running_peak - 1.0

    daily_returns = frame["nav"].pct_change().dropna()
    annual_volatility = (
        float(daily_returns.std(ddof=1) * np.sqrt(252))
        if len(daily_returns) >= 2
        else float("nan")
    )
    annual_return = (
        float((end_nav / start_nav) ** (365.25 / elapsed_days) - 1.0)
        if elapsed_days > 0
        else float("nan")
    )

    return {
        "start_date": frame.iloc[0]["nav_date"].date(),
        "end_date": frame.iloc[-1]["nav_date"].date(),
        "observations": int(len(frame)),
        "start_nav": start_nav,
        "end_nav": end_nav,
        "total_return": float(total_return),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": float(drawdown.min()),
    }


def json_download_payload(history: dict[str, Any]) -> bytes:
    return json.dumps(
        history,
        sort_keys=True,
        indent=2,
        default=str,
    ).encode("utf-8")


st.title("Public Portfolio Performance")
st.caption(
    "A plain-language record of what the model recommended, what was executed, "
    "and how the published portfolio changed over time. No uploads are required."
)

database_url = get_public_basket_database_url()
if not database_url:
    st.info(
        "Public history is not connected yet. Performance will appear after the "
        "durable public database is configured and verified records are published."
    )
    st.stop()

try:
    with st.spinner("Loading verified public records…"):
        history = load_public_history(database_url, DEFAULT_BASKET_ID)
except Exception:
    st.error(
        "The verified public record is temporarily unavailable. No performance "
        "figures are shown because unverified or partial data could be misleading."
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
    st.info(
        "There is not yet enough published NAV history to calculate performance. "
        "This page will not estimate or backfill results from private app activity."
    )
else:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Growth since public start", f"{summary['total_return']:.2%}")
    metric_2.metric("Latest portfolio index", f"{summary['end_nav']:,.2f}")
    metric_3.metric("Largest fall from a prior high", f"{summary['max_drawdown']:.2%}")
    metric_4.metric("Published trading days", f"{summary['observations']:,}")

    if np.isfinite(summary["annual_return"]):
        st.caption(
            f"Annualized growth: {summary['annual_return']:.2%}. This converts the "
            "observed period into a one-year rate; it is not a forecast."
        )

    chart = nav_frame.copy()
    chart["nav_date"] = pd.to_datetime(chart["nav_date"])
    chart["nav"] = pd.to_numeric(chart["nav"], errors="coerce")
    chart = chart.dropna(subset=["nav_date", "nav"]).sort_values("nav_date")
    st.line_chart(chart.set_index("nav_date")[["nav"]], y_label="Portfolio index")

    with st.expander("How to read these figures"):
        st.write(
            "Growth since public start compares the latest published portfolio index "
            "with its first value. The largest fall measures the worst decline from an "
            "earlier high. Annualized growth is shown only when dates permit it. "
            "Returns can be negative, and past performance does not predict future results."
        )

st.subheader("What has been published")
count_1, count_2, count_3, count_4 = st.columns(4)
count_1.metric("Official weekly signals", len(history["signals"]))
count_2.metric("Rebalance decisions", len(history["rebalances"]))
count_3.metric("Orders", len(history["orders"]))
count_4.metric("Recorded executions", len(history["executions"]))

latest_signal = history["signals"][0] if history["signals"] else None
if latest_signal:
    st.write(
        {
            "latest_official_week": str(latest_signal["week_start_date"]),
            "scheduled_nse_session": str(latest_signal["scheduled_session_date"]),
            "strategy_version": latest_signal["strategy_version"],
            "code_version": latest_signal.get("git_commit_sha") or "Not recorded",
            "signal_hash": latest_signal["payload_sha256"],
        }
    )
else:
    st.info("No official weekly signal has been published yet.")

st.subheader("Verification")
if audit_ok:
    st.success(audit_message)
else:
    st.warning(audit_message)

st.write(
    "Each official record carries a SHA-256 fingerprint. Changing a stored signal, "
    "decision, order, execution, or NAV input would change its fingerprint. The audit "
    "records are also linked in sequence so readers can detect a broken history."
)

st.download_button(
    "Download the public evidence bundle",
    data=json_download_payload(history),
    file_name=f"{DEFAULT_BASKET_ID.lower()}-public-evidence.json",
    mime="application/json",
    width="stretch",
)

with st.expander("How an independent reader can reproduce the record"):
    st.markdown(
        """
1. Download the evidence bundle above.
2. Confirm that every weekly signal points to the first actual NSE trading session of its calendar week.
3. Re-create each SHA-256 fingerprint from the canonical JSON fields.
4. Follow each signal into its rebalance decision, orders, and recorded executions.
5. Rebuild holdings and cash from executions rather than from later edited snapshots.
6. Recalculate each daily NAV using the published calculation version and input fingerprint.

The optimizer's mathematical rules belong to the versioned source code. A full
independent replication also needs the same code revision, settings, market-data
cutoff, prices, foreign-exchange inputs, and corporate-action treatment.
"""
    )

st.subheader("Important limits")
st.warning(
    "This is a model portfolio, not personal investment advice. Figures are shown "
    "only from durable public records. They may include costs and slippage only when "
    "those items are present in the execution ledger."
)

st.info(
    "Benchmark comparison is not shown yet because the current public database does "
    "not contain a versioned benchmark history. It should be added only after the "
    "benchmark, price source, adjustment rules, and calculation method are published."
)

with st.expander("Portfolio identity"):
    st.write(
        {
            "basket_id": basket["basket_id"],
            "basket_name": basket["basket_name"],
            "market": basket["calendar_market"],
            "weekly_rule": basket["rebalance_rule"],
            "strategy_version": basket["strategy_version"],
            "schema_version": basket["schema_version"],
        }
    )
