from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from public_basket_optimizer_adapter import (
    STRATEGY_VERSION,
    build_public_signal,
    parse_public_basket_input,
)
from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    connect_public_basket_db,
    get_basket_record,
    get_public_basket_database_url,
)
from public_basket_rebalance_service import (
    RebalanceDecision,
    orders_from_rebalance_rows,
    run_public_basket_rebalance,
)


IST = ZoneInfo("Asia/Kolkata")
PUBLISH_LOCK = 7_104_202_601
PAGE_VERSION = "event-publisher-r1"


st.set_page_config(
    page_title="Public Basket Publisher",
    page_icon="🧾",
    layout="wide",
)


def secret_value(name: str) -> str:
    try:
        section = st.secrets["public_basket"]
        return str(section.get(name, "")).strip()
    except (KeyError, TypeError, AttributeError):
        return ""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def snapshot_hash(payload: dict[str, Any], data_as_of: datetime) -> str:
    material = {"payload": payload, "data_as_of": data_as_of.isoformat()}
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def safe_error(exc: Exception, *secrets: str) -> str:
    message = str(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return f"{type(exc).__name__}: {message}"


def clear_preview() -> None:
    for key in ("publisher_preview", "publisher_receipt"):
        st.session_state.pop(key, None)


st.title("🧾 Public Basket Publisher")
st.caption(f"Basket: {DEFAULT_BASKET_ID} · Timezone: Asia/Kolkata · {PAGE_VERSION}")
st.warning(
    "Operator-only page. Previewing does not write to PostgreSQL. Publishing creates "
    "immutable signal, rebalance, and model-order records only; it never executes trades."
)

publisher_token = secret_value("publisher_token")
if len(publisher_token) < 32:
    st.error(
        "Configure a private token of at least 32 characters at "
        "[public_basket].publisher_token in Streamlit secrets."
    )
    st.stop()

entered_token = st.text_input("Publisher token", type="password")
if not entered_token or not hmac.compare_digest(entered_token, publisher_token):
    st.info("Enter the private publisher token to continue.")
    st.stop()

database_url = get_public_basket_database_url()
if not database_url:
    st.error("The public basket PostgreSQL connection is not configured.")
    st.stop()

try:
    with connect_public_basket_db(database_url) as conn:
        basket = get_basket_record(conn, DEFAULT_BASKET_ID)
except Exception as exc:
    st.error("Could not verify the public basket database. No changes were made.")
    with st.expander("Technical details"):
        st.code(safe_error(exc, publisher_token, database_url), language="text")
    st.stop()

if basket is None:
    st.error(
        f"Basket {DEFAULT_BASKET_ID} does not exist. Complete the basket-ID rename before publishing."
    )
    st.stop()

uploaded = st.file_uploader("Public-basket input snapshot", type=["json"])
left, right = st.columns(2)
with left:
    as_of_date = st.date_input("Market data as of date", value="today")
with right:
    as_of_time = st.time_input("Market data as of time (IST)", value=time(15, 30))

if uploaded is None:
    clear_preview()
    st.info("Upload the JSON produced by your controlled basket-input preparation step.")
    st.stop()

raw_input = uploaded.getvalue()
try:
    input_payload = parse_public_basket_input(raw_input)
except Exception as exc:
    clear_preview()
    st.error(str(exc))
    st.stop()

data_as_of = datetime.combine(as_of_date, as_of_time, tzinfo=IST)
digest = snapshot_hash(input_payload, data_as_of)

if st.session_state.get("publisher_input_hash") != digest:
    clear_preview()
    st.session_state.publisher_input_hash = digest

st.write("Input fingerprint:", f"`{digest}`")

if st.button("Build read-only preview", type="primary"):
    try:
        with st.spinner("Running the optimizer without writing to the database…"):
            contract = build_public_signal(payload=input_payload, data_as_of=data_as_of)
            rows = contract.get("signal_output", [])
            if not isinstance(rows, list):
                raise TypeError("Optimizer signal_output must be a list of rows")
            orders = orders_from_rebalance_rows(rows)
            status = "REBALANCED" if orders else "NO_CHANGE"
            contract["decision_status"] = status
            st.session_state.publisher_preview = {
                "digest": digest,
                "data_as_of": data_as_of,
                "contract": contract,
                "orders": orders,
            }
            st.session_state.pop("publisher_receipt", None)
    except Exception as exc:
        clear_preview()
        st.error("Preview failed. Nothing was written to PostgreSQL.")
        with st.expander("Technical details"):
            st.code(safe_error(exc, publisher_token, database_url), language="text")

preview = st.session_state.get("publisher_preview")
if not preview:
    st.stop()

contract = preview["contract"]
orders = preview["orders"]
status = contract["decision_status"]

st.subheader("Preview")
c1, c2, c3 = st.columns(3)
c1.metric("Decision", status)
c2.metric("Model orders", len(orders))
c3.metric("Input fingerprint", digest[:12])

allocation = contract.get("optimizer_output", {}).get("target_allocation", [])
if allocation:
    st.markdown("#### Target allocation")
    st.dataframe(pd.DataFrame(allocation), use_container_width=True, hide_index=True)

if orders:
    st.markdown("#### Proposed model orders")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Symbol": order.symbol,
                    "Side": order.side,
                    "Quantity": order.requested_quantity,
                    "Reference price": order.reference_price,
                    "Current weight": order.current_weight,
                    "Target weight": order.target_weight,
                }
                for order in orders
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("The optimizer produced no executable model orders. Publishing records NO_CHANGE.")

with st.expander("Complete immutable preview"):
    st.json(contract, expanded=False)

if st.session_state.get("publisher_receipt"):
    st.success("This preview has already been published during the current session.")
    st.json(st.session_state.publisher_receipt)
    st.stop()

st.subheader("Publish immutable records")
rationale = st.text_input(
    "Operator rationale",
    value="Operator-approved event-driven optimizer result",
)
confirmation_phrase = f"PUBLISH {digest[:12].upper()}"
confirmation = st.text_input(
    f"Type exactly: {confirmation_phrase}",
    placeholder=confirmation_phrase,
)
confirmed = st.checkbox(
    "I reviewed the allocation and model orders and understand this appends immutable records."
)

if st.button(
    "Publish to public ledger",
    type="primary",
    disabled=not (confirmed and confirmation == confirmation_phrase and rationale.strip()),
):
    try:
        with connect_public_basket_db(database_url) as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (PUBLISH_LOCK,))
                duplicate = conn.execute(
                    """
                    SELECT signal_run_id
                    FROM signal_runs
                    WHERE basket_id = %s AND input_snapshot_sha256 = %s
                    LIMIT 1
                    """,
                    (DEFAULT_BASKET_ID, digest),
                ).fetchone()
                if duplicate:
                    raise RuntimeError(
                        "This exact input snapshot was already published as "
                        f"{duplicate['signal_run_id']}."
                    )

                decision = RebalanceDecision(
                    decision_status=status,
                    signal_output=contract.get("signal_output", []),
                    orders=orders,
                    rationale=rationale.strip(),
                    effective_at=preview["data_as_of"],
                )
                receipt = run_public_basket_rebalance(
                    conn=conn,
                    optimizer=lambda: contract.get("optimizer_output", {}),
                    optimizer_kwargs={},
                    decision_adapter=lambda _optimizer_output: decision,
                    portfolio_before=contract.get("portfolio_before", []),
                    settings=contract.get("settings", {}),
                    data_as_of=preview["data_as_of"],
                    basket_id=DEFAULT_BASKET_ID,
                    strategy_version=contract.get("strategy_version", STRATEGY_VERSION),
                    input_snapshot_sha256=digest,
                )

        result = {
            "decision_status": receipt.decision_status,
            "signal_run_id": receipt.signal_run_id,
            "rebalance_id": receipt.rebalance_id,
            "order_ids": list(receipt.order_ids),
        }
        st.session_state.publisher_receipt = result
        st.success("Published successfully. No trade executions were created.")
        st.json(result)
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception as exc:
        st.error("Publish failed. The transaction was rolled back.")
        with st.expander("Technical details"):
            st.code(safe_error(exc, publisher_token, database_url), language="text")

