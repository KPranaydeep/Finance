from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from public_basket_postgres import (
    DEFAULT_BASKET_ID,
    DEFAULT_BASKET_NAME,
    DEFAULT_STRATEGY_VERSION,
    connect_public_basket_db,
    create_public_basket,
    create_rebalance_event,
    create_trade_order,
    init_public_basket_schema,
    record_signal_run,
)


VALID_DECISIONS = {"REBALANCED", "NO_CHANGE"}
VALID_SIDES = {"BUY", "SELL"}


@dataclass(frozen=True)
class TradeOrderDraft:
    symbol: str
    side: str
    requested_quantity: float
    current_weight: float | None = None
    target_weight: float | None = None
    theoretical_quantity: float | None = None
    reference_price: float | None = None
    yahoo_ticker: str | None = None
    isin: str | None = None
    execution_rule: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RebalanceDecision:
    decision_status: str
    signal_output: Mapping[str, Any] | Sequence[Any]
    orders: Sequence[TradeOrderDraft] = field(default_factory=tuple)
    rationale: str | None = None
    effective_at: datetime | None = None
    rebalance_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RebalanceReceipt:
    decision_status: str
    signal_run_id: str
    rebalance_id: str | None
    order_ids: tuple[str, ...]


def _json_safe(value: Any) -> Any:
    """Convert common optimizer values into deterministic JSON-safe objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(orient="records"))
        except TypeError:
            return _json_safe(value.to_dict())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Optimizer output contains NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _call_backend(function: Callable[..., Any], conn: Any, **values: Any) -> Any:
    """Call a backend function without coupling to optional argument additions."""
    parameters = inspect.signature(function).parameters
    connection_name = next(
        (name for name in ("conn", "connection") if name in parameters),
        None,
    )
    if connection_name is None:
        raise TypeError(f"{function.__name__} has no connection parameter")

    arguments = {connection_name: conn}
    arguments.update(
        {
            name: value
            for name, value in values.items()
            if name in parameters
        }
    )

    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in arguments
    ]
    if missing:
        raise TypeError(
            f"Unsupported {function.__name__} backend signature; missing values: "
            + ", ".join(missing)
        )
    return function(**arguments)


def _validate_decision(decision: RebalanceDecision) -> RebalanceDecision:
    status = decision.decision_status.strip().upper()
    if status not in VALID_DECISIONS:
        raise ValueError("decision_status must be REBALANCED or NO_CHANGE")
    if status == "NO_CHANGE" and decision.orders:
        raise ValueError("NO_CHANGE cannot contain trade orders")
    if status == "REBALANCED" and not decision.orders:
        raise ValueError("REBALANCED must contain at least one trade order")

    seen: set[tuple[str, str]] = set()
    for order in decision.orders:
        symbol = order.symbol.strip().upper()
        side = order.side.strip().upper()
        if not symbol:
            raise ValueError("Every order requires a symbol")
        if side not in VALID_SIDES:
            raise ValueError(f"Invalid side for {symbol}: {side}")
        if not math.isfinite(order.requested_quantity) or order.requested_quantity <= 0:
            raise ValueError(f"Requested quantity must be positive for {symbol}")
        key = (symbol, side)
        if key in seen:
            raise ValueError(f"Duplicate {side} order for {symbol}")
        seen.add(key)

    return RebalanceDecision(
        decision_status=status,
        signal_output=decision.signal_output,
        orders=decision.orders,
        rationale=decision.rationale,
        effective_at=decision.effective_at,
        rebalance_payload=decision.rebalance_payload,
    )


def run_public_basket_rebalance(
    *,
    conn: Any,
    optimizer: Callable[..., Any],
    optimizer_kwargs: Mapping[str, Any],
    decision_adapter: Callable[[Any], RebalanceDecision],
    portfolio_before: Mapping[str, Any] | Sequence[Any],
    settings: Mapping[str, Any],
    data_as_of: datetime | None = None,
    basket_id: str = DEFAULT_BASKET_ID,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    git_commit_sha: str | None = None,
    input_snapshot_sha256: str | None = None,
) -> RebalanceReceipt:
    """Run one optimizer evaluation and atomically append its ledger records.

    This function never records executions or changes cash/positions. The caller
    supplies a decision adapter because optimizer return formats are strategy-specific.
    """
    if not callable(optimizer) or not callable(decision_adapter):
        raise TypeError("optimizer and decision_adapter must be callable")

    optimizer_output_raw = optimizer(**dict(optimizer_kwargs))
    decision = _validate_decision(decision_adapter(optimizer_output_raw))

    optimizer_output = _json_safe(optimizer_output_raw)
    signal_output = _json_safe(decision.signal_output)
    portfolio_snapshot = _json_safe(portfolio_before)
    settings_snapshot = _json_safe(settings)
    generated_for = data_as_of or datetime.now(timezone.utc)
    effective_at = decision.effective_at or generated_for

    with conn.transaction():
        signal_run_id = _call_backend(
            record_signal_run,
            conn,
            basket_id=basket_id,
            data_as_of=generated_for,
            strategy_version=strategy_version,
            git_commit_sha=git_commit_sha,
            input_snapshot_sha256=input_snapshot_sha256,
            settings=settings_snapshot,
            portfolio_before=portfolio_snapshot,
            optimizer_output=optimizer_output,
            signal_output=signal_output,
            decision_status=decision.decision_status,
        )

        if decision.decision_status == "NO_CHANGE":
            return RebalanceReceipt("NO_CHANGE", signal_run_id, None, ())

        rebalance_id = _call_backend(
            create_rebalance_event,
            conn,
            basket_id=basket_id,
            signal_run_id=signal_run_id,
            effective_at=effective_at,
            status="CREATED",
            rationale=decision.rationale,
            payload=_json_safe(decision.rebalance_payload),
            payload_json=_json_safe(decision.rebalance_payload),
        )

        order_ids: list[str] = []
        for draft in decision.orders:
            order_payload = _json_safe(draft.payload)
            order_ids.append(
                _call_backend(
                    create_trade_order,
                    conn,
                    rebalance_id=rebalance_id,
                    symbol=draft.symbol.strip().upper(),
                    yahoo_ticker=draft.yahoo_ticker,
                    isin=draft.isin,
                    side=draft.side.strip().upper(),
                    current_weight=draft.current_weight,
                    target_weight=draft.target_weight,
                    theoretical_quantity=draft.theoretical_quantity,
                    requested_quantity=float(draft.requested_quantity),
                    reference_price=draft.reference_price,
                    execution_rule=draft.execution_rule,
                    payload=order_payload,
                    payload_json=order_payload,
                )
            )

    return RebalanceReceipt(
        decision_status="REBALANCED",
        signal_run_id=signal_run_id,
        rebalance_id=rebalance_id,
        order_ids=tuple(order_ids),
    )


def initialize_and_run_public_basket_rebalance(
    *,
    optimizer: Callable[..., Any],
    optimizer_kwargs: Mapping[str, Any],
    decision_adapter: Callable[[Any], RebalanceDecision],
    portfolio_before: Mapping[str, Any] | Sequence[Any],
    settings: Mapping[str, Any],
    database_url: str | None = None,
    basket_id: str = DEFAULT_BASKET_ID,
    basket_name: str = DEFAULT_BASKET_NAME,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    data_as_of: datetime | None = None,
    git_commit_sha: str | None = None,
    input_snapshot_sha256: str | None = None,
) -> RebalanceReceipt:
    """Initialize durable storage, run once, and close the connection."""
    conn = connect_public_basket_db(database_url)
    try:
        init_public_basket_schema(conn)
        create_public_basket(
            conn=conn,
            basket_id=basket_id,
            basket_name=basket_name,
            strategy_version=strategy_version,
        )
        return run_public_basket_rebalance(
            conn=conn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            decision_adapter=decision_adapter,
            portfolio_before=portfolio_before,
            settings=settings,
            data_as_of=data_as_of,
            basket_id=basket_id,
            strategy_version=strategy_version,
            git_commit_sha=git_commit_sha,
            input_snapshot_sha256=input_snapshot_sha256,
        )
    finally:
        conn.close()


def orders_from_rebalance_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[TradeOrderDraft, ...]:
    """Adapt the existing rebalancing-plan rows into immutable order drafts."""
    orders: list[TradeOrderDraft] = []
    for row in rows:
        quantity = row.get("Executable Quantity", row.get("requested_quantity", 0))
        quantity = float(quantity or 0)
        if quantity == 0:
            continue
        explicit_side = str(row.get("Side", row.get("Action", ""))).strip().upper()
        side = explicit_side if explicit_side in VALID_SIDES else ("BUY" if quantity > 0 else "SELL")
        orders.append(
            TradeOrderDraft(
                symbol=str(row.get("Symbol", row.get("symbol", ""))),
                yahoo_ticker=row.get("Yahoo Ticker", row.get("yahoo_ticker")),
                isin=row.get("ISIN", row.get("isin")),
                side=side,
                current_weight=row.get("Current Weight", row.get("current_weight")),
                target_weight=row.get("Target Weight", row.get("target_weight")),
                theoretical_quantity=row.get(
                    "Theoretical Quantity", row.get("theoretical_quantity")
                ),
                requested_quantity=abs(quantity),
                reference_price=row.get("Latest Price", row.get("reference_price")),
                execution_rule=row.get("Execution Rule", row.get("execution_rule")),
                payload=_json_safe(row),
            )
        )
    return tuple(orders)
