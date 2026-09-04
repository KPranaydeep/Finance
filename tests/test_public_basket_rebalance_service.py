from __future__ import annotations

import math
import sys
import types
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

try:
    import psycopg  # noqa: F401
except ModuleNotFoundError:
    backend = types.ModuleType("public_basket_postgres")
    backend.DEFAULT_BASKET_ID = "PUBLIC-01"
    backend.DEFAULT_BASKET_NAME = "Public Dynamic Portfolio"
    backend.DEFAULT_STRATEGY_VERSION = "portfolio-rebalancer-v1"
    for name in (
        "connect_public_basket_db",
        "create_public_basket",
        "create_rebalance_event",
        "create_trade_order",
        "init_public_basket_schema",
        "record_signal_run",
    ):
        setattr(backend, name, lambda *args, **kwargs: None)
    sys.modules["public_basket_postgres"] = backend

import public_basket_rebalance_service as service


class FakeConnection:
    def transaction(self):
        return nullcontext()


class PublicBasketRebalanceServiceTests(unittest.TestCase):
    def test_no_change_records_only_signal(self):
        captured = {}

        def record_signal_run(conn, basket_id, strategy_version, settings,
                              portfolio_before, optimizer_output, signal_output,
                              decision_status, **kwargs):
            captured.update(
                basket_id=basket_id,
                signal_output=signal_output,
                decision_status=decision_status,
            )
            return "SIG-1"

        def adapter(_output):
            return service.RebalanceDecision(
                decision_status="NO_CHANGE",
                signal_output={"reason": "within tolerance"},
            )

        with patch.object(service, "record_signal_run", record_signal_run):
            receipt = service.run_public_basket_rebalance(
                conn=FakeConnection(),
                optimizer=lambda: {"weights": {"ABC": 1.0}},
                optimizer_kwargs={},
                decision_adapter=adapter,
                portfolio_before=[],
                settings={},
            )

        self.assertEqual(receipt.signal_run_id, "SIG-1")
        self.assertIsNone(receipt.rebalance_id)
        self.assertEqual(receipt.order_ids, ())
        self.assertEqual(captured["basket_id"], "PUBLIC-01")
        self.assertEqual(captured["decision_status"], "NO_CHANGE")

    def test_rebalance_normalizes_numeric_values_and_retains_metadata(self):
        captured = {"orders": []}

        def record_signal_run(conn, basket_id, strategy_version, settings,
                              portfolio_before, optimizer_output, signal_output,
                              decision_status, **kwargs):
            captured["signal_output"] = signal_output
            return "SIG-2"

        def create_rebalance_event(conn, basket_id, signal_run_id, *, rationale=None,
                                   effective_at=None, status="CREATED"):
            captured["effective_at"] = effective_at
            return "REB-1"

        def create_trade_order(
            conn,
            rebalance_id,
            symbol,
            side,
            requested_quantity,
            *,
            yahoo_ticker=None,
            isin=None,
            current_weight=None,
            target_weight=None,
            theoretical_quantity=None,
            reference_price=None,
            execution_rule=None,
        ):
            captured["orders"].append(
                {"quantity": requested_quantity, "reference_price": reference_price}
            )
            return "ORD-1"

        effective_at = datetime(2026, 9, 4, 10, 30, tzinfo=timezone.utc)

        def adapter(_output):
            return service.RebalanceDecision(
                decision_status="REBALANCED",
                signal_output={"action": "rebalance"},
                rationale="target weights changed",
                effective_at=effective_at,
                rebalance_payload={"model": "v1"},
                orders=(
                    service.TradeOrderDraft(
                        symbol="abc",
                        side="buy",
                        requested_quantity=Decimal("2"),
                        reference_price=Decimal("123.45"),
                        payload={"source_row": 7},
                    ),
                ),
            )

        with (
            patch.object(service, "record_signal_run", record_signal_run),
            patch.object(service, "create_rebalance_event", create_rebalance_event),
            patch.object(service, "create_trade_order", create_trade_order),
        ):
            receipt = service.run_public_basket_rebalance(
                conn=FakeConnection(),
                optimizer=lambda: {"ok": True},
                optimizer_kwargs={},
                decision_adapter=adapter,
                portfolio_before=[],
                settings={},
            )

        self.assertEqual(receipt.order_ids, ("ORD-1",))
        self.assertIs(type(captured["orders"][0]["quantity"]), float)
        self.assertIs(type(captured["orders"][0]["reference_price"]), float)
        self.assertEqual(captured["effective_at"], effective_at)
        metadata = captured["signal_output"]["ledger_metadata"]
        self.assertEqual(metadata["rebalance"], {"model": "v1"})
        self.assertEqual(metadata["orders"], [{"source_row": 7}])

    def test_naive_data_as_of_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "data_as_of must include a timezone"):
            service.run_public_basket_rebalance(
                conn=FakeConnection(),
                optimizer=lambda: {},
                optimizer_kwargs={},
                decision_adapter=lambda _output: service.RebalanceDecision(
                    decision_status="NO_CHANGE",
                    signal_output={},
                ),
                portfolio_before=[],
                settings={},
                data_as_of=datetime(2026, 9, 4, 10, 30),
            )

    def test_non_finite_optional_order_value_is_rejected(self):
        decision = service.RebalanceDecision(
            decision_status="REBALANCED",
            signal_output={},
            orders=(
                service.TradeOrderDraft(
                    symbol="ABC",
                    side="BUY",
                    requested_quantity=1,
                    reference_price=math.nan,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "reference_price for ABC must be finite"):
            service._validate_decision(decision)


if __name__ == "__main__":
    unittest.main()
