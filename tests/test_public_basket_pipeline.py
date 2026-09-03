from __future__ import annotations

import importlib
import json
import os
import sys
import types
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from nse_market_calendar import (  # noqa: E402
    CALENDAR_YEAR,
    NSE_EQUITY_HOLIDAYS_2026,
    build_nse_equity_sessions_2026,
    first_nse_trading_day_of_week,
)
import public_basket_optimizer_adapter as adapter  # noqa: E402


def valid_input_payload(session_date: str = "2026-01-27") -> dict:
    return {
        "scheduled_session_date": session_date,
        "market_data_cutoff": f"{session_date}T16:00:00+05:30",
        "input_created_at": f"{session_date}T16:01:00+05:30",
        "settings": {},
        "portfolio": [
            {
                "Symbol": "ALPHA",
                "Yahoo Ticker": "ALPHA.NS",
                "Currency": "INR",
                "Quantity": 10,
                "Average Price": 100.0,
                "FX to INR": 1.0,
                "Weight": 0.50,
            },
            {
                "Symbol": "BETA",
                "Yahoo Ticker": "BETA.NS",
                "Currency": "INR",
                "Quantity": 5,
                "Average Price": 200.0,
                "FX to INR": 1.0,
                "Weight": 0.50,
            },
        ],
    }


def write_input(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "public-basket-input.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestNseCalendar:
    def test_calendar_contains_every_day_of_2026(self):
        rows = build_nse_equity_sessions_2026()
        assert len(rows) == 365
        assert rows[0]["session_date"] == f"{CALENDAR_YEAR}-01-01"
        assert rows[-1]["session_date"] == f"{CALENDAR_YEAR}-12-31"

    def test_ordinary_monday_is_first_session(self):
        monday = date(2026, 2, 2)
        assert first_nse_trading_day_of_week(monday) == monday

    def test_republic_day_monday_rolls_to_tuesday(self):
        monday_holiday = date(2026, 1, 26)
        assert monday_holiday in NSE_EQUITY_HOLIDAYS_2026
        assert first_nse_trading_day_of_week(monday_holiday) == date(2026, 1, 27)

    def test_ganesh_chaturthi_monday_rolls_to_tuesday(self):
        monday_holiday = date(2026, 9, 14)
        assert monday_holiday in NSE_EQUITY_HOLIDAYS_2026
        assert first_nse_trading_day_of_week(monday_holiday) == date(2026, 9, 15)

    def test_unsupported_year_fails_closed(self):
        with pytest.raises(ValueError, match="No verified NSE equity calendar"):
            first_nse_trading_day_of_week(date(2027, 1, 4))


class TestAdapterValidation:
    def test_missing_controlled_input_path_fails_closed(self, monkeypatch):
        monkeypatch.delenv(adapter.INPUT_PATH_ENV, raising=False)
        with pytest.raises(RuntimeError, match=adapter.INPUT_PATH_ENV):
            adapter._read_frozen_input()

    def test_stale_session_input_is_rejected(self):
        payload = valid_input_payload("2026-01-26")
        with pytest.raises(ValueError, match="does not match"):
            adapter._validate_effective_date(payload, date(2026, 1, 27))

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("Quantity", -1),
            ("Average Price", 0),
            ("FX to INR", 0),
            ("Weight", -0.01),
            ("Yahoo Ticker", ""),
        ],
    )
    def test_invalid_portfolio_values_are_rejected(self, column, value):
        payload = valid_input_payload()
        payload["portfolio"][0][column] = value
        with pytest.raises(ValueError, match="invalid"):
            adapter._validate_and_build_portfolio(payload)

    def test_duplicate_tickers_are_rejected(self):
        payload = valid_input_payload()
        payload["portfolio"][1]["Yahoo Ticker"] = "ALPHA.NS"
        with pytest.raises(ValueError, match="Duplicate Yahoo tickers"):
            adapter._validate_and_build_portfolio(payload)

    def test_zero_total_weight_is_rejected(self):
        payload = valid_input_payload()
        for row in payload["portfolio"]:
            row["Weight"] = 0
        with pytest.raises(ValueError, match="positive total"):
            adapter._validate_and_build_portfolio(payload)

    def test_unknown_setting_is_rejected(self):
        payload = valid_input_payload()
        payload["settings"] = {"new_unreviewed_knob": 1}
        with pytest.raises(ValueError, match="Unknown optimizer settings"):
            adapter._validated_settings(payload)

    def test_json_safe_removes_non_finite_numbers(self):
        converted = adapter._json_safe(
            {
                "numpy_integer": np.int64(3),
                "numpy_float": np.float64(1.5),
                "nan": np.nan,
                "positive_infinity": np.inf,
                "date": pd.Timestamp("2026-01-27"),
            }
        )
        assert converted["numpy_integer"] == 3
        assert converted["numpy_float"] == 1.5
        assert converted["nan"] is None
        assert converted["positive_infinity"] is None
        assert converted["date"].startswith("2026-01-27")
        json.dumps(converted, allow_nan=False)


class TestAdapterContract:
    def test_frozen_input_produces_complete_json_safe_contract(
        self, tmp_path, monkeypatch
    ):
        input_path = write_input(tmp_path, valid_input_payload())
        monkeypatch.setenv(adapter.INPUT_PATH_ENV, str(input_path))

        tickers = ["ALPHA.NS", "BETA.NS"]
        index = pd.date_range("2025-12-01", periods=30, freq="B")
        log_returns = pd.DataFrame(
            {
                "ALPHA.NS": np.linspace(0.001, 0.003, len(index)),
                "BETA.NS": np.linspace(0.002, -0.001, len(index)),
            },
            index=index,
        )

        def fake_analysis(symbols, current_alloc, **kwargs):
            assert symbols == tickers
            assert kwargs["max_dd"] == pytest.approx(-0.20)
            return (
                np.array([0.60, 0.40]),
                log_returns,
                {"Annual Return": np.float64(0.10)},
                {"Annual Return": np.float64(0.12)},
                {
                    "valid_start": index[0],
                    "valid_end": index[-1],
                    "dropped_df": pd.DataFrame(),
                    "redundant_df": pd.DataFrame(),
                },
            )

        def fake_prices(symbols):
            assert symbols == tickers
            return {"ALPHA.NS": 110.0, "BETA.NS": 210.0}

        def fake_rebalance(current_alloc, weights, returns, prices, days_to_flip):
            assert days_to_flip == 10
            return (
                pd.DataFrame(
                    [
                        {
                            "Symbol": "ALPHA",
                            "Yahoo Ticker": "ALPHA.NS",
                            "Action": "Buy",
                            "Executable Quantity": np.int64(1),
                        }
                    ]
                ),
                [],
                [],
            )

        monkeypatch.setattr(
            adapter,
            "_load_core_functions",
            lambda: (fake_analysis, fake_rebalance, fake_prices),
        )

        result = adapter.build_public_signal(
            scheduled_session_date=date(2026, 1, 27)
        )

        assert set(result) == {
            "strategy_version",
            "settings",
            "portfolio_before",
            "optimizer_output",
            "signal_output",
            "decision_status",
        }
        assert result["decision_status"] == "REBALANCED"
        assert result["signal_output"][0]["Executable Quantity"] == 1
        json.dumps(result, allow_nan=False)

    def test_missing_latest_price_prevents_publication(self, tmp_path, monkeypatch):
        input_path = write_input(tmp_path, valid_input_payload())
        monkeypatch.setenv(adapter.INPUT_PATH_ENV, str(input_path))
        index = pd.date_range("2025-12-01", periods=5, freq="B")
        returns = pd.DataFrame(
            {"ALPHA.NS": 0.001, "BETA.NS": 0.002}, index=index
        )

        monkeypatch.setattr(
            adapter,
            "_load_core_functions",
            lambda: (
                lambda *args, **kwargs: (
                    np.array([0.5, 0.5]),
                    returns,
                    {},
                    {},
                    {"valid_start": index[0], "valid_end": index[-1]},
                ),
                lambda *args, **kwargs: (pd.DataFrame(), [], []),
                lambda symbols: {"ALPHA.NS": 100.0},
            ),
        )

        with pytest.raises(RuntimeError, match="Missing valid scheduled-session prices"):
            adapter.build_public_signal(
                scheduled_session_date=date(2026, 1, 27)
            )


class TestCoreExtraction:
    def test_core_uses_original_function_bodies_without_running_ui(self):
        pytest.importorskip("streamlit")
        core = importlib.import_module("portfolio_optimizer_core")

        assert core.run_portfolio_analysis_multi.__code__.co_filename.endswith(
            "portfolio_rebalancer_database.py"
        )
        assert core.rebalance_plan_multi.__code__.co_filename.endswith(
            "portfolio_rebalancer_database.py"
        )
        assert core.get_latest_price_map.__code__.co_filename.endswith(
            "portfolio_rebalancer_database.py"
        )

        source = core._source_path().read_text(encoding="utf-8")
        assert core._ui_boundary(source) > max(
            core.run_portfolio_analysis_multi.__code__.co_firstlineno,
            core.rebalance_plan_multi.__code__.co_firstlineno,
            core.get_latest_price_map.__code__.co_firstlineno,
        )


TEST_DATABASE_ENV = "TEST_PUBLIC_BASKET_DATABASE_URL"


@pytest.mark.integration
def test_postgres_holiday_rollover_and_exactly_once_signal():
    """
    Opt-in integration test.

    It creates and removes an isolated schema. Never point
    TEST_PUBLIC_BASKET_DATABASE_URL at a production database.
    """
    database_url = os.getenv(TEST_DATABASE_ENV)
    if not database_url:
        pytest.skip(f"Set {TEST_DATABASE_ENV} to a dedicated test database")

    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql
    from psycopg.rows import dict_row

    from public_basket_postgres import (
        DEFAULT_BASKET_ID,
        create_public_basket,
        init_public_basket_schema,
        rebalance_gate,
        seed_nse_2026_calendar,
    )
    from public_basket_scheduler import store_official_signal

    schema_name = f"public_basket_test_{uuid.uuid4().hex}"
    conn = psycopg.connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    )

    try:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        conn.execute(
            sql.SQL("SET search_path TO {}, public").format(
                sql.Identifier(schema_name)
            )
        )

        init_public_basket_schema(conn)
        create_public_basket(conn)
        seed_nse_2026_calendar(conn)

        assert rebalance_gate(
            conn, DEFAULT_BASKET_ID, date(2026, 1, 26)
        )["status"] == "NOT_DUE"
        assert rebalance_gate(
            conn, DEFAULT_BASKET_ID, date(2026, 1, 27)
        )["status"] == "DUE"
        assert rebalance_gate(
            conn, DEFAULT_BASKET_ID, date(2026, 1, 28)
        )["status"] == "MISSED"

        optimizer_payload = {
            "strategy_version": "portfolio-rebalancer-v1-test",
            "settings": {"fixture": True},
            "portfolio_before": [{"symbol": "ALPHA", "quantity": 10}],
            "optimizer_output": {"target": {"ALPHA": 1.0}},
            "signal_output": [],
            "decision_status": "NO_CHANGE",
        }

        first = store_official_signal(
            conn,
            basket_id=DEFAULT_BASKET_ID,
            run_date=date(2026, 1, 27),
            optimizer_payload=optimizer_payload,
            commit_sha="test-commit",
        )
        second = store_official_signal(
            conn,
            basket_id=DEFAULT_BASKET_ID,
            run_date=date(2026, 1, 27),
            optimizer_payload=optimizer_payload,
            commit_sha="test-commit",
        )

        assert first["status"] == "NO_CHANGE"
        assert second["status"] == "ALREADY_EVALUATED"
        assert second["signal_run_id"] == first["signal_run_id"]

        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM signal_runs) AS signals,
                (SELECT COUNT(*) FROM weekly_rebalance_cycles) AS cycles
            """
        ).fetchone()
        assert counts["signals"] == 1
        assert counts["cycles"] == 1
    finally:
        conn.execute("RESET search_path")
        conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                sql.Identifier(schema_name)
            )
        )
        conn.close()
