from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import public_basket_optimizer_adapter as adapter  # noqa: E402


def valid_input_payload() -> dict:
    return {
        "market_data_cutoff": "2026-01-27T16:00:00+05:30",
        "input_created_at": "2026-01-27T16:01:00+05:30",
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


class TestAdapterValidation:
    def test_data_as_of_must_be_timezone_aware(self):
        with pytest.raises(ValueError, match="timezone"):
            adapter._validated_data_as_of(datetime(2026, 1, 27, 10, 0))

    def test_invalid_start_mode_fails_closed(self):
        payload = valid_input_payload()
        payload["basket_start_mode"] = "weekly"
        with pytest.raises(ValueError, match="basket_start_mode"):
            adapter.build_public_signal(
                payload=payload,
                data_as_of=datetime(2026, 1, 27, 10, 0, tzinfo=timezone.utc),
            )

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
    def test_event_input_produces_complete_json_safe_contract(
        self, monkeypatch
    ):
        payload = valid_input_payload()

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
            payload=payload,
            data_as_of=datetime(2026, 1, 27, 10, 30, tzinfo=timezone.utc),
        )

        assert set(result) == {
            "strategy_version",
            "data_as_of",
            "settings",
            "portfolio_before",
            "optimizer_output",
            "signal_output",
            "decision_status",
        }
        assert result["decision_status"] == "REBALANCED"
        assert result["signal_output"][0]["Executable Quantity"] == 1
        json.dumps(result, allow_nan=False)

    def test_missing_latest_price_prevents_publication(self, monkeypatch):
        payload = valid_input_payload()
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

        with pytest.raises(RuntimeError, match="Missing valid market prices"):
            adapter.build_public_signal(
                payload=payload,
                data_as_of=datetime(2026, 1, 27, 10, 30, tzinfo=timezone.utc),
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
