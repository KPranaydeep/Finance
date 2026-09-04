from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

try:
    import pandas as pd
    import public_basket_optimizer_adapter as adapter
except ModuleNotFoundError:
    pd = None
    adapter = None


@unittest.skipIf(adapter is None, "pandas/numpy are installed by the application requirements")
class EventDrivenOptimizerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "portfolio": [
                {
                    "Symbol": "ABC",
                    "Yahoo Ticker": "ABC.NS",
                    "Currency": "INR",
                    "Quantity": 10,
                    "Average Price": 100,
                    "FX to INR": 1,
                    "Weight": 1,
                }
            ],
            "settings": {},
            "market_data_cutoff": "2026-09-04T15:30:00+05:30",
        }

    @staticmethod
    def fake_core():
        def analyze(*args, **kwargs):
            history = pd.DataFrame(
                {"ABC.NS": [0.01, -0.01]},
                index=pd.date_range("2026-09-01", periods=2),
            )
            meta = {
                "valid_start": history.index.min(),
                "valid_end": history.index.max(),
                "dropped_df": pd.DataFrame(),
                "redundant_df": pd.DataFrame(),
            }
            return [1.0], history, {"Annual Return": 0.1}, {"Annual Return": 0.2}, meta

        def rebalance(*args, **kwargs):
            return (
                pd.DataFrame(
                    [
                        {
                            "Symbol": "ABC",
                            "Yahoo Ticker": "ABC.NS",
                            "Side": "BUY",
                            "Executable Quantity": 2,
                            "Latest Price": 110,
                        }
                    ]
                ),
                [],
                [],
            )

        def prices(_tickers):
            return {"ABC.NS": 110.0}

        return analyze, rebalance, prices

    def test_builds_event_driven_contract_without_scheduler_fields(self):
        as_of = datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)
        with patch.object(adapter, "_load_core_functions", self.fake_core):
            result = adapter.build_public_signal(payload=self.payload, data_as_of=as_of)

        self.assertEqual(result["data_as_of"], as_of.isoformat())
        self.assertEqual(result["decision_status"], "REBALANCED")
        self.assertEqual(result["signal_output"][0]["Executable Quantity"], 2)
        self.assertNotIn("scheduled_session_date", result["settings"])

    def test_rejects_naive_data_as_of(self):
        with self.assertRaisesRegex(ValueError, "data_as_of must include a timezone"):
            adapter.build_public_signal(
                payload=self.payload,
                data_as_of=datetime(2026, 9, 4, 10, 0),
            )

    def test_parses_only_json_objects(self):
        self.assertEqual(adapter.parse_public_basket_input(b'{"portfolio": []}'), {"portfolio": []})
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            adapter.parse_public_basket_input("[]")


if __name__ == "__main__":
    unittest.main()
