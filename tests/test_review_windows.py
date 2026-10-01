import json
import unittest
from pathlib import Path

from public_review.windows import estimate_review_window


class ReviewWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(
            (Path(__file__).resolve().parents[1] / "public_review_policy.json")
            .read_text(encoding="utf-8")
        )

    def test_mixed_market_review_waits_for_both_then_finishes_pre_nse(self):
        result = estimate_review_window(
            "2026-10-08",
            self.policy,
            {"A.NS": "equity", "ABBV": "foreign_us_listing"},
        )
        self.assertEqual(result["date_label"], "09 OCT")
        self.assertEqual(result["time_label"], "08:00-09:00 IST")
        self.assertEqual(result["market_context"], "Post NYSE | Pre NSE")
        self.assertTrue(result["execution_windows"]["NSE"].endswith("09:30:00+05:30"))
        self.assertTrue(result["execution_windows"]["NYSE"].endswith("19:15:00+05:30"))

    def test_single_market_is_not_delayed_for_an_absent_exchange(self):
        result = estimate_review_window(
            "2026-10-08", self.policy, {"A.NS": "equity"}
        )
        self.assertEqual(result["date_label"], "08 OCT")
        self.assertEqual(result["time_label"], "16:00-17:00 IST")
        self.assertEqual(set(result["execution_windows"]), {"NSE"})


if __name__ == "__main__":
    unittest.main()
