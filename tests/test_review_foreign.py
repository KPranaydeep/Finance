import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from public_review.core import evaluate, freeze
from public_review.costs import charges, tax_rate
from public_review.market import AwaitingMarketEntry, synchronized_dates
from public_review.service import run
from public_us_funding import fx_gst
from review_fixtures import policy
from test_review_operations import FakeDB


class ForeignReviewTests(unittest.TestCase):
    def foreign_policy(self):
        result = policy()
        result["instrument_kinds"] = {"AXTI": "foreign_us_listing"}
        return result

    def test_tickertape_pro_sell_tariff_has_no_nse_charges(self):
        fee = charges(100_000, "SELL", "foreign_us_listing", self.foreign_policy())
        self.assertEqual(fee["dp"], 0)
        self.assertEqual(fee["stt"], 0)
        self.assertEqual(fee["stamp"], 0)
        self.assertGreater(fee["total"], 100_000 * (.0015 + .0002716))

    def test_foreign_tax_holding_period_is_more_than_24_months(self):
        p = self.foreign_policy()
        self.assertAlmostEqual(tax_rate("foreign_us_listing", "2026-05-01", "2028-05-01", p), .30 * 1.04)
        self.assertAlmostEqual(tax_rate("foreign_us_listing", "2026-05-01", "2028-05-02", p), .125 * 1.04)

    def test_baseline_and_exit_include_transfer_level_fx_gst(self):
        p = self.foreign_policy()
        publication = {"publication_id": "PUB-US", "basket_id": "B", "portfolio_version": 1,
                       "published_at": "2026-05-01T10:00:00+00:00"}
        baseline = freeze(publication, {"AXTI": 1.}, {"AXTI": 8_000.}, "2026-05-04",
                          100_000., p["instrument_kinds"], p, "2026-05-04T12:00:00+00:00")
        self.assertEqual(baseline["foreign_funding_fx_gst"], fx_gst(100_000))
        assessed = evaluate(baseline, {"AXTI": 9_000.}, "2026-09-09", p)
        self.assertGreater(assessed["rows"][0]["fees"]["fx_gst"], 0)
        self.assertEqual(assessed["exit_costs"], assessed["rows"][0]["fees"]["total"])

    def test_mixed_market_uses_latest_shared_completed_date(self):
        domestic = pd.DataFrame({"Open": [100., 101.], "Close": [100., 101.],
                                 "Volume": [1000., 1000.]},
                                index=["2026-09-10", "2026-09-11"])
        foreign = pd.DataFrame({"Open": [8000.], "Close": [8000.], "Volume": [1000.]},
                               index=["2026-09-10"])
        self.assertEqual(synchronized_dates({"A.NS": domestic, "AXTI": foreign},
                                            "2026-09-10", "2026-09-11", new_baseline=False),
                         ("2026-09-10", "2026-09-10"))

    def test_new_mixed_baseline_waits_for_post_publication_common_close(self):
        domestic = pd.DataFrame({"Close": [101.]}, index=["2026-09-11"])
        foreign = pd.DataFrame({"Close": [8000.]}, index=["2026-09-10"])
        with self.assertRaises(AwaitingMarketEntry):
            synchronized_dates({"A.NS": domestic, "AXTI": foreign},
                               "2026-09-11", "2026-09-11", new_baseline=True)

    def test_mixed_market_wait_is_successful_workflow_state(self):
        p = self.foreign_policy()
        p["instrument_kinds"]["A.NS"] = "equity"
        publication = {"publication_id": "PUB-MIXED", "basket_id": "B",
                       "portfolio_version": 1, "published_at": "2026-09-10T12:52:00+00:00",
                       "weights": {"A.NS": .5, "AXTI": .5}}
        histories = {
            "A.NS": pd.DataFrame({"Open": [100.], "Close": [101.], "Volume": [1000.]},
                                  index=["2026-09-11"]),
            "AXTI": pd.DataFrame({"Open": [8000.], "Close": [8000.], "Volume": [1000.]},
                                  index=["2026-09-10"]),
        }
        with patch("public_review.service.publications", return_value=[publication]), \
             patch("public_review.market.sessions", return_value=("2026-09-11", "2026-09-11", ["2026-09-14"])), \
             patch("public_review.market.fetch", return_value=histories):
            result = run(FakeDB(), "B", p, now=datetime(2026, 9, 11, 13, tzinfo=timezone.utc))
        self.assertEqual((result["failed"], result["waiting"]), (0, 1))
        self.assertEqual(result["results"][0]["status"], "WAITING")

    def test_hdfc_fx_gst_high_slab_is_capped(self):
        self.assertEqual(fx_gst(1_000_000), 990.)
        self.assertEqual(fx_gst(100_000_000), 10_800.)


if __name__ == "__main__":
    unittest.main()
