import unittest
import pandas as pd
from public_portfolio_trust import model_timestamp_utc, versioned_model_nav


class NavTimezoneTests(unittest.TestCase):
    def test_mixed_backfill_and_database_dates_rebalance(self):
        prices=pd.DataFrame({"A":[100,110,110],"B":[100,100,100]},
                            index=pd.date_range("2026-09-03",periods=3))
        publications=[
            {"publication_id":"new","as_of":"2026-09-04T12:00:00+00:00","weights":{"B":1}},
            {"publication_id":"backfill","as_of":prices.index[0].to_pydatetime(),"weights":{"A":1}},
        ]
        result=versioned_model_nav(prices,publications)
        self.assertEqual(result.publication_id.tolist(),["backfill","backfill","new"])
        self.assertAlmostEqual(result.iloc[-1]["nav"],110*(1-.0022))

    def test_equivalent_instants_and_exchange_midnight(self):
        self.assertEqual(model_timestamp_utc("2026-09-05T00:00:00+05:30"),
                         model_timestamp_utc("2026-09-04T18:30:00Z"))
        self.assertEqual(model_timestamp_utc("2026-09-05"),
                         model_timestamp_utc("2026-09-04T18:30:00Z"))

    def test_aware_price_index_has_identical_results(self):
        prices=pd.DataFrame({"A":[100,101]},index=pd.date_range("2026-09-03",periods=2))
        versions=[{"publication_id":"A","as_of":"2026-09-02T00:00:00Z","weights":{"A":1}}]
        expected=versioned_model_nav(prices,versions)
        prices.index=prices.index.tz_localize("Asia/Kolkata")
        actual=versioned_model_nav(prices,versions)
        self.assertEqual(expected.nav.tolist(),actual.nav.tolist())


if __name__=="__main__": unittest.main()
