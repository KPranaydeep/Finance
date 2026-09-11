import unittest
import pandas as pd
from public_world_benchmark import compare_world_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_inr_conversion_and_common_base(self):
        dates=pd.date_range("2026-01-01",periods=3)
        nav=[dict(nav_date=d,nav=v) for d,v in zip(dates,[100,105,110])]
        result=compare_world_benchmark(nav,pd.Series([10,11,12],index=dates),
                                      pd.Series([80,80,88],index=dates))
        self.assertEqual(result.iloc[0].tolist(),[100,100])
        self.assertAlmostEqual(result.iloc[-1,0],110)
        self.assertAlmostEqual(result.iloc[-1,1],132)

    def test_missing_dates_are_not_filled(self):
        dates=pd.date_range("2026-01-01",periods=4)
        nav=[dict(nav_date=d,nav=100+i) for i,d in enumerate(dates)]
        result=compare_world_benchmark(nav,pd.Series([10,12,13],index=dates[[0,2,3]]),
                                      pd.Series([80,81,82],index=dates[[0,1,3]]))
        self.assertEqual(result.index.tolist(),dates[[0,3]].tolist())

    def test_no_overlap_has_no_comparison(self):
        nav=[dict(nav_date="2026-01-01",nav=100)]
        price=pd.Series([10],index=pd.to_datetime(["2026-01-02"]))
        self.assertTrue(compare_world_benchmark(nav,price,price).empty)


if __name__=="__main__": unittest.main()
