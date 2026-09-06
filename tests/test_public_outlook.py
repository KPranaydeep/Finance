import unittest
from datetime import date, timedelta
import pandas as pd
from public_outlook import calendar_outlook, calendar_realization


class CalendarOutlookTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(nav_date=d.date(),nav=100*(1.001**i))
                   for i,d in enumerate(pd.bdate_range("2023-01-02",periods=300))]
        self.origin=self.rows[-1]["nav_date"]

    def test_four_weeks_is_twenty_weekday_returns(self):
        result=calendar_outlook(self.rows,self.origin)
        self.assertEqual(result["horizon_days"],28)
        self.assertEqual(result["horizon_unit"],"calendar_days")
        self.assertAlmostEqual(result["median_return"],1.001**20-1)
        self.assertEqual(result["target_date"],(self.origin+timedelta(days=28)).isoformat())
        self.assertEqual(result,calendar_outlook(self.rows,self.origin))

    def test_no_future_data_and_minimum_history(self):
        result=calendar_outlook(self.rows,self.origin)
        future=self.rows+[dict(nav_date=self.origin+timedelta(days=1),nav=100000)]
        self.assertEqual(result,calendar_outlook(future,self.origin))
        self.assertIsNone(calendar_outlook(self.rows[:60],self.origin))

    def test_invalid_nav_rejected(self):
        rows=[dict(row) for row in self.rows]
        rows[50]["nav"]=float("nan")
        with self.assertRaises(ValueError):
            calendar_outlook(rows,self.origin)

    def test_weekend_expiry_uses_friday_and_frozen_reference(self):
        payload=dict(target_date="2026-10-04",reference_date="2026-09-04",reference_nav=100)
        rows=[dict(nav_date=date(2026,10,2),nav=110)]
        self.assertIsNone(calendar_realization(payload,rows))
        rows.append(dict(nav_date=date(2026,10,5),nav=999))
        start,end=calendar_realization(payload,rows)
        self.assertEqual(start,(date(2026,9,4),100))
        self.assertEqual(end,(date(2026,10,2),110))


if __name__ == "__main__":
    unittest.main()
