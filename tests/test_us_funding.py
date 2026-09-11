import unittest
from public_us_funding import fx_gst, pro_brokerage, known_cost_floor


class FundingTests(unittest.TestCase):
    def test_quote(self):
        self.assertEqual(fx_gst(10_000), 45)
        self.assertEqual(10_000 - fx_gst(10_000), 9955)

    def test_boundaries_and_cap(self):
        self.assertEqual(fx_gst(100_000), 180)
        self.assertEqual(fx_gst(1_000_000), 990)
        self.assertAlmostEqual(pro_brokerage(10_000_000, 90), 25 * 90 * 1.18)

    def test_first_grid_pass(self):
        result = known_cost_floor()
        amount = result['amount_inr']
        self.assertLessEqual(result['known_drag'], .005)
        previous = amount - 100
        self.assertGreater((fx_gst(previous) + (previous - fx_gst(previous)) * .0015 * 1.18) / previous, .005)

    def test_invalid(self):
        with self.assertRaises(ValueError): fx_gst(float('nan'))
        self.assertIsNone(known_cost_floor(.00001))
