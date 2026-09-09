import unittest
from public_withdrawal_allocator import allocate_withdrawal


class AllocatorTests(unittest.TestCase):
    def setUp(self):
        self.plan={"orders":[{"ticker":"A","quantity":10,"planning_price":100},
                             {"ticker":"B","quantity":10,"planning_price":100}],
                   "residual_cash_inr":0}
        self.weights=[{"ticker":"A","target_weight":.5},{"ticker":"B","target_weight":.5}]
        self.cost={"fixed_cost_per_order":0,"statutory_cost_rate":0,"slippage_rate":0}

    def test_balanced_sales_and_reconciliation(self):
        r=allocate_withdrawal(self.plan,self.weights,400,self.cost)
        self.assertEqual(r["status"],"PLANNED")
        self.assertEqual([o["Shares"] for o in r["orders"]],[2,2])
        self.assertEqual(r["residual_cash"],0)

    def test_one_rupee_and_charges(self):
        self.cost["fixed_cost_per_order"]=20
        r=allocate_withdrawal(self.plan,self.weights,1,self.cost)
        self.assertEqual(len(r["orders"]),1)
        self.assertEqual(r["gross_sales"],100)
        self.assertEqual(r["costs"],20)
        self.assertEqual(r["residual_cash"],79)

    def test_cost_priority_over_better_diversification(self):
        self.cost["fixed_cost_per_order"]=20
        r=allocate_withdrawal(self.plan,self.weights,400,self.cost)
        self.assertEqual(len(r["orders"]),1)
        self.assertEqual(r["costs"],20)
        self.assertEqual(r["gross_sales"],500)

    def test_proportional_fee_minimum_matches_exhaustive_search(self):
        import math
        self.cost.update(fixed_cost_per_order=20,statutory_cost_rate=.0012,slippage_rate=.001)
        r=allocate_withdrawal(self.plan,self.weights,400,self.cost)
        candidates=[]
        for a in range(11):
            for b in range(11):
                gross=(a+b)*10000
                fee=math.ceil(gross*.0022+2000*((a>0)+(b>0))-1e-7)
                if gross-fee>=40000:
                    candidates.append(fee)
        self.assertEqual(round(r["costs"]*100),min(candidates))

    def test_cash_only_and_insufficient(self):
        self.plan["residual_cash_inr"]=5
        r=allocate_withdrawal(self.plan,self.weights,1,self.cost)
        self.assertEqual(r["status"],"CASH_ONLY")
        self.assertEqual(r["orders"],[])
        r=allocate_withdrawal(self.plan,self.weights,3000,self.cost)
        self.assertEqual(r["status"],"INSUFFICIENT")

    def test_full_liquidation_and_no_overselling(self):
        r=allocate_withdrawal(self.plan,self.weights,2000,self.cost)
        self.assertEqual([o["Shares"] for o in r["orders"]],[10,10])
        self.assertEqual(r["residual_cash"],0)

    def test_invalid_and_missing_prices(self):
        with self.assertRaises(ValueError):
            allocate_withdrawal(self.plan,self.weights,0,self.cost)
        self.plan["missing_prices"]=["C"]
        with self.assertRaises(ValueError):
            allocate_withdrawal(self.plan,self.weights,100,self.cost)


if __name__=="__main__": unittest.main()
