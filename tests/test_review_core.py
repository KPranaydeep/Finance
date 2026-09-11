import unittest
import math
from copy import deepcopy
from public_review.core import xirr, freeze, evaluate, decision, compare_exits, digest
from public_review.costs import charges, tax_rate, sell_value
from review_fixtures import policy, baseline


class ReviewCoreTests(unittest.TestCase):
    def test_xirr_one_year_doubling(self):
        self.assertAlmostEqual(xirr([("2026-05-04", -100), ("2027-05-04", 200)]), 1.)

    def test_xirr_half_year_not_double(self):
        r = xirr([("2026-05-04", -100), ("2026-11-03", 141.555)])
        self.assertAlmostEqual(r, 1., places=3)

    def test_xirr_multiple_cashflows(self):
        flows = [("2026-05-04", -100), ("2026-06-04", -50), ("2027-05-04", 180)]
        r = xirr(flows)
        from datetime import date
        start = date.fromisoformat(flows[0][0])
        self.assertAlmostEqual(sum(v / (1+r)**((date.fromisoformat(d)-start).days/365) for d,v in flows), 0., places=7)

    def test_xirr_ambiguous_and_same_day(self):
        self.assertIsNone(xirr([("2026-05-04", -100), ("2026-05-04", 101)]))
        self.assertIsNone(xirr([("2026-05-04", -100), ("2026-06-04", 300), ("2026-07-04", -200), ("2026-08-04", 10)]))

    def test_cost_min_cap_and_dp(self):
        p = policy()
        self.assertEqual(charges(1, "BUY", "equity", p)["brokerage"], .03)
        self.assertEqual(charges(1000, "BUY", "equity", p)["brokerage"], 5.)
        self.assertEqual(charges(100000, "BUY", "equity", p)["brokerage"], 20.)
        self.assertEqual(charges(99, "SELL", "equity", p)["dp"], 13.)
        self.assertEqual(charges(100, "SELL", "equity", p)["dp"], 20.)
        self.assertEqual(charges(0, "SELL", "equity", p)["total"], 0.)

    def test_instrument_stt(self):
        p = policy()
        self.assertEqual(charges(10000, "BUY", "equity", p)["stt"], 10.)
        self.assertEqual(charges(10000, "BUY", "equity_etf", p)["stt"], 0.)
        self.assertEqual(charges(10000, "SELL", "listed_non_equity_etf", p)["stt"], 0.)

    def test_tax_anniversary_and_classification(self):
        p = policy()
        self.assertAlmostEqual(tax_rate("equity", "2026-05-04", "2027-05-04", p), .208)
        self.assertAlmostEqual(tax_rate("equity", "2026-05-04", "2027-05-05", p), .13)
        self.assertAlmostEqual(tax_rate("listed_non_equity_etf", "2026-05-04", "2026-09-09", p), .312)
        self.assertAlmostEqual(tax_rate("specified_debt_etf", "2026-05-04", "2028-09-09", p), .312)
        with self.assertRaises(ValueError): tax_rate("unknown", "2026-05-04", "2026-09-09", p)

    def test_fee_reconciliation(self):
        for v in [1, 99, 100, 500, 100000]:
            fees = charges(v, "SELL", "equity", policy())
            self.assertAlmostEqual(fees["total"], sum(v for k,v in fees.items() if k != "total"), places=2)

    def test_baseline_no_overspend(self):
        b = baseline()
        spent = sum(l["quantity"]*l["price"]+l["entry_charges"]["total"] for l in b["lots"])
        self.assertLessEqual(spent, b["capital"])
        self.assertAlmostEqual(spent+b["cash"], b["capital"])
        self.assertTrue(all(isinstance(l["quantity"],int) for l in b["lots"]))

    def test_baseline_preserves_per_security_entry_dates(self):
        p = policy()
        records = {
            'A.NS': {'entry_date': '2026-05-04', 'requested_entry_at': '2026-05-04T05:00:00+00:00',
                     'quote_at': '2026-05-04T05:00:00+00:00', 'basis': 'PUBLICATION_DURING_MARKET',
                     'source': 'test'},
            'B.NS': {'entry_date': '2026-05-05', 'requested_entry_at': '2026-05-05T04:45:00+00:00',
                     'quote_at': '2026-05-05T04:45:00+00:00', 'basis': 'NEXT_OPEN_PLUS_CONFIGURED_WAIT',
                     'source': 'test'},
        }
        b = freeze({'publication_id':'P','basket_id':'B','portfolio_version':1,
                    'published_at':'2026-05-04T05:00:00+00:00'},
                   {'A.NS':.5,'B.NS':.5},{'A.NS':100.,'B.NS':50.},'2026-05-04',
                   10000.,p['instrument_kinds'],p,'2026-05-05T05:00:00+00:00',records)
        self.assertEqual(b['entry_date'], '2026-05-04')
        self.assertEqual(b['fully_invested_date'], '2026-05-05')
        self.assertEqual({lot['ticker']:lot['entry_date'] for lot in b['lots']},
                         {'A.NS':'2026-05-04','B.NS':'2026-05-05'})

    def test_basket_return_not_security_average(self):
        b = baseline(); before = digest(b)
        m = evaluate(b, {"A.NS": 130, "B.NS": 45}, "2026-09-09", policy())
        self.assertEqual(before, digest(b))
        self.assertAlmostEqual(m["net_proceeds"], b["cash"]+sum(r["net"] for r in m["rows"]), places=2)
        self.assertAlmostEqual(m["xirr"], (m["net_proceeds"]/b["capital"])**(365/m["days_held"])-1)

    def test_retained_dividends_not_double_counted(self):
        b = baseline(); prices = {"A.NS": 100, "B.NS": 50}
        a = evaluate(b,prices,"2026-09-09",policy())
        d = evaluate(b,prices,"2026-09-09",policy(),[{"ticker":"A.NS","date":"2026-06-01","net":100}])
        self.assertAlmostEqual(d["net_proceeds"]-a["net_proceeds"],100)

    def test_no_loss_tax_credit(self):
        b=baseline(); sale=sell_value(1,50,b["lots"][0],"2026-09-09",policy())
        self.assertEqual(sale["tax"],0)

    def test_future_dividends_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(baseline(),{'A.NS':100,'B.NS':50},'2026-09-09',policy(),
                     [{'ticker':'A.NS','date':'2026-09-10','net':10}])

    def test_keep_earlier_promised_date(self):
        b=baseline(); m=evaluate(b,{"A.NS":100,"B.NS":50},"2026-09-09",policy())
        d=decision(m,b,b["weights"],policy(),10000,{"next_review":"2026-10-01"},"2026-09-15")
        self.assertEqual(d["next_review"],"2026-09-15")

    def test_profit_and_risk_independent(self):
        b=baseline(); m=evaluate(b,{"A.NS":500,"B.NS":250},"2026-09-09",policy())
        d=decision(m,b,b["weights"],policy(),100000)
        self.assertIn("PROFIT_TAKING_REVIEW",d["reasons"])
        self.assertIn("RISK_REVIEW",d["reasons"])

    def test_no_rebalance_without_benefit_evidence(self):
        b=baseline(); m=evaluate(b,{"A.NS":100,"B.NS":50},"2026-09-09",policy())
        d=decision(m,b,{"A.NS":.8,"B.NS":.2},policy(),10000)
        self.assertNotIn("REBALANCE_REVIEW",d["reasons"])
        self.assertEqual(d["rebalance"]["status"],"BENEFIT_NOT_ESTABLISHED")

    def test_invalid_prices_fail(self):
        with self.assertRaises(ValueError): evaluate(baseline(),{"A.NS":float('nan'),"B.NS":50},"2026-09-09",policy())

    def test_cost_first_exit_reconciles(self):
        b=baseline(); p=policy(); prices={"A.NS":150,"B.NS":80}
        m=evaluate(b,prices,"2026-09-09",p)
        options=compare_exits(b,prices,"2026-09-09",p,m)
        recovered=next(o for o in options if o["option"]=="Recover initial capital")
        self.assertGreaterEqual(recovered["cash_raised"],b["capital"])
        for order in recovered["orders"]:
            lot=next(l for l in b["lots"] if l["ticker"]==order["ticker"])
            self.assertLessEqual(order["shares"],lot["quantity"])

    def test_insufficient_capital_recovery(self):
        b=baseline(); p=policy(); prices={"A.NS":50,"B.NS":25}
        m=evaluate(b,prices,"2026-09-09",p)
        options=compare_exits(b,prices,"2026-09-09",p,m)
        self.assertEqual(options[2]["status"],"INSUFFICIENT_NET_PROCEEDS")
