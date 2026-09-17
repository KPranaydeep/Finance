import unittest
import numpy as np
import pandas as pd
from public_review.forecast import (paths, estimate, validate, METHOD,
                                    allocation_weighted_review)
from public_review.core import digest
from review_fixtures import policy, baseline


class ForecastTests(unittest.TestCase):
    def test_allocation_probability_weighted_review_formula(self):
        crossings=[
            {'ticker':'A.NS','crossing_date':'2026-09-18','review_date':'2026-09-18','probability':.20},
            {'ticker':'B.NS','crossing_date':'2026-09-22','review_date':'2026-09-22','probability':.40},
            {'ticker':'C.NS','crossing_date':None,'probability':None},
        ]
        result=allocation_weighted_review(
            crossings, {'A.NS':.75,'B.NS':.25,'C.NS':.10},
            ['2026-09-18','2026-09-21','2026-09-22','2026-09-23'])
        self.assertEqual(result['raw_weighted_date'],'2026-09-20')
        self.assertEqual(result['review_date'],'2026-09-21')
        self.assertEqual(result['review_followup_date'],'2026-09-22')
        self.assertAlmostEqual(result['probability_weighted_mass'],.25)
        self.assertAlmostEqual(result['contributing_target_weight'],1.)

    def test_allocation_weighted_review_ignores_zero_and_missing_values(self):
        self.assertIsNone(allocation_weighted_review(
            [{'ticker':'A.NS','crossing_date':None,'probability':None},
             {'ticker':'B.NS','crossing_date':'2026-09-22','probability':0.}],
            {'A.NS':.5,'B.NS':.5}, ['2026-09-22','2026-09-23']))

    def test_joint_sampling_preserves_relationship(self):
        a=np.linspace(-.05,.05,252); matrix=np.column_stack([a,2*a])
        simulated=paths(matrix,20,100,5,123)
        np.testing.assert_allclose(simulated[:,:,1],2*simulated[:,:,0])
        np.testing.assert_array_equal(simulated,paths(matrix,20,100,5,123))

    def test_short_or_bad_history_rejected(self):
        for r in [np.zeros((125,2)),np.full((252,2),np.nan),np.full((252,2),-1.)]:
            with self.assertRaises(ValueError): paths(r,20,10,5,1)

    def test_never_crossed_kept_and_no_date_without_validation(self):
        b=baseline(); p=policy(); p['concentration_limit']=1; p['drift_limit']=1
        r=pd.DataFrame(np.zeros((252,2)),columns=['A.NS','B.NS'])
        days=[str(d.date()) for d in pd.bdate_range('2026-09-10',periods=20)]
        f=estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000)
        self.assertEqual(f['never_crossed_fraction'],1.)
        self.assertIsNone(f['next_review'])
        self.assertEqual(f['unadjusted_research_candidate'],days[-1])
        self.assertEqual(f['research_candidate'],days[-2])
        self.assertEqual(f['next_common_review_session'],days[-1])
        self.assertIsNone(f['expected_security_crossing'])
        self.assertEqual(f['any_security_crossing_probability'],0.)

    def test_expected_security_date_uses_joint_path_first_passage(self):
        b=baseline(); p=policy(); p['concentration_limit']=1; p['drift_limit']=1
        r=pd.DataFrame(np.full((252,2),.10),columns=['A.NS','B.NS'])
        days=[str(d.date()) for d in pd.bdate_range('2026-09-16',periods=20)]
        f=estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000)
        self.assertIsNotNone(f['expected_security_crossing'])
        self.assertEqual(f['any_security_crossing_probability'],1.)
        self.assertEqual(f['research_candidate'],f['expected_security_crossing'])

    def test_policy_hash_gates_date(self):
        b=baseline(); p=policy()
        r=pd.DataFrame(np.zeros((252,2)),columns=['A.NS','B.NS'])
        days=[str(d.date()) for d in pd.bdate_range('2026-09-10',periods=20)]
        v={'passed':True,'policy_hash':digest(p),'tickers':['A.NS','B.NS'],'method':METHOD}
        self.assertIsNotNone(estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000,v)['next_review'])
        v['policy_hash']='wrong'
        self.assertIsNone(estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000,v)['next_review'])

    def test_minimum_forecast_session_is_readiness_not_future_date_offset(self):
        b=baseline(); p=policy()
        p['minimum_forecast_review_sessions']=3
        p['concentration_limit']=.1
        r=pd.DataFrame(np.zeros((252,2)),columns=['A.NS','B.NS'])
        days=[str(d.date()) for d in pd.bdate_range('2026-09-10',periods=20)]
        f=estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000)
        self.assertEqual(f['research_candidate'],days[0])
        self.assertEqual(f['minimum_forecast_review_sessions'],3)

    def test_review_window_uses_next_market_session_not_next_calendar_day(self):
        b=baseline(); p=policy()
        p['max_review_sessions']=1
        p['concentration_limit']=.1
        r=pd.DataFrame(np.zeros((252,2)),columns=['A.NS','B.NS'])
        # Friday is modeled, but Saturday is closed. The operational window
        # therefore uses the next verified consecutive pair, Monday-Tuesday.
        days=['2026-09-18','2026-09-21','2026-09-22']
        f=estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000)
        self.assertEqual(f['unadjusted_research_candidate'],'2026-09-18')
        self.assertEqual(f['review_session'],'2026-09-21')
        self.assertEqual(f['next_common_review_session'],'2026-09-22')
        self.assertEqual(len(f['curve']),1)

    def test_walkforward_nonoverlap_and_no_invented_pass(self):
        p=policy(); b=baseline()
        r=pd.DataFrame(np.zeros((292,2)),columns=['A.NS','B.NS'])
        dates=[str(d.date()) for d in pd.bdate_range(end='2026-09-09',periods=292)]
        v=validate(r,b,{'A.NS':100,'B.NS':50},dates,p)
        self.assertEqual(v['folds'],2)
        self.assertFalse(v['passed'])
        for f in v['detail']:
            self.assertLess(f['train_end_row'],f['test_start_row'])
        self.assertLess(v['detail'][0]['test_end_row'],v['detail'][1]['test_start_row'])
