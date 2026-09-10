import unittest
import numpy as np
import pandas as pd
from public_review.forecast import paths, estimate, validate, METHOD
from public_review.core import digest
from review_fixtures import policy, baseline


class ForecastTests(unittest.TestCase):
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
        self.assertEqual(f['research_candidate'],days[-1])

    def test_policy_hash_gates_date(self):
        b=baseline(); p=policy()
        r=pd.DataFrame(np.zeros((252,2)),columns=['A.NS','B.NS'])
        days=[str(d.date()) for d in pd.bdate_range('2026-09-10',periods=20)]
        v={'passed':True,'policy_hash':digest(p),'tickers':['A.NS','B.NS'],'method':METHOD}
        self.assertIsNotNone(estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000,v)['next_review'])
        v['policy_hash']='wrong'
        self.assertIsNone(estimate(b,{'A.NS':100,'B.NS':50},r,days,p,10000,v)['next_review'])

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
