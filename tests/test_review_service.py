import unittest
from datetime import datetime, timezone
from unittest.mock import patch
import pandas as pd
from public_review.service import run, build_assessment
from public_review.market import calendar
from review_fixtures import baseline, policy
from test_review_operations import FakeDB


class ServiceTests(unittest.TestCase):
    def histories(self,p):
        schedule=calendar('2026-05-04','2026-09-09',p)
        dates=[str(d.date()) for d in schedule.index]
        return {t:pd.DataFrame({'Open':price,'Close':price,'Volume':1000,'Dividends':0.,'Stock Splits':0.},index=dates)
                for t,price in [('A.NS',100.),('B.NS',50.)]}

    def test_end_to_end_idempotent_model_only_run(self):
        p=policy(); p['capital_inr']=10000
        b=baseline(); db=FakeDB(); histories=self.histories(p)
        pub={k:b[k] for k in ['publication_id','basket_id','portfolio_version','published_at','weights']}
        now=datetime(2026,9,9,13,tzinfo=timezone.utc)
        with patch('public_review.service.publications',return_value=[pub]), \
             patch('public_review.market.fetch',return_value=histories), \
             patch('public_market_mood.fetch_mmi',return_value={'status':'UNAVAILABLE'}), \
             patch('public_review.service.send',return_value=False):
            first=run(db,'TEST',p,now=now)
            second=run(db,'TEST',p,now=now)
        self.assertEqual(first['failed'],0)
        self.assertEqual(second['failed'],0)
        self.assertEqual(sum(r['kind']=='BASELINE' for r in db.rows),1)
        self.assertEqual(sum(r['kind']=='ASSESSMENT' for r in db.rows),1)
        assessment=next(r['payload'] for r in db.rows if r['kind']=='ASSESSMENT')
        self.assertIsNone(assessment['decision']['next_review'])
        self.assertEqual(assessment['metrics']['date'],'2026-09-09')

    def test_split_does_not_silently_restate_frozen_units(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS'].loc['2026-09-09','Stock Splits']=2
        with self.assertRaisesRegex(ValueError,'CORPORATE_ACTION_REVIEW_REQUIRED'):
            build_assessment(b,h,'2026-09-09',['2026-09-10'],p,[],b['weights'],datetime(2026,9,9,13,tzinfo=timezone.utc))

    def test_interior_missing_history_is_not_filled(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS']=h['A.NS'].drop('2026-09-08')
        with self.assertRaisesRegex(ValueError,'COMMON_HISTORY_HAS_MISSING_SESSIONS'):
            build_assessment(b,h,'2026-09-09',['2026-09-10'],p,[],b['weights'],datetime(2026,9,9,13,tzinfo=timezone.utc))

