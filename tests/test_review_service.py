import unittest
from datetime import datetime, timezone
from unittest.mock import patch
import pandas as pd
from public_review import market, store
from public_review.service import (run, build_assessment, price_history_evidence,
                                   capture_publication_entries)
from public_review.core import evaluate
from public_review.market import calendar
from review_fixtures import baseline, policy
from test_review_operations import FakeDB


class ServiceTests(unittest.TestCase):
    def histories(self,p):
        schedule=calendar('2026-05-04','2026-09-09',p)
        dates=[str(d.date()) for d in schedule.index]
        return {t:pd.DataFrame({'Open':price,'Close':price,'Volume':1000,'Dividends':0.,'Stock Splits':0.},index=dates)
                for t,price in [('A.NS',100.),('B.NS',50.)]}

    def test_only_same_policy_validated_forecast_is_carried_forward(self):
        from copy import deepcopy
        from public_review.service import validated_promised_review
        p = policy()
        last = {'seq': 4, 'payload': {
            'policy': deepcopy(p),
            'forecast': {'next_review': '2026-09-30'},
            'decision': {'next_review': '2026-09-16',
                         'date_basis': 'CONTINUOUS_MONITORING'},
        }}
        self.assertEqual(
            validated_promised_review(last, None, p), '2026-09-30')
        changed = deepcopy(p)
        changed['minimum_forecast_review_sessions'] = 4
        self.assertIsNone(validated_promised_review(last, None, changed))
        self.assertIsNone(validated_promised_review(last, {'seq': 5}, p))

    def test_routine_monitoring_date_is_never_carried_forward(self):
        from public_review.service import validated_promised_review
        p = policy()
        last = {'seq': 4, 'payload': {
            'policy': p,
            'forecast': {'next_review': None},
            'decision': {'next_review': '2026-09-16',
                         'date_basis': 'NEXT_SESSION_RISK_CHECK'},
        }}
        self.assertIsNone(validated_promised_review(last, None, p))

    def test_end_to_end_idempotent_model_only_run(self):
        p=policy(); p['capital_inr']=10000
        b=baseline(); db=FakeDB(); histories=self.histories(p)
        pub={k:b[k] for k in ['publication_id','basket_id','portfolio_version','published_at','weights']}
        now=datetime(2026,9,9,13,tzinfo=timezone.utc)
        schedule={ticker:{'ticker':ticker,'kind':p['instrument_kinds'][ticker],
                  'market':'NSE','requested_entry_at':'2026-05-04T04:45:00+00:00',
                  'entry_date':'2026-05-04','basis':'NEXT_OPEN_PLUS_CONFIGURED_WAIT','ready':True}
                  for ticker in pub['weights']}
        prices={'A.NS':100.,'B.NS':50.}
        quote=lambda ticker,planned,policy_,now=None: {**planned,'price_inr':prices[ticker],
              'native_price':prices[ticker],'fx_to_inr':1.,
              'quote_at':planned['requested_entry_at'],'source':'test'}
        with patch('public_review.service.publications',return_value=[pub]), \
             patch('public_review.market.security_entry_schedule',return_value=schedule), \
             patch('public_review.market.fetch_entry_quote',side_effect=quote), \
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
        self.assertEqual(assessment['decision']['date_basis'], 'CONTINUOUS_MONITORING')
        self.assertIsNone(assessment['forecast']['next_review'])
        self.assertEqual(assessment['metrics']['date'],'2026-09-09')

    def test_entry_capture_freezes_each_quote_once_then_one_baseline(self):
        p=policy(); p['capital_inr']=10000
        b=baseline(); db=FakeDB()
        pub={k:b[k] for k in ['publication_id','basket_id','portfolio_version',
                              'published_at','weights']}
        now=datetime(2026,9,9,13,tzinfo=timezone.utc)
        schedule={ticker:{'ticker':ticker,'kind':p['instrument_kinds'][ticker],
                  'market':'NSE','requested_entry_at':'2026-09-09T04:01:00+00:00',
                  'session_open_at':'2026-09-09T03:45:00+00:00',
                  'session_close_at':'2026-09-09T10:00:00+00:00',
                  'entry_date':'2026-09-09','basis':'PUBLICATION_IN_SESSION',
                  'ready':True} for ticker in pub['weights']}
        calls=[]

        def first_quote(ticker, planned, policy_, now=None):
            calls.append(ticker)
            if ticker == 'B.NS':
                pending=market.AwaitingMarketEntry('2026-09-09',
                                                    '2026-09-09T13:30:00+00:00')
                pending.planned_entry=planned
                raise pending
            return {**planned,'price_inr':100.,'native_price':100.,
                    'fx_to_inr':1.,'quote_at':'2026-09-09T04:02:00+00:00',
                    'source':'test'}

        with patch('public_review.market.security_entry_schedule',
                   return_value=schedule), \
             patch('public_review.market.fetch_entry_quote',
                   side_effect=first_quote):
            partial=capture_publication_entries(
                db,'TEST',pub,p,now=now,history=[])
        self.assertEqual(partial['status'],'PARTIAL_ENTRY')
        self.assertEqual(partial['captured_entries'],1)
        self.assertEqual(sum(r['kind']=='SECURITY_ENTRY' for r in db.rows),1)
        self.assertFalse(any(r['kind']=='BASELINE' for r in db.rows))

        def final_quote(ticker, planned, policy_, now=None):
            calls.append(ticker)
            return {**planned,'price_inr':50.,'native_price':50.,
                    'fx_to_inr':1.,'quote_at':'2026-09-09T04:03:00+00:00',
                    'source':'test'}

        with patch('public_review.market.security_entry_schedule',
                   return_value=schedule), \
             patch('public_review.market.fetch_entry_quote',
                   side_effect=final_quote):
            frozen=capture_publication_entries(
                db,'TEST',pub,p,now=now,history=store.read(db,'TEST'))
            repeated=capture_publication_entries(
                db,'TEST',pub,p,now=now,history=store.read(db,'TEST'))
        self.assertEqual(frozen['status'],'BASELINE_FROZEN')
        self.assertTrue(frozen['baseline_created'])
        self.assertFalse(repeated['baseline_created'])
        self.assertEqual(calls.count('A.NS'),1)
        self.assertEqual(calls.count('B.NS'),2)
        self.assertEqual(sum(r['kind']=='SECURITY_ENTRY' for r in db.rows),2)
        self.assertEqual(sum(r['kind']=='BASELINE' for r in db.rows),1)

    def test_page_capture_budget_limits_synchronous_provider_work(self):
        p=policy(); p['capital_inr']=10000
        b=baseline(); db=FakeDB()
        pub={k:b[k] for k in ['publication_id','basket_id','portfolio_version',
                              'published_at','weights']}
        now=datetime(2026,9,9,13,tzinfo=timezone.utc)
        schedule={ticker:{'ticker':ticker,'kind':p['instrument_kinds'][ticker],
                  'market':'NSE','requested_entry_at':'2026-09-09T04:01:00+00:00',
                  'session_open_at':'2026-09-09T03:45:00+00:00',
                  'session_close_at':'2026-09-09T10:00:00+00:00',
                  'entry_date':'2026-09-09','basis':'PUBLICATION_IN_SESSION',
                  'ready':True} for ticker in pub['weights']}
        calls=[]

        def quote(ticker, planned, policy_, now=None):
            calls.append(ticker)
            return {**planned,'price_inr':100.,'native_price':100.,
                    'fx_to_inr':1.,'quote_at':'2026-09-09T04:02:00+00:00',
                    'source':'test'}

        with patch('public_review.market.security_entry_schedule',
                   return_value=schedule), \
             patch('public_review.market.fetch_entry_quote', side_effect=quote):
            partial=capture_publication_entries(
                db,'TEST',pub,p,now=now,history=[],max_new_entries=1)

        self.assertEqual(partial['status'],'PARTIAL_ENTRY')
        self.assertEqual(partial['new_entries'],1)
        self.assertEqual(len(calls),1)
        self.assertEqual(sum(r['kind']=='SECURITY_ENTRY' for r in db.rows),1)

    def test_build_assessment_marks_unchanged_review_as_acknowledged(self):
        from public_review.service import build_assessment, review_trigger_state
        from public_review.core import digest
        p = policy(); b = baseline(); histories = self.histories(p)
        for frame in histories.values():
            frame.loc['2026-09-09', ['Open', 'Close']] *= 2.5
        first = build_assessment(
            b, histories, '2026-09-09', ['2026-09-10'], p, [],
            b['weights'], datetime(2026,9,9,13,tzinfo=timezone.utc),
            comparisons=False)
        self.assertTrue(first['decision']['reasons'])
        state = review_trigger_state(first['decision'])
        prior = [{'kind':'ACKNOWLEDGED','baseline_id':b['baseline_id'],'seq':3,
                  'payload':{'at':'2026-09-09T12:00:00+00:00',
                             'policy_digest':digest(p),
                             'trigger_state':state,
                             'trigger_signature':digest(state)}}]
        second = build_assessment(
            b, histories, '2026-09-09', ['2026-09-10'], p, prior,
            b['weights'], datetime(2026,9,9,13,tzinfo=timezone.utc),
            comparisons=False)
        self.assertTrue(second['decision']['review_acknowledged'])
        self.assertFalse(second['decision']['review_required'])
        self.assertEqual(
            second['decision']['acknowledged_at'],
            '2026-09-09T12:00:00+00:00')

    def test_split_does_not_silently_restate_frozen_units(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS'].loc['2026-09-09','Stock Splits']=2
        with self.assertRaisesRegex(ValueError,'CORPORATE_ACTION_REVIEW_REQUIRED'):
            build_assessment(b,h,'2026-09-09',['2026-09-10'],p,[],b['weights'],datetime(2026,9,9,13,tzinfo=timezone.utc))

    def test_interior_missing_history_is_not_filled(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS']=h['A.NS'].drop('2026-09-08')
        result = build_assessment(b,h,'2026-09-09',['2026-09-10'],p,[],b['weights'],datetime(2026,9,9,13,tzinfo=timezone.utc))
        self.assertIn('2026-09-08', result['history_coverage']['missing_sessions'])
        self.assertEqual(result['history_coverage']['method'], 'complete-adjacent-session-pairs-no-fill')

    def test_forecast_wait_does_not_suppress_observed_net_return(self):
        p=policy(); b=baseline(); h=self.histories(p)
        result=build_assessment(
            b,h,'2026-09-09',['2026-09-10','2026-09-11'],p,[],
            b['weights'],datetime(2026,9,9,13,tzinfo=timezone.utc),
            comparisons=False,forecast_ready=False,
            observation_ready_at='2026-09-18T10:30:00+00:00',
            observation_rows=[{'ticker':'A.NS','observation_session':'2026-09-18'}])
        self.assertIn('net_total_return',result['metrics'])
        self.assertTrue(result['forecast_observation_pending'])
        self.assertEqual(result['observation_ready_at'],
                         '2026-09-18T10:30:00+00:00')
        self.assertIsNone(result['forecast']['next_review'])
        self.assertEqual(result['forecast']['status'],
                         'AWAITING_MINIMUM_OBSERVATION_SESSIONS')
        self.assertEqual(result['decision']['date_basis'],
                         'PROVISIONAL_OBSERVATION_WINDOW')

    def test_audit_fingerprint_preserves_mixed_market_nan_as_null(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS'].loc['2026-09-08','Close']=float('nan')
        evidence=price_history_evidence(h)
        self.assertIsNone(evidence['A.NS']['2026-09-08'])
        result=build_assessment(
            b,h,'2026-09-09',['2026-09-10'],p,[],b['weights'],
            datetime(2026,9,9,13,tzinfo=timezone.utc))
        self.assertEqual(len(result['price_hash']),64)
        self.assertIn('2026-09-08',result['history_coverage']['missing_sessions'])

    def test_real_dividend_without_usable_fx_fails_closed(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS'].loc['2026-09-08','Dividends']=float('nan')
        with self.assertRaisesRegex(ValueError,'STALE_OR_INCOMPLETE_MARKET_HISTORY'):
            build_assessment(
                b,h,'2026-09-09',['2026-09-10'],p,[],b['weights'],
                datetime(2026,9,9,13,tzinfo=timezone.utc))

    def test_actions_after_synchronized_as_of_are_not_assessed_early(self):
        p=policy(); b=baseline(); h=self.histories(p)
        h['A.NS'].loc['2026-09-09','Dividends']=2.
        h['A.NS'].loc['2026-09-09','Stock Splits']=2.
        result=build_assessment(
            b,h,'2026-09-08',['2026-09-10'],p,[],b['weights'],
            datetime(2026,9,9,13,tzinfo=timezone.utc))
        self.assertEqual(result['as_of'],'2026-09-08')
        self.assertEqual(result['metrics']['date'],'2026-09-08')
        self.assertEqual(result['metrics']['net_proceeds'],
                         evaluate(b,{'A.NS':100.,'B.NS':50.},'2026-09-08',p)['net_proceeds'])
