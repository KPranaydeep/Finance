import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from public_review.market import calendar
from public_review.partial import estimate_captured_security, METHOD
from public_review.service import run
from review_fixtures import baseline, policy
from test_review_operations import FakeDB
from public_review import market, store


class PartialSecurityReviewTests(unittest.TestCase):
    def publication(self):
        return {'publication_id':'PUB-PARTIAL', 'basket_id':'TEST', 'portfolio_version':7,
                'published_at':'2026-09-10T12:52:29+00:00',
                'weights':{'A.NS':.5, 'B.NS':.5}}

    def entry(self):
        return {'publication_id':'PUB-PARTIAL', 'ticker':'A.NS', 'kind':'equity',
                'market':'NSE', 'requested_entry_at':'2026-09-11T04:45:00+00:00',
                'quote_at':'2026-09-11T04:52:00+00:00', 'entry_date':'2026-09-11',
                'basis':'NEXT_OPEN_PLUS_CONFIGURED_WAIT', 'price_inr':100.,
                'native_price':100., 'fx_to_inr':1., 'source':'test'}

    def test_one_share_forecast_is_explicitly_provisional(self):
        p = policy()
        dates = [str(day.date()) for day in calendar('2025-01-01','2026-09-11',p).index]
        values = 100 * np.cumprod(np.full(len(dates), 1.001))
        history = pd.DataFrame({'Open':values, 'Close':values, 'Volume':1000.,
                                'Dividends':0., 'Stock Splits':0.}, index=dates)
        with patch('public_review.market.fetch', return_value={'A.NS':history}):
            result = estimate_captured_security(
                self.publication(), self.entry(), p,
                datetime(2026,9,11,12,tzinfo=timezone.utc))
        self.assertEqual(result['status'], 'PROVISIONAL_RESEARCH')
        self.assertEqual(result['method'], METHOD)
        self.assertEqual(result['notional_basis'],
                         'ONE_SHARE_WITH_MODELED_ENTRY_EXIT_COSTS_AND_TAX')
        self.assertEqual(result['ticker'], 'A.NS')
        self.assertIn('evidence_hash', result)

    def test_workflow_records_partial_forecast_without_freezing_basket(self):
        p, db, publication = policy(), FakeDB(), self.publication()
        schedule = {
            'A.NS': {**self.entry(), 'session_open_at':'2026-09-11T03:45:00+00:00',
                     'session_close_at':'2026-09-11T10:00:00+00:00', 'ready':True},
            'B.NS': {'ticker':'B.NS', 'kind':'listed_non_equity_etf', 'market':'NSE',
                     'requested_entry_at':'2026-09-14T04:45:00+00:00',
                     'session_open_at':'2026-09-14T03:45:00+00:00',
                     'session_close_at':'2026-09-14T10:00:00+00:00',
                     'entry_date':'2026-09-14', 'basis':'DEFERRED_NEXT_OPEN_PLUS_CONFIGURED_WAIT',
                     'ready':False}}
        quote = {**schedule['A.NS'], 'price_inr':100., 'native_price':100.,
                 'fx_to_inr':1., 'quote_at':'2026-09-11T04:52:00+00:00', 'source':'test'}
        partial = {'status':'PROVISIONAL_RESEARCH', 'method':METHOD,
                   'publication_id':'PUB-PARTIAL', 'ticker':'A.NS', 'as_of':'2026-09-11',
                   'entry_price_inr':100., 'estimated_crossing_date':'2026-09-21',
                   'crossing_probability':.2, 'horizon_probability':.3, 'target_xirr':1.,
                   'notional_basis':'ONE_SHARE_WITH_MODELED_ENTRY_EXIT_COSTS_AND_TAX',
                   'checked_at':'2026-09-11T12:00:00+00:00', 'evidence_hash':'x'}
        with patch('public_review.service.publications', return_value=[publication]), \
             patch('public_review.market.security_entry_schedule', return_value=schedule), \
             patch('public_review.market.fetch_entry_quote', return_value=quote), \
             patch('public_review.partial.estimate_captured_security', return_value=partial):
            result = run(db, 'TEST', p, now=datetime(2026,9,11,12,tzinfo=timezone.utc))
        self.assertEqual((result['failed'], result['waiting']), (0, 1))
        self.assertEqual([row['kind'] for row in db.rows],
                         ['SECURITY_ENTRY', 'SECURITY_REVIEW_PREVIEW', 'WAITING'])
        self.assertFalse(any(row['kind'] == 'BASELINE' for row in db.rows))

    def test_ui_labels_partial_date_as_research_only(self):
        def app():
            from public_review.ui import render_partial_security_reviews
            render_partial_security_reviews([{'kind':'SECURITY_REVIEW_PREVIEW','payload':{
                'publication_id':'PUB-PARTIAL','ticker':'A.NS','entry_price_inr':100.,
                'as_of':'2026-09-11','estimated_crossing_date':'2026-09-21',
                'crossing_probability':.2}}], 'PUB-PARTIAL')
        at = AppTest.from_function(app, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Provisional security review estimates' in item.value
                            for item in at.markdown))
        self.assertTrue(any('Research estimates' in item.value for item in at.caption))

    def test_fresh_wait_suppresses_older_failure_panel(self):
        def app():
            from datetime import datetime, timezone
            from public_review.ui import render_events
            events = [{'kind':'FAILURE', 'baseline_id':'PUB-PARTIAL', 'seq':1,
                       'payload':{'publication_id':'PUB-PARTIAL',
                                  'reason':'ENTRY_INTRADAY_HISTORY_UNAVAILABLE',
                                  'checked_at':'2026-09-11T13:46:00+00:00'}}]
            render_events(events, {'PUB-PARTIAL'},
                          datetime(2026,9,11,14,tzinfo=timezone.utc),
                          latest_publication_id='PUB-PARTIAL',
                          suppress_latest_failure=True)
        at = AppTest.from_function(app, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.warning)
        self.assertFalse(any('No monitoring record' in item.value for item in at.info))

    def test_partial_dates_remain_visible_until_first_complete_assessment(self):
        def app():
            from datetime import datetime, timezone
            from public_review.ui import render_events
            events = [
                {'kind':'SECURITY_REVIEW_PREVIEW', 'baseline_id':'PUB-PARTIAL',
                 'seq':1, 'payload':{'publication_id':'PUB-PARTIAL',
                 'ticker':'A.NS', 'entry_price_inr':100., 'as_of':'2026-09-11',
                 'estimated_crossing_date':'2026-09-21',
                 'crossing_probability':.2}},
                {'kind':'BASELINE', 'baseline_id':'BASE-PARTIAL', 'seq':2,
                 'payload':{'publication_id':'PUB-PARTIAL',
                 'baseline_id':'BASE-PARTIAL', 'portfolio_version':7,
                 'entry_date':'2026-09-11', 'capital':1000., 'lots':[]}},
                {'kind':'WAITING', 'baseline_id':'BASE-PARTIAL', 'seq':3,
                 'payload':{'publication_id':'PUB-PARTIAL',
                 'reason':'AWAITING_MARKET_ENTRY', 'entry_frozen':True,
                 'ready_at':'2026-09-11T20:30:00+00:00',
                 'checked_at':'2026-09-11T15:00:00+00:00'}},
            ]
            render_events(events, {'PUB-PARTIAL'},
                          datetime(2026,9,11,15,tzinfo=timezone.utc),
                          latest_publication_id='PUB-PARTIAL')
        at = AppTest.from_function(app, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Provisional security review estimates' in item.value
                            for item in at.markdown))

    def test_missing_completed_close_uses_verified_entry_mark(self):
        from public_review.preview import _immediate_baseline_preview
        publication = self.publication()
        b = {'baseline_id':'BASE-PARTIAL', 'publication_id':'PUB-PARTIAL',
             'basket_id':'TEST', 'portfolio_version':7, 'entry_date':'2026-09-11',
             'capital':1000., 'weights':{'A.NS':.5, 'B.NS':.5},
             'lots':[
                 {'ticker':'A.NS', 'kind':'equity', 'entry_date':'2026-09-11',
                  'price':101., 'entry_quote_at':'2026-09-11T04:45:00+00:00'},
                 {'ticker':'B.NS', 'kind':'listed_non_equity_etf',
                  'entry_date':'2026-09-11', 'price':50.,
                  'entry_quote_at':'2026-09-11T04:45:00+00:00'}]}
        histories = {
            'A.NS':pd.DataFrame({'Close':[100., np.nan, np.nan]},
                                index=['2026-09-10','2026-09-11','2026-09-15']),
            'B.NS':pd.DataFrame({'Close':[50., 54., 55.]},
                                index=['2026-09-10','2026-09-11','2026-09-15'])}
        schedule = pd.DataFrame({
            'market_open':pd.to_datetime(['2026-09-10T03:45:00Z','2026-09-11T03:45:00Z','2026-09-15T03:45:00Z']),
            'market_close':pd.to_datetime(['2026-09-10T10:00:00Z','2026-09-11T10:00:00Z','2026-09-15T10:00:00Z'])},
            index=pd.to_datetime(['2026-09-10','2026-09-11','2026-09-15']))
        future = pd.DataFrame({
            'market_open':pd.date_range('2026-09-16T03:45:00Z', periods=20, freq='D'),
            'market_close':pd.date_range('2026-09-16T10:00:00Z', periods=20, freq='D')},
            index=pd.date_range('2026-09-16', periods=20, freq='D'))
        returns = pd.DataFrame(np.zeros((126,2)), columns=['A.NS','B.NS'])
        forecast = {'next_review':None, 'research_candidate':'2026-09-16'}
        with patch('public_review.market.fetch', return_value=histories), \
             patch('public_review.market.calendar', return_value=schedule), \
             patch('public_review.market.joint_calendar', return_value=future), \
             patch('public_review.history.common_history',
                   return_value=(None, returns, {'start':'x','end':'2026-09-10',
                                 'usable_daily_returns':126,'missing_sessions':[]})) as common, \
             patch('public_review.preview.validate', return_value={'passed':False}), \
             patch('public_review.preview.estimate', return_value=forecast):
            preview = _immediate_baseline_preview(
                publication, b, policy(), [],
                datetime(2026,9,15,11,tzinfo=timezone.utc), 0)
        timing = {row['ticker']:row for row in preview['valuation_timing']['rows']}
        self.assertEqual(timing['A.NS']['price_source'],
                         'FROZEN_ENTRY_PRICE_PENDING_FIRST_CLOSE')
        self.assertEqual(timing['B.NS']['price_source'],
                         'LATEST_COMPLETED_POST_ENTRY_CLOSE')
        self.assertFalse(preview['valuation_timing']['all_prices_synchronized'])
        self.assertEqual(common.call_args.args[1], '2026-09-10')

    def test_stale_synchronized_history_becomes_provisional_success(self):
        db = FakeDB()
        b = baseline()
        b['entry_model_version'] = market.ENTRY_MODEL_VERSION
        store.append(db, 'TEST', 'baseline-test', 'BASELINE', b['baseline_id'], b)
        publication = {'publication_id':'PUB-TEST', 'basket_id':'TEST',
                       'portfolio_version':1, 'published_at':'2026-05-01T10:00:00+00:00',
                       'weights':{'A.NS':.5,'B.NS':.5}}
        provisional = {'provisional':True, 'publication_id':'PUB-TEST',
                       'as_of':'mixed chronology-safe marks',
                       'checked_at':'2026-09-12T03:00:00+00:00',
                       'assumption':'test', 'valuation_timing':{'rows':[]},
                       'forecast':{'research_candidate':'2026-09-15'},
                       'decision':{'next_review':'2026-09-15','reasons':[]}}
        with patch('public_review.service.publications', return_value=[publication]), \
             patch('public_review.market.sessions',
                   return_value=('2026-05-04','2026-09-11',['2026-09-15'])), \
             patch('public_review.market.fetch', return_value={}), \
             patch('public_review.market.synchronized_dates',
                   side_effect=ValueError('STALE_OR_INCOMPLETE_MARKET_HISTORY')), \
             patch('public_review.preview._immediate_baseline_preview',
                   return_value=provisional):
            result = run(db, 'TEST', policy(),
                         now=datetime(2026,9,12,3,tzinfo=timezone.utc))
        self.assertEqual(result['failed'], 0)
        self.assertEqual(result['results'][0]['status'], 'PROVISIONAL_ASSESSED')
        self.assertEqual(db.rows[-1]['kind'], 'PREVIEW')


if __name__ == '__main__':
    unittest.main()
