import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from public_review.market import calendar
from public_review.partial import estimate_captured_security, METHOD
from public_review.service import run
from review_fixtures import policy
from test_review_operations import FakeDB


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


if __name__ == '__main__':
    unittest.main()
