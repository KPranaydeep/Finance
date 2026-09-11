import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import pandas as pd
from public_review.forecast import estimate, METHOD
from public_review.core import decision, evaluate, digest
from public_review.preview import historical_preview
from review_fixtures import baseline, policy


class SecurityTargetTests(unittest.TestCase):
    def setup_case(self):
        b, p = baseline(), policy()
        p.update(concentration_limit=1., drift_limit=1., drawdown_limit=.9)
        r = pd.DataFrame(np.zeros((252, 2)), columns=['A.NS', 'B.NS'])
        v = {'passed': True, 'policy_hash': digest(p), 'tickers': list(r), 'method': METHOD}
        return b, p, r, v

    def test_security_crosses_even_when_basket_does_not(self):
        b, p, r, v = self.setup_case()
        shocks = np.zeros((100, 2, 2))
        shocks[:, 0, 0] = .1
        shocks[:, 0, 1] = -.1
        with patch('public_review.forecast.paths', return_value=shocks):
            f = estimate(b, {'A.NS':100, 'B.NS':50}, r,
                         ['2026-05-05', '2026-05-06'], p, b['capital'], v)
        self.assertEqual(f['earliest_security_crossing'], '2026-05-05')
        self.assertEqual(f['trigger_securities'], ['A.NS'])
        self.assertEqual(f['next_review'], '2026-05-05')
        self.assertEqual(f['curve'][0]['profit_crossing_probability'], 0)

    def test_not_one_lucky_path_or_union_of_weak_security_chances(self):
        b, p, r, v = self.setup_case()
        shocks = np.zeros((100, 1, 2))
        shocks[:15, 0, 0] = .05
        shocks[:15, 0, 1] = -.05
        shocks[15:30, 0, 1] = .05
        shocks[15:30, 0, 0] = -.05
        with patch('public_review.forecast.paths', return_value=shocks):
            f = estimate(b, {'A.NS':100, 'B.NS':50}, r, ['2026-05-05'], p, b['capital'], v)
        self.assertIsNone(f['earliest_security_crossing'])
        self.assertEqual(f['trigger_securities'], [])

    def test_entry_charges_and_exit_deductions_prevent_false_crossing(self):
        b, p, r, v = self.setup_case()
        shocks = np.full((100, 1, 2), .003)
        with patch('public_review.forecast.paths', return_value=shocks):
            f = estimate(b, {'A.NS':100, 'B.NS':50}, r, ['2026-05-05'], p, b['capital'], v)
        # Gross 0.3% daily growth exceeds 100% annualized; net need not.
        self.assertIsNone(f['earliest_security_crossing'])

    def test_old_validation_cannot_approve_new_method(self):
        b, p, r, v = self.setup_case()
        v.pop('method')
        f = estimate(b, {'A.NS':100, 'B.NS':50}, r, ['2026-05-05'], p, b['capital'], v)
        self.assertIsNone(f['next_review'])

    def test_dated_dividend_is_included_in_security_target(self):
        b, p, r, v = self.setup_case()
        f = estimate(b, {'A.NS':100, 'B.NS':50}, r, ['2026-05-06'], p, b['capital'], v,
                     dividends=[{'ticker':'A.NS', 'date':'2026-05-05', 'net':1000.}])
        self.assertEqual(f['trigger_securities'], ['A.NS'])

    def test_preview_ui_labels_provisional_and_shows_date(self):
        from streamlit.testing.v1 import AppTest
        def app():
            from public_review.ui import render_fresh_preview
            render_fresh_preview({'publication_id':'TEST', 'provisional':True,
                'assumption':'Hypothetical entry; not earned returns.',
                'as_of':'2026-09-09', 'checked_at':'2026-09-09T18:00:00+00:00',
                'forecast':{'next_review':None},
                'decision':{'next_review':'2099-01-01', 'reasons':[]}})
        at = AppTest.from_function(app).run()
        self.assertFalse(at.exception)
        self.assertEqual(at.metric[0].value, '2099-01-01')
        self.assertTrue(any('Provisional:' in c.value for c in at.caption))

    def test_crossed_security_requests_review_without_auto_rebalance(self):
        b, p, _, _ = self.setup_case()
        m = evaluate(b, {'A.NS':110, 'B.NS':45}, '2026-05-05', p)
        d = decision(m, b, b['weights'], p, b['capital'])
        self.assertIn('SECURITY_TARGET_REVIEW', d['reasons'])
        self.assertEqual(d['target_crossed_securities'], ['A.NS'])
        self.assertNotIn('REBALANCE_REVIEW', d['reasons'])

    def test_earlier_promised_date_is_retained(self):
        b, p, _, _ = self.setup_case()
        m = evaluate(b, {'A.NS':100, 'B.NS':50}, '2026-05-06', p)
        d = decision(m, b, b['weights'], p, b['capital'],
                     {'next_review':'2026-05-20'}, '2026-05-05')
        self.assertEqual(d['next_review'], '2026-05-05')
        self.assertIn('SCHEDULED_REVIEW_DUE', d['reasons'])

    def test_provisional_preview_has_no_performance_or_event_writes(self):
        from datetime import datetime, timezone
        from public_review.market import calendar
        p = policy()
        publication = {'publication_id':'NEW', 'basket_id':'TEST', 'portfolio_version':6,
                       'published_at':'2026-09-09T14:11:00+00:00',
                       'weights':{'A.NS':.5, 'B.NS':.5}}
        dates = [str(d.date()) for d in calendar('2025-06-01', '2026-09-10', p).index]
        histories = {t: pd.DataFrame({'Close':price, 'Open':price, 'Volume':1000.,
                                      'Dividends':0., 'Stock Splits':0.}, index=dates)
                     for t, price in [('A.NS',100.), ('B.NS',50.)]}
        events = []
        before = deepcopy(events)
        with patch('public_review.market.fetch', return_value=histories), \
             patch('public_review.preview.validate', return_value={'passed':False}):
            result = historical_preview(publication, p, events,
                                        datetime(2026,9,10,11,tzinfo=timezone.utc))
        self.assertTrue(result['provisional'])
        self.assertNotIn('metrics', result)
        self.assertEqual(events, before)
        self.assertEqual(result['as_of'], '2026-09-10')
        self.assertEqual(result['assumed_entry_date'], '2026-09-10')
        self.assertEqual(result['decision']['next_review'], '2026-09-11')
        self.assertIsNone(result['forecast']['next_review'])
