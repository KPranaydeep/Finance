from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from public_review import market, store
from public_review.service import build_assessment, run
from review_fixtures import baseline, policy
from test_review_operations import FakeDB


class ReviewCalendarCoverageTests(unittest.TestCase):
    def histories(self, p):
        dates = [str(d.date()) for d in market.calendar('2025-01-01', '2026-09-09', p).index]
        return {t: pd.DataFrame({'Open': price, 'Close': price, 'Volume': 1000,
                                'Dividends': 0., 'Stock Splits': 0.}, index=dates)
                for t, price in [('A.NS', 100.), ('B.NS', 50.)]}

    def test_optional_followup_tail_does_not_block_completed_assessment(self):
        p = policy()
        p['max_review_sessions'] = 40
        _, as_of, future = market.sessions(
            datetime(2026, 11, 1, 3, tzinfo=timezone.utc),
            '2026-09-30T03:43:00+00:00', p,
            {'A.NS': 'equity', 'B': 'foreign_us_listing'},
        )
        self.assertEqual(as_of, '2026-10-30')
        self.assertGreater(len(future), 0)
        self.assertLess(len(future), 45)
        self.assertLessEqual(future[-1], p['calendar_verified_through'])

    def test_expired_calendar_still_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'CALENDAR_REVIEW_REQUIRED'):
            market.sessions(datetime(2027, 1, 1, 3, tzinfo=timezone.utc),
                            '2026-09-30T03:43:00+00:00', policy())

    def test_empty_future_keeps_observed_metrics_without_fabricating_forecast(self):
        p, b = policy(), baseline()
        with patch('public_review.service.estimate') as estimate, \
             patch('public_review.service.validate') as validate, \
             patch('public_market_mood.fetch_mmi', return_value={'status': 'UNAVAILABLE'}):
            result = build_assessment(
                b, self.histories(p), '2026-09-09', [], p, [], b['weights'],
                datetime(2026, 9, 9, 13, tzinfo=timezone.utc), comparisons=False)
        estimate.assert_not_called()
        validate.assert_not_called()
        self.assertIn('net_total_return', result['metrics'])
        self.assertEqual(result['forecast']['status'], 'CALENDAR_HORIZON_INCOMPLETE')
        self.assertIsNone(result['forecast']['next_review'])
        self.assertFalse(result['calendar_coverage']['complete'])
        self.assertEqual(result['calendar_coverage']['available_forecast_sessions'], 0)

    def test_exact_full_horizon_runs_forecast_without_extra_five_sessions(self):
        p, b = policy(), baseline()
        future = [str(d.date()) for d in pd.bdate_range('2026-09-10', periods=p['max_review_sessions'])]
        with patch('public_review.service.estimate', return_value={
                'status': 'RESEARCH_ONLY', 'next_review': None}) as estimate, \
             patch('public_review.service.validate', return_value={'passed': False}), \
             patch('public_market_mood.fetch_mmi', return_value={'status': 'UNAVAILABLE'}):
            result = build_assessment(
                b, self.histories(p), '2026-09-09', future, p, [], b['weights'],
                datetime(2026, 9, 9, 13, tzinfo=timezone.utc), comparisons=False)
        estimate.assert_called_once()
        self.assertEqual(estimate.call_args.args[3], future)
        self.assertTrue(result['calendar_coverage']['complete'])

    def test_short_forecast_calendar_still_produces_genuine_heartbeat(self):
        p, b, db = policy(), baseline(), FakeDB()
        b['entry_model_version'] = market.ENTRY_MODEL_VERSION
        store.append(db, 'TEST', 'baseline-test', 'BASELINE', b['baseline_id'], b)
        publication = {k: b[k] for k in ['publication_id', 'basket_id', 'portfolio_version',
                                        'published_at', 'weights']}
        with patch('public_review.service.publications', return_value=[publication]), \
             patch('public_review.market.require_forecast_observation_sessions', return_value=[]), \
             patch('public_review.market.sessions', return_value=('2026-05-04', '2026-09-09', [])), \
             patch('public_review.market.fetch', return_value=self.histories(p)), \
             patch('public_market_mood.fetch_mmi', return_value={'status': 'UNAVAILABLE'}), \
             patch('public_review.service.compare_exits', return_value=[]), \
             patch('public_review.service.send', return_value=False):
            result = run(db, 'TEST', p, now=datetime(2026, 9, 9, 13, tzinfo=timezone.utc))
        self.assertEqual(result['failed'], 0, result)
        self.assertIsNotNone(store.latest(db.rows, 'HEARTBEAT', b['baseline_id']))
        assessment = store.latest(db.rows, 'ASSESSMENT', b['baseline_id'])['payload']
        self.assertEqual(assessment['forecast']['status'], 'CALENDAR_HORIZON_INCOMPLETE')

    def test_ui_shows_observed_values_and_calendar_limitation(self):
        def app():
            from datetime import datetime, timezone
            from public_review.ui import render_events
            from public_review.core import evaluate, decision
            from review_fixtures import baseline, policy
            b, p = baseline(), policy()
            now = datetime(2026, 9, 9, 13, tzinfo=timezone.utc)
            metrics = evaluate(b, {'A.NS': 100., 'B.NS': 50.}, '2026-09-09', p)
            payload = {'checked_at': now.isoformat(), 'as_of': '2026-09-09',
                       'metrics': metrics, 'decision': decision(metrics, b, b['weights'], p, 10000),
                       'forecast': {'status': 'CALENDAR_HORIZON_INCOMPLETE'}, 'policy': p,
                       'validation': {'passed': False}, 'comparisons': [], 'mmi': {},
                       'dividend_assumption': 'Synthetic test',
                       'calendar_coverage': {'complete': False, 'available_forecast_sessions': 3,
                                             'required_forecast_sessions': 20,
                                             'verified_through': '2026-12-31'}}
            render_events([
                {'kind': 'BASELINE', 'seq': 1, 'baseline_id': b['baseline_id'], 'payload': b},
                {'kind': 'ASSESSMENT', 'seq': 2, 'baseline_id': b['baseline_id'],
                 'payload': payload, 'event_hash': 'test-only'},
            ], now=now)
        at = AppTest.from_function(app, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.metric), 4)
        self.assertTrue(any('3 of 20 future sessions' in x.value for x in at.info))
        self.assertFalse(any('heartbeat is stale' in x.value for x in at.warning))

    def test_calendar_change_is_not_deduplicated_against_older_same_price_assessment(self):
        p, b, db = policy(), baseline(), FakeDB()
        b['entry_model_version'] = market.ENTRY_MODEL_VERSION
        store.append(db, 'TEST', 'baseline-test', 'BASELINE', b['baseline_id'], b)
        pub = {k: b[k] for k in ['publication_id', 'basket_id', 'portfolio_version',
                                'published_at', 'weights']}
        future = [str(d.date()) for d in pd.bdate_range('2026-09-10', periods=p['max_review_sessions'])]
        with patch('public_review.service.publications', return_value=[pub]), \
             patch('public_review.market.require_forecast_observation_sessions', return_value=[]), \
             patch('public_review.market.sessions', side_effect=[
                 ('2026-05-04', '2026-09-09', future), ('2026-05-04', '2026-09-09', [])]), \
             patch('public_review.market.fetch', return_value=self.histories(p)), \
             patch('public_review.service.estimate', return_value={
                 'status': 'RESEARCH_ONLY', 'next_review': None}), \
             patch('public_review.service.validate', return_value={'passed': False}), \
             patch('public_market_mood.fetch_mmi', return_value={'status': 'UNAVAILABLE'}), \
             patch('public_review.service.compare_exits', return_value=[]), \
             patch('public_review.service.send', return_value=False):
            first = run(db, 'TEST', p, now=datetime(2026, 9, 9, 13, tzinfo=timezone.utc))
            second = run(db, 'TEST', p, now=datetime(2026, 9, 9, 13, 1, tzinfo=timezone.utc))
        self.assertEqual(first['failed'] + second['failed'], 0)
        assessments = [e for e in db.rows if e['kind'] == 'ASSESSMENT']
        self.assertEqual(len(assessments), 2)
        self.assertTrue(assessments[0]['payload']['calendar_coverage']['complete'])
        self.assertFalse(assessments[1]['payload']['calendar_coverage']['complete'])

    def test_failed_review_can_publish_failure_evidence_without_greenwashing(self):
        workflow = (Path(__file__).resolve().parents[1] /
                    '.github/workflows/public_portfolio_review.yml').read_text()
        self.assertIn("steps.model-review.outcome == 'failure'", workflow)
        self.assertIn("steps.export-snapshot.outcome == 'success'", workflow)
        self.assertNotIn('continue-on-error: true', workflow)


if __name__ == '__main__':
    unittest.main()
