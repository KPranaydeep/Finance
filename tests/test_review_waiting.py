import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from public_review.market import sessions, AwaitingMarketEntry
from public_review.service import run
from review_fixtures import policy
from test_review_operations import FakeDB


class WaitingTests(unittest.TestCase):
    def publication(self):
        return {'publication_id': 'PUB-P006', 'basket_id': 'TEST', 'portfolio_version': 6,
                'published_at': '2026-09-09T14:11:00+00:00',
                'weights': {'A.NS': .5, 'B.NS': .5}}

    def test_after_close_publication_has_explicit_ready_time(self):
        with self.assertRaises(AwaitingMarketEntry) as caught:
            sessions(datetime(2026, 9, 9, 18, tzinfo=timezone.utc),
                     self.publication()['published_at'], policy())
        self.assertEqual(caught.exception.entry_date, '2026-09-10')
        self.assertEqual(caught.exception.ready_at, '2026-09-10T10:30:00+00:00')

    def test_normal_wait_is_not_a_failed_job_or_trade(self):
        db = FakeDB()
        with patch('public_review.service.publications', return_value=[self.publication()]), \
             patch('public_review.market.fetch') as fetch, patch('public_review.service.send') as send:
            result = run(db, 'TEST', policy(), now=datetime(2026, 9, 9, 18, tzinfo=timezone.utc))
        self.assertEqual(result['failed'], 0)
        self.assertEqual(result['waiting'], 1)
        self.assertEqual(result['results'][0]['reason'], 'AWAITING_MARKET_ENTRY')
        self.assertEqual([r['kind'] for r in db.rows], ['WAITING'])
        fetch.assert_not_called()
        send.assert_not_called()

    def test_entry_session_becomes_available_after_buffer(self):
        entry, as_of, _ = sessions(datetime(2026, 9, 10, 10, 30, tzinfo=timezone.utc),
                                  self.publication()['published_at'], policy())
        self.assertEqual((entry, as_of), ('2026-09-10', '2026-09-10'))

    def test_real_error_still_fails_and_sanitizes_details(self):
        db = FakeDB()
        with patch('public_review.service.publications', return_value=[self.publication()]), \
             patch('public_review.market.sessions', side_effect=ValueError('SECRET_DATABASE_PASSWORD')), \
             patch('public_review.service.send', return_value=False):
            result = run(db, 'TEST', policy(), now=datetime(2026, 9, 9, 18, tzinfo=timezone.utc))
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['waiting'], 0)
        self.assertEqual(result['results'][0]['reason'], 'MONITOR_CHECK_FAILED')
        self.assertEqual(result['results'][0]['stage'], 'session_calendar')
        self.assertNotIn('SECRET_DATABASE_PASSWORD', json.dumps(result) + json.dumps(db.rows))

    def test_missing_classifications_remain_a_real_failure(self):
        p = policy()
        p['instrument_kinds'] = {}
        db = FakeDB()
        with patch('public_review.service.publications', return_value=[self.publication()]), \
             patch('public_review.service.send', return_value=False):
            result = run(db, 'TEST', p, now=datetime(2026, 9, 10, 11, tzinfo=timezone.utc))
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['results'][0]['reason'], 'INSTRUMENT_CLASSIFICATION_REQUIRED')

    def test_waiting_ui_displays_status_without_metrics(self):
        def app():
            from datetime import datetime, timezone
            from public_review.ui import render_events
            events = [{'kind': 'WAITING', 'baseline_id': 'PUB-P006', 'seq': 1, 'payload': {
                'publication_id': 'PUB-P006', 'reason': 'AWAITING_MARKET_ENTRY',
                'entry_date': '2026-09-10', 'ready_at': '2026-09-10T10:30:00+00:00',
                'checked_at': '2026-09-09T18:00:00+00:00'}}]
            render_events(events, {'PUB-P006'}, datetime(2026, 9, 9, 18, tzinfo=timezone.utc),
                          latest_publication_id='PUB-P006')
        at = AppTest.from_function(app, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.warning)
        self.assertEqual(len(at.metric), 0)
        self.assertTrue(any('Awaiting market entry' in x.value for x in at.info))
        self.assertTrue(any('16:00 IST' in x.value for x in at.caption))

    def test_empty_calendar_is_not_treated_as_waiting(self):
        import pandas as pd
        empty = pd.DataFrame({'market_open': pd.Series(dtype='datetime64[ns, UTC]'),
                              'market_close': pd.Series(dtype='datetime64[ns, UTC]')})
        with patch('public_review.market.calendar', return_value=empty):
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_SESSION_CALENDAR'):
                sessions(datetime(2026, 9, 9, 18, tzinfo=timezone.utc),
                         self.publication()['published_at'], policy())
