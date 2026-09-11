import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from public_review.market import (sessions, entry_session, security_entry_schedule,
                                  AwaitingMarketEntry, _first_traded_intraday_bar,
                                  fetch_entry_quote)
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

    def test_nse_entry_can_freeze_during_market_hours(self):
        entry, ready = entry_session(datetime(2026, 9, 10, 4, 45, tzinfo=timezone.utc),
                                     self.publication()['published_at'], policy(),
                                     {'A.NS': 'equity', 'B.NS': 'listed_non_equity_etf'})
        self.assertEqual(entry, '2026-09-10')
        self.assertEqual(ready, '2026-09-10T04:45:00+00:00')

    def test_world_entry_waits_until_every_required_market_opens(self):
        kinds = {'A.NS': 'equity', 'AXTI': 'foreign_us_listing'}
        published = '2026-09-10T12:52:29+00:00'
        with self.assertRaises(AwaitingMarketEntry):
            entry_session(datetime(2026, 9, 11, 14, 29, tzinfo=timezone.utc),
                          published, policy(), kinds)
        entry, ready = entry_session(datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc),
                                     published, policy(), kinds)
        self.assertEqual(entry, '2026-09-11')
        self.assertEqual(ready, '2026-09-11T14:30:00+00:00')

    def test_world_assessment_waits_until_every_required_market_closes(self):
        kinds = {'A.NS': 'equity', 'AXTI': 'foreign_us_listing'}
        published = '2026-09-10T12:52:29+00:00'
        with self.assertRaises(AwaitingMarketEntry) as caught:
            sessions(datetime(2026, 9, 11, 20, 29, tzinfo=timezone.utc),
                     published, policy(), kinds)
        self.assertEqual(caught.exception.ready_at, '2026-09-11T20:30:00+00:00')
        entry, as_of, _ = sessions(datetime(2026, 9, 11, 20, 30, tzinfo=timezone.utc),
                                   published, policy(), kinds)
        self.assertEqual((entry, as_of), ('2026-09-11', '2026-09-11'))

    def test_each_security_uses_its_own_market_at_publication(self):
        published = '2026-09-10T14:00:29+00:00'
        entries = security_entry_schedule(
            datetime(2026, 9, 11, 5, tzinfo=timezone.utc), published, policy(),
            {'A.NS': 'equity', 'AXTI': 'foreign_us_listing'})
        self.assertEqual(entries['AXTI']['requested_entry_at'], published)
        self.assertEqual(entries['AXTI']['basis'], 'PUBLICATION_DURING_MARKET')
        self.assertEqual(entries['A.NS']['requested_entry_at'], '2026-09-11T04:45:00+00:00')
        self.assertEqual(entries['A.NS']['basis'], 'NEXT_OPEN_PLUS_CONFIGURED_WAIT')

    def test_entry_uses_first_trade_after_requested_time(self):
        index = pd.to_datetime(['2026-09-11T04:44:00Z', '2026-09-11T04:45:00Z',
                                '2026-09-11T04:52:00Z'])
        bars = pd.DataFrame({'Open': [99., 100., 101.], 'Volume': [10., 0., 20.]}, index=index)
        price, quote_at = _first_traded_intraday_bar(
            bars, pd.Timestamp('2026-09-11T04:45:00Z'),
            pd.Timestamp('2026-09-11T10:00:00Z'))
        self.assertEqual(price, 101.)
        self.assertEqual(quote_at, pd.Timestamp('2026-09-11T04:52:00Z'))

    def test_no_trade_defers_to_next_security_session(self):
        planned = {'ticker':'A.NS', 'kind':'equity', 'market':'NSE',
                   'requested_entry_at':'2026-09-11T04:45:00+00:00',
                   'session_open_at':'2026-09-11T03:45:00+00:00',
                   'session_close_at':'2026-09-11T10:00:00+00:00',
                   'entry_date':'2026-09-11', 'basis':'NEXT_OPEN_PLUS_CONFIGURED_WAIT',
                   'ready':True}
        bars = pd.DataFrame({'Open':[99.], 'Volume':[10.]},
                            index=pd.to_datetime(['2026-09-11T04:44:00Z']))
        deferred = {**planned, 'requested_entry_at':'2026-09-14T04:45:00+00:00',
                    'session_open_at':'2026-09-14T03:45:00+00:00',
                    'session_close_at':'2026-09-14T10:00:00+00:00',
                    'entry_date':'2026-09-14',
                    'basis':'DEFERRED_NEXT_OPEN_PLUS_CONFIGURED_WAIT', 'ready':False}
        with patch('public_review.market._intraday_frame', return_value=bars), \
             patch('public_review.market._next_session_entry', return_value=deferred):
            with self.assertRaises(AwaitingMarketEntry) as caught:
                fetch_entry_quote('A.NS', planned, policy(),
                                  now=datetime(2026,9,11,11,tzinfo=timezone.utc))
        self.assertEqual(caught.exception.ready_at, '2026-09-14T04:45:00+00:00')
        self.assertEqual(caught.exception.planned_entry['basis'],
                         'DEFERRED_NEXT_OPEN_PLUS_CONFIGURED_WAIT')

    def test_baseline_freezes_before_first_assessment(self):
        db = FakeDB()
        schedule = {ticker: {'ticker': ticker, 'kind': kind, 'market': 'NSE',
                    'requested_entry_at': '2026-09-10T04:45:00+00:00',
                    'entry_date': '2026-09-10', 'basis': 'NEXT_OPEN_PLUS_CONFIGURED_WAIT',
                    'ready': True} for ticker, kind in policy()['instrument_kinds'].items()}
        prices = {'A.NS': 100., 'B.NS': 50.}
        quote = lambda ticker, planned, p, now=None: {**planned, 'price_inr': prices[ticker],
                    'native_price': prices[ticker], 'fx_to_inr': 1.,
                    'quote_at': planned['requested_entry_at'], 'source': 'test'}
        pending = AwaitingMarketEntry('2026-09-10', '2026-09-10T10:30:00+00:00')
        with patch('public_review.service.publications', return_value=[self.publication()]), \
             patch('public_review.market.security_entry_schedule', return_value=schedule), \
             patch('public_review.market.fetch_entry_quote', side_effect=quote), \
             patch('public_review.market.sessions', side_effect=pending):
            result = run(db, 'TEST', policy(), now=datetime(2026, 9, 10, 5, tzinfo=timezone.utc))
        self.assertEqual((result['failed'], result['waiting']), (0, 1))
        self.assertEqual(result['results'][0]['reason'], 'AWAITING_FIRST_ASSESSMENT')
        self.assertEqual([row['kind'] for row in db.rows],
                         ['SECURITY_ENTRY', 'SECURITY_ENTRY', 'BASELINE', 'WAITING'])

    def test_real_error_still_fails_and_sanitizes_details(self):
        db = FakeDB()
        schedule = {ticker: {'ticker': ticker, 'kind': kind, 'market': 'NSE',
                    'requested_entry_at': '2026-09-10T04:45:00+00:00',
                    'entry_date': '2026-09-10', 'basis': 'NEXT_OPEN_PLUS_CONFIGURED_WAIT',
                    'ready': True} for ticker, kind in policy()['instrument_kinds'].items()}
        prices = {'A.NS': 100., 'B.NS': 50.}
        quote = lambda ticker, planned, p, now=None: {**planned, 'price_inr': prices[ticker],
                    'native_price': prices[ticker], 'fx_to_inr': 1.,
                    'quote_at': planned['requested_entry_at'], 'source': 'test'}
        with patch('public_review.service.publications', return_value=[self.publication()]), \
             patch('public_review.market.security_entry_schedule', return_value=schedule), \
             patch('public_review.market.fetch_entry_quote', side_effect=quote), \
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
                'entry_model_version': 'per-security-publication-or-open-plus-wait-v1',
                'entry_date': '2026-09-10', 'ready_at': '2026-09-10T10:30:00+00:00',
                'checked_at': '2026-09-09T18:00:00+00:00'}}]
            render_events(events, {'PUB-P006'}, datetime(2026, 9, 9, 18, tzinfo=timezone.utc),
                          latest_publication_id='PUB-P006')
        at = AppTest.from_function(app, default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.warning)
        self.assertEqual(len(at.metric), 0)
        self.assertTrue(any('Awaiting market entry' in x.value for x in at.info))
        self.assertTrue(any('Next pending entry check: 10 Sep 2026 16:00 IST' in x.value for x in at.caption))

    def test_empty_calendar_is_not_treated_as_waiting(self):
        import pandas as pd
        empty = pd.DataFrame({'market_open': pd.Series(dtype='datetime64[ns, UTC]'),
                              'market_close': pd.Series(dtype='datetime64[ns, UTC]')})
        with patch('public_review.market.calendar', return_value=empty):
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_SESSION_CALENDAR'):
                sessions(datetime(2026, 9, 9, 18, tzinfo=timezone.utc),
                         self.publication()['published_at'], policy())
