import unittest
from streamlit.testing.v1 import AppTest


def review_window_app():
    from datetime import datetime
    import streamlit as st
    from public_review.ui import render_operational_window
    render_operational_window({
        'operational_review_window': {
            'start_at': '2026-10-31T08:00:00+05:30',
            'end_at': '2026-10-31T09:00:00+05:30',
            'date_label': '31 OCT', 'time_label': '08:00-09:00 IST',
            'market_context': 'Post NYSE | Pre NSE',
            'execution_windows': {'NSE': '2026-11-02T09:30:00+05:30'},
        }
    }, now=datetime.fromisoformat(st.session_state['check_time']))


class ReviewWindowDisplayTests(unittest.TestCase):
    def display(self, now):
        at = AppTest.from_function(review_window_app, default_timeout=20)
        at.session_state['check_time'] = now
        at.run()
        self.assertFalse(at.exception)
        return at

    def test_past_window_is_historical_and_does_not_recommend_old_execution(self):
        at = self.display('2026-11-01T09:03:00+05:30')
        self.assertTrue(any('Previous planned review window' in c.value for c in at.caption))
        self.assertFalse(at.info)
        self.assertFalse(any('Earliest post-review execution' in c.value for c in at.caption))

    def test_future_window_remains_recommended(self):
        at = self.display('2026-10-31T07:00:00+05:30')
        self.assertTrue(any('Recommended one-hour review' in c.value for c in at.info))

    def test_started_window_is_current_rather_than_a_new_one_hour_recommendation(self):
        at = self.display('2026-10-31T08:30:00+05:30')
        self.assertTrue(any('Current planned review window' in c.value for c in at.info))
        self.assertFalse(any('Recommended one-hour review' in c.value for c in at.info))

    def test_exact_end_is_already_elapsed(self):
        at = self.display('2026-10-31T09:00:00+05:30')
        self.assertTrue(any('passed' in c.value for c in at.caption))
        self.assertFalse(at.info)
