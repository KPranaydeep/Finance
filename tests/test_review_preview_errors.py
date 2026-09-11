import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from streamlit.testing.v1 import AppTest
from public_review.preview import historical_preview
from review_fixtures import policy


class PreviewErrorTests(unittest.TestCase):
    def test_empty_classifications_reproduces_failure_before_fetch(self):
        p = policy()
        p['instrument_kinds'] = {}
        pub = {'publication_id':'P006', 'basket_id':'TEST', 'portfolio_version':6,
               'published_at':'2026-09-09T14:11:00+00:00', 'weights':{'A.NS':1.}}
        with patch('public_review.market.fetch') as fetch, patch('public_review.instruments._registry', return_value={}):
            with self.assertRaisesRegex(ValueError, 'INSTRUMENT_CLASSIFICATION_REQUIRED'):
                historical_preview(pub, p, [], datetime(2026,9,10,3,30,tzinfo=timezone.utc))
        fetch.assert_not_called()

    def test_configured_security_reaches_price_fetch(self):
        pub = {'publication_id':'P006', 'basket_id':'TEST', 'portfolio_version':6,
               'published_at':'2026-09-09T14:11:00+00:00', 'weights':{'A.NS':1.}}
        quote = lambda ticker, entry, policy_, now=None: {**entry, 'price_inr':100., 'native_price':100.,
                'fx_to_inr':1., 'quote_at':entry['requested_entry_at'], 'source':'test'}
        with patch('public_review.market.fetch_entry_quote', side_effect=quote), \
             patch('public_review.market.fetch', side_effect=ValueError('TEST_FETCH_REACHED')):
            with self.assertRaisesRegex(ValueError, 'TEST_FETCH_REACHED'):
                historical_preview(pub, policy(), [], datetime(2026,9,10,11,tzinfo=timezone.utc))

    def test_ui_explains_missing_classifications(self):
        def app():
            from unittest.mock import patch
            from public_review.ui import render_review_panel
            with patch('public_review.ui.load_fresh_preview', side_effect=ValueError('INSTRUMENT_CLASSIFICATION_REQUIRED')), patch('public_review.ui.load_events', return_value=[]):
                render_review_panel('TEST', [{'publication_id':'P006'}])
        at = AppTest.from_function(app).run()
        self.assertFalse(at.exception)
        self.assertTrue(any('INSTRUMENT_CLASSIFICATION_REQUIRED' in x.value for x in at.warning))
