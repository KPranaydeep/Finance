import unittest
from unittest.mock import patch
from public_review.instruments import parse_registry, complete_policy, sync_policy_file
from review_fixtures import policy

EQUITY = "SYMBOL,NAME OF COMPANY, SERIES,ISIN NUMBER\nA,Example Limited,EQ,INE000A01012\nF,Fund Limited,EQ,INE000A01013\n"
ETF = "Symbol,ETF Underlying,ISINNumber\nF,GLOBAL INDICES,INF000A01012\nG,COMMODITY,INF000A01013\nD,DEBT,INF000A01014\nE,EQUITY,INF000A01015\nH,Hybrid,INF000A01016\n"


class InstrumentTests(unittest.TestCase):
    def test_foreign_identification_is_supported_by_review_model(self):
        from public_review.instruments import require_supported_review
        with patch('public_review.instruments._foreign_kind', return_value='foreign_us_listing'):
            result = complete_policy(policy(), ['AXTI'])
        self.assertEqual(result['instrument_kinds']['AXTI'], 'foreign_us_listing')
        self.assertTrue(require_supported_review(['AXTI']))

    def test_explicit_metadata_categories(self):
        r = parse_registry(EQUITY, ETF)
        self.assertEqual(r, {'A.NS':'equity', 'F.NS':'listed_non_equity_etf',
                            'G.NS':'listed_non_equity_etf', 'D.NS':'specified_debt_etf',
                            'E.NS':'equity_etf'})

    def test_preserves_owner_entries_and_does_not_mutate(self):
        p = policy()
        p['instrument_kinds'] = {'A.NS':'equity'}
        q = complete_policy(p, ['A.NS','F.NS'], {'A.NS':'equity_etf','F.NS':'listed_non_equity_etf'})
        self.assertEqual(q['instrument_kinds']['A.NS'], 'equity')
        self.assertNotIn('F.NS', p['instrument_kinds'])
        self.assertEqual(q['policy_approved'], p['policy_approved'])

    def test_unknown_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'INSTRUMENT_CLASSIFICATION_REQUIRED'):
            complete_policy(policy(), ['UNKNOWN.NS'], {})

    def test_configured_names_need_no_network(self):
        with patch('public_review.instruments._registry') as fetch:
            complete_policy(policy(), ['A.NS'])
        fetch.assert_not_called()

    def test_invalid_feed_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'INSTRUMENT_METADATA_INVALID'):
            parse_registry('<html>blocked</html>', ETF)

    def test_compatibility_entry_point_resolves_without_policy_write(self):
        p = policy()
        p.pop('instrument_kinds')
        with patch('public_review.instruments.load_policy', create=True), \
             patch('public_review.config.load_policy', return_value=p), \
             patch('public_review.instruments._registry', return_value={'NEW.NS':'equity'}):
            result = sync_policy_file(['NEW.NS'])
        self.assertEqual(result['instrument_kinds'], {'NEW.NS':'equity'})
        self.assertNotIn('instrument_kinds', p)

    def test_policy_without_registry_is_fully_resolved(self):
        p = policy()
        p.pop('instrument_kinds')
        result = complete_policy(p, ['A.NS', 'F.NS'],
                                 {'A.NS':'equity', 'F.NS':'listed_non_equity_etf'})
        self.assertEqual(result['instrument_kinds'],
                         {'A.NS':'equity', 'F.NS':'listed_non_equity_etf'})
        self.assertNotIn('instrument_kinds', p)
