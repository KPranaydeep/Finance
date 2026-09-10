import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from public_review.instruments import parse_registry, complete_policy, sync_policy_file
from review_fixtures import policy

EQUITY = "SYMBOL,NAME OF COMPANY, SERIES,ISIN NUMBER\nA,Example Limited,EQ,INE000A01012\nF,Fund Limited,EQ,INE000A01013\n"
ETF = "Symbol,ETF Underlying,ISINNumber\nF,GLOBAL INDICES,INF000A01012\nG,COMMODITY,INF000A01013\nD,DEBT,INF000A01014\nE,EQUITY,INF000A01015\nH,Hybrid,INF000A01016\n"


class InstrumentTests(unittest.TestCase):
    def test_foreign_identification_preserves_review_block(self):
        from public_review.instruments import require_supported_review
        with patch('public_review.instruments._foreign_kind', return_value='foreign_us_listing'):
            result = complete_policy(policy(), ['AXTI'])
        self.assertEqual(result['instrument_kinds']['AXTI'], 'foreign_us_listing')
        with self.assertRaisesRegex(ValueError, 'FOREIGN_REVIEW_COST_MODEL_REQUIRED'):
            require_supported_review(['AXTI'])

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

    def test_file_sync_changes_only_kinds_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'policy.json'
            p = policy()
            path.write_text(json.dumps(p), encoding='utf-8')
            with patch.dict(os.environ, {'PUBLIC_REVIEW_POLICY_PATH': str(path)}), \
                 patch('public_review.instruments._registry', return_value={'NEW.NS':'equity'}):
                first = sync_policy_file(['NEW.NS'])
                second = sync_policy_file(['NEW.NS'])
            self.assertEqual(first, second)
            persisted = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(persisted.pop('instrument_kinds')['NEW.NS'], 'equity')
            p.pop('instrument_kinds')
            self.assertEqual(persisted, p)
