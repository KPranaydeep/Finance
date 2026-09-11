"""Exercise persistence helpers without executing the Streamlit app."""
import ast
import sqlite3
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np


class ManualAddTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE TABLE master_holdings (owner TEXT, symbol TEXT, stock_name TEXT, yahoo_ticker TEXT, exchange TEXT, currency TEXT, quantity REAL, average_price REAL, added_at TEXT, updated_at TEXT, UNIQUE(owner,symbol))')
        source = Path(__file__).resolve().parents[1] / 'portfolio_rebalancer_database.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names = {'add_symbols_to_master', '_normalize_currency_code'}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.env = {'np': np, 'datetime': datetime,
                    'get_db_connection': lambda: self.db,
                    'get_nse_company_lookup': lambda: {},
                    'resolve_yahoo_instrument': self.resolve,
                    'get_latest_price_map': lambda tickers: {'AXTI': 12.5}}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), self.env)

    def tearDown(self):
        self.db.close()

    def resolve(self, ticker, lookup):
        if ticker == 'INVALID': return None
        return dict(symbol=ticker, yahoo_ticker=ticker, stock_name=ticker, exchange='NMS', currency='USD')

    def test_quantity_one_native_price_and_owner_isolation(self):
        add = self.env['add_symbols_to_master']
        self.assertEqual(add(['AXTI'], 'alice')[0], ['AXTI'])
        row = self.db.execute('SELECT * FROM master_holdings').fetchone()
        self.assertEqual((row['quantity'], row['average_price'], row['currency']), (1, 12.5, 'USD'))
        self.assertEqual(add(['AXTI'], 'bob')[0], ['AXTI'])

    def test_duplicate_does_not_reset_user_edit(self):
        add = self.env['add_symbols_to_master']
        add(['AXTI'], 'alice')
        self.db.execute('UPDATE master_holdings SET quantity=7, average_price=9')
        self.db.commit()
        self.assertEqual(add(['AXTI', 'AXTI'], 'alice')[1], ['AXTI'])
        row = self.db.execute('SELECT quantity,average_price FROM master_holdings').fetchone()
        self.assertEqual(tuple(row), (7, 9))

    def test_unknown_and_missing_prices_are_explicit(self):
        added, _, invalid, missing = self.env['add_symbols_to_master'](['INVALID', 'CWT'], 'alice')
        self.assertEqual((added, invalid, missing), (['CWT'], ['INVALID'], ['CWT']))
        self.assertIsNone(self.db.execute('SELECT average_price FROM master_holdings').fetchone()[0])
