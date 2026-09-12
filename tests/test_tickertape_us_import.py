"""Regression tests for Tickertape US broker-report ingestion."""
import ast
import io
import sys
import types
import unittest
from difflib import SequenceMatcher
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class Uploaded:
    name = "us_portfolio_report.csv"

    def __init__(self, payload):
        self.payload = payload

    def getvalue(self):
        return self.payload


class TickertapeUsImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "portfolio_rebalancer_database.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        wanted = {
            "_normalise_company_name",
            "resolve_us_instrument_by_name",
            "_read_tickertape_us_holdings_csv",
        }
        nodes = []
        aliases = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "BROKER_HOLDINGS_COLUMN_ALIASES"
                for target in node.targets
            ):
                aliases = node
            elif isinstance(node, ast.FunctionDef) and node.name in wanted:
                node.decorator_list = []
                nodes.append(node)
        cls.resolved = []
        cls.env = {
            "io": io,
            "pd": pd,
            "re": __import__("re"),
            "SequenceMatcher": SequenceMatcher,
            "resolve_yahoo_instrument": lambda symbol, _lookup: cls.resolved.append(symbol) or {"symbol": symbol},
        }
        exec(compile(ast.Module(body=[aliases, *nodes], type_ignores=[]), str(source), "exec"), cls.env)

    def test_reads_tickertape_us_columns_and_preserves_native_cost(self):
        payload = (
            b"Stock Name,1D Change,Quantity,LTP ($),Avg Buy Price ($),Current Value ($)\n"
            b"Consolidated Water Co Ltd,0.37,1,28.14,28.10,28.14\n"
        )
        result = self.env["_read_tickertape_us_holdings_csv"](Uploaded(payload))
        self.assertEqual(result.loc[0, "Stock Name"], "Consolidated Water Co Ltd")
        self.assertEqual(result.loc[0, "Quantity"], 1)
        self.assertEqual(result.loc[0, "Average Buy Price"], 28.10)
        self.assertEqual(result.loc[0, "Currency"], "USD")

    def test_name_resolution_accepts_us_listing_not_foreign_replica(self):
        quotes = [
            {"symbol": "CW2.F", "longname": "Consolidated Water Co Ltd", "quoteType": "EQUITY", "exchange": "FRA"},
            {"symbol": "CWCO", "longname": "Consolidated Water Co Ltd", "quoteType": "EQUITY", "exchange": "NMS"},
        ]
        fake_yf = types.SimpleNamespace(Search=lambda *_args, **_kwargs: types.SimpleNamespace(quotes=quotes))
        self.resolved.clear()
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            result = self.env["resolve_us_instrument_by_name"]("Consolidated Water Co Ltd")
        self.assertEqual(result, {"symbol": "CWCO"})
        self.assertEqual(self.resolved, ["CWCO"])

    def test_ambiguous_us_name_is_not_guessed(self):
        quotes = [
            {"symbol": "AAA", "longname": "Example Water", "quoteType": "EQUITY", "exchange": "NMS"},
            {"symbol": "BBB", "longname": "Example Water", "quoteType": "EQUITY", "exchange": "NYQ"},
        ]
        fake_yf = types.SimpleNamespace(Search=lambda *_args, **_kwargs: types.SimpleNamespace(quotes=quotes))
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            self.assertIsNone(self.env["resolve_us_instrument_by_name"]("Example Water Co"))


if __name__ == "__main__":
    unittest.main()
