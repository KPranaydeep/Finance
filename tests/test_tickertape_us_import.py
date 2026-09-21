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
    def __init__(self, payload, name="us_portfolio_report.csv"):
        self.payload = payload
        self.name = name

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
            "_detect_broker_holdings_header_row",
            "_read_holdings_preview",
            "_detect_holdings_report_type",
            "_read_indian_broker_holdings",
            "_indian_report_ticker_candidates",
            "_instrument_from_official_nse_symbol",
            "normalize_portfolio_symbol",
            "load_equity_mapping",
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
            "Path": Path,
            "__file__": str(source),
            "resolve_yahoo_instrument": lambda symbol, _lookup: cls.resolved.append(symbol) or {"symbol": symbol},
            "get_yahoo_metadata": lambda ticker: {
                "yahoo_ticker": ticker,
                "stock_name": ticker,
                "exchange": "NSE",
                "currency": "INR",
            },
            "_normalize_currency_code": lambda value: str(value).upper(),
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

    def test_schema_detection_keeps_groww_csv_in_india(self):
        payload = (
            b"Stock Name,ISIN,Quantity,Average Buy Price\n"
            b"SBC Exports Ltd,INE04AK01028,10,45.76\n"
        )
        upload = Uploaded(payload, "groww_holdings.csv")
        self.assertEqual(self.env["_detect_holdings_report_type"](upload), "INDIAN")
        result = self.env["_read_indian_broker_holdings"](upload)
        self.assertEqual(result.loc[0, "ISIN"], "INE04AK01028")
        self.assertEqual(result.loc[0, "Average Buy Price"], 45.76)

    def test_schema_detection_identifies_explicit_us_report(self):
        payload = (
            b"Stock Name,Quantity,Avg Buy Price ($),Current Value ($)\n"
            b"SBC Medical Group Holdings,2,4.50,9.00\n"
        )
        upload = Uploaded(payload)
        self.assertEqual(
            self.env["_detect_holdings_report_type"](upload), "TICKERTAPE_US"
        )

    def test_ambiguous_csv_fails_closed_instead_of_guessing_us(self):
        payload = b"Stock Name,Quantity,Average Price\nSBC,2,45.76\n"
        with self.assertRaisesRegex(ValueError, "cannot be identified safely"):
            self.env["_detect_holdings_report_type"](
                Uploaded(payload, "holdings.csv")
            )

    def test_indian_sbc_candidates_never_include_unsuffixed_us_symbol(self):
        candidates = self.env["_indian_report_ticker_candidates"](
            "SBC", "", {"SBC": "SBC Exports Limited"}
        )
        self.assertEqual(candidates, ["SBC.NS"])
        self.assertNotIn("SBC", candidates)

    def test_bse_hint_is_respected(self):
        candidates = self.env["_indian_report_ticker_candidates"](
            "500325", "BSE", {}
        )
        self.assertEqual(candidates, ["500325.BO"])

    def test_official_nse_identity_does_not_require_live_price_probe(self):
        instrument = self.env["_instrument_from_official_nse_symbol"](
            "SBC", "SBC Exports Limited"
        )
        self.assertEqual(instrument["symbol"], "SBC")
        self.assertEqual(instrument["yahoo_ticker"], "SBC.NS")
        self.assertEqual(instrument["currency"], "INR")

    def test_bundled_nse_mapping_contains_sbc_isin(self):
        mapping = self.env["load_equity_mapping"]()
        row = mapping.loc[mapping["ISIN"].eq("INE04AK01028")].iloc[0]
        self.assertEqual(row["Symbol"], "SBC")

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
