import ast
import sqlite3
from datetime import datetime
from pathlib import Path


def _load_helpers():
    source = Path(__file__).resolve().parents[1] / "portfolio_rebalancer_database.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    wanted = {
        "normalize_portfolio_symbol",
        "resolve_yahoo_instrument",
        "add_symbols_to_universal",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE master_holdings (
            owner TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stock_name TEXT NOT NULL,
            yahoo_ticker TEXT,
            exchange TEXT,
            currency TEXT,
            quantity REAL,
            average_price REAL,
            added_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (owner, symbol)
        )
        """
    )
    env = {
        "datetime": datetime,
        "UNIVERSAL_OWNER": "__universal__",
        "get_db_connection": lambda: conn,
        "get_nse_company_lookup": lambda: {"SBC": "SBC Exports Limited"},
        "_download_probe": lambda ticker: ticker in {"SBC", "SBC.NS"},
        "get_yahoo_metadata": lambda ticker: {
            "yahoo_ticker": ticker,
            "stock_name": (
                "SBC Medical Group Holdings" if ticker == "SBC"
                else "SBC Exports Limited"
            ),
            "exchange": "NGM" if ticker == "SBC" else "NSI",
            "currency": "USD" if ticker == "SBC" else "INR",
        },
        "_normalize_currency_code": lambda value: str(value).upper(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    return env, conn


def test_market_qualifier_disambiguates_same_symbol_across_exchanges():
    env, conn = _load_helpers()
    try:
        resolve = env["resolve_yahoo_instrument"]
        india = resolve("SBC", {"SBC": "SBC Exports Limited"})
        nasdaq = resolve("NASDAQ:SBC", {"SBC": "SBC Exports Limited"})

        assert (india["symbol"], india["yahoo_ticker"]) == ("SBC", "SBC.NS")
        assert (nasdaq["symbol"], nasdaq["yahoo_ticker"]) == (
            "NASDAQ:SBC",
            "SBC",
        )
    finally:
        conn.close()


def test_universal_portfolio_can_store_both_sbc_listings():
    env, conn = _load_helpers()
    try:
        added, duplicates, invalid = env["add_symbols_to_universal"](
            ["SBC", "NASDAQ:SBC"]
        )
        assert added == ["SBC", "NASDAQ:SBC"]
        assert duplicates == []
        assert invalid == []
        rows = conn.execute(
            """
            SELECT symbol, yahoo_ticker
            FROM master_holdings
            ORDER BY yahoo_ticker
            """
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("NASDAQ:SBC", "SBC"),
            ("SBC", "SBC.NS"),
        ]
    finally:
        conn.close()
