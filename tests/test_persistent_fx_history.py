import ast
from datetime import timedelta
from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(__file__).resolve().parents[1] / "portfolio_rebalancer_database.py"


def _load_fx_function(series_by_ticker):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "download_fx_history_to_inr"
    )
    function.decorator_list = []
    calls = []

    def fake_sync(path, tickers, start, end, fetcher, **kwargs):
        ticker = tickers[0]
        calls.append(ticker)
        series = series_by_ticker.get(ticker, pd.Series(dtype=float))
        closes = series.to_frame(ticker) if not series.empty else pd.DataFrame()
        return closes, pd.DataFrame(), {}

    namespace = {
        "pd": pd,
        "timedelta": timedelta,
        "sync_market_history": fake_sync,
        "MARKET_HISTORY_CACHE_PATH": Path("market-history.duckdb"),
        "_fetch_combined_daily_history": object(),
        "_currency_major_and_unit_factor": lambda currency: (str(currency), 1.0),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace["download_fx_history_to_inr"], calls


def test_usd_history_uses_yahoo_canonical_inr_alias_after_verbose_alias():
    index = pd.date_range("2026-01-02", periods=2, freq="D")
    expected = pd.Series([85.1, 85.2], index=index)
    download, calls = _load_fx_function({"INR=X": expected})

    result = download("USD", "2026-01-01", "2026-01-05")

    pd.testing.assert_series_equal(result, expected, check_names=False)
    assert calls[:2] == ["USDINR=X", "INR=X"]


def test_usd_history_can_invert_inr_usd_quote():
    index = pd.date_range("2026-01-02", periods=2, freq="D")
    inverse = pd.Series([1 / 85.0, 1 / 86.0], index=index)
    download, calls = _load_fx_function({"INRUSD=X": inverse})

    result = download("USD", "2026-01-01", "2026-01-05")

    pd.testing.assert_series_equal(
        result, pd.Series([85.0, 86.0], index=index), check_names=False
    )
    assert calls[:3] == ["USDINR=X", "INR=X", "INRUSD=X"]


def test_daily_return_pipeline_prefetches_fx_before_asset_history():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_daily_log_returns"
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    fx_download_line = min(
        node.lineno for node in calls if node.func.id == "download_fx_history_to_inr"
    )
    history_sync_line = min(
        node.lineno for node in calls if node.func.id == "sync_market_history"
    )
    conversion = next(
        node for node in calls if node.func.id == "convert_price_history_to_inr"
    )

    assert fx_download_line < history_sync_line
    assert any(keyword.arg == "fx_histories" for keyword in conversion.keywords)
