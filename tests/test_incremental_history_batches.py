import ast
import unittest
from pathlib import Path

import pandas as pd


def _load_function(name, namespace):
    source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "portfolio_rebalancer_database.py", "exec"), namespace)
    return namespace[name]


class IncrementalHistoryBatchTests(unittest.TestCase):
    def test_close_history_uses_stable_large_batches(self):
        requested = []

        def chunked(values, size):
            for start in range(0, len(values), size):
                yield values[start : start + size]

        def download_batch(symbols, _start, _end):
            requested.append(tuple(symbols))
            return pd.DataFrame([[1.0] * len(symbols)], columns=symbols), {}

        download = _load_function(
            "download_close_history",
            {
                "pd": pd,
                "DEFAULT_HISTORY_START_DATE": "2020-01-01",
                "FULL_HISTORY_BATCH_SIZE": 120,
                "_resolve_history_window_end": lambda end, _buffer: (end, "2026-01-01"),
                "_chunked": chunked,
                "_download_close_history_batch": download_batch,
                "_download_close_prices_resilient": lambda *_args, **_kwargs: (
                    pd.DataFrame(),
                    {},
                ),
                "_format_download_failure_message": lambda *_args, **_kwargs: "failed",
                "apply_price_integrity_gate": lambda prices, owned_tickers=(): (
                    prices,
                    pd.DataFrame(),
                ),
                "PRICE_INTEGRITY_VERSION": "test-v1",
            },
        )

        symbols = [f"TICKER{i:03d}" for i in range(250)]
        result = download(symbols)

        self.assertEqual([len(batch) for batch in requested], [120, 120, 10])
        self.assertEqual(list(result.columns), symbols)

    def test_close_history_retries_only_missing_owned_symbols(self):
        requested = []
        recovered_owned = []

        def chunked(values, size):
            for start in range(0, len(values), size):
                yield values[start : start + size]

        def download_batch(symbols, _start, _end):
            requested.append(tuple(symbols))
            available = [symbol for symbol in symbols if symbol != "OWNED"]
            return pd.DataFrame([[1.0] * len(available)], columns=available), {}

        def recover(symbols, **_kwargs):
            recovered_owned.extend(symbols)
            return pd.DataFrame([[1.0] * len(symbols)], columns=symbols), {}

        download = _load_function(
            "download_close_history",
            {
                "pd": pd,
                "DEFAULT_HISTORY_START_DATE": "2020-01-01",
                "FULL_HISTORY_BATCH_SIZE": 120,
                "_resolve_history_window_end": lambda end, _buffer: (end, "2026-01-01"),
                "_chunked": chunked,
                "_download_close_history_batch": download_batch,
                "_download_close_prices_resilient": recover,
                "_format_download_failure_message": lambda *_args, **_kwargs: "failed",
                "apply_price_integrity_gate": lambda prices, owned_tickers=(): (
                    prices,
                    pd.DataFrame(),
                ),
                "PRICE_INTEGRITY_VERSION": "test-v1",
            },
        )

        result = download(["AAA", "OWNED", "CANDIDATE"], owned_tickers=("OWNED",))

        self.assertEqual(requested, [("AAA", "OWNED", "CANDIDATE")])
        self.assertEqual(recovered_owned, ["OWNED"])
        self.assertEqual(set(result.columns), {"AAA", "OWNED", "CANDIDATE"})

    def test_history_batch_cache_is_bounded(self):
        source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
        self.assertIn(
            '_download_close_history_batch = st.cache_data(\n'
            '    show_spinner=False,\n'
            '    ttl="24h",\n'
            '    max_entries=256,',
            source,
        )


if __name__ == "__main__":
    unittest.main()
