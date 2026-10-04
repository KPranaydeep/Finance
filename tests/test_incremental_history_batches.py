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
    def test_close_history_uses_stable_twelve_symbol_batches(self):
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
                "_resolve_history_window_end": lambda end, _buffer: (end, "2026-01-01"),
                "_chunked": chunked,
                "_download_close_history_batch": download_batch,
                "_format_download_failure_message": lambda *_args, **_kwargs: "failed",
            },
        )

        symbols = [f"TICKER{i:02d}" for i in range(25)]
        result = download(symbols)

        self.assertEqual([len(batch) for batch in requested], [12, 12, 1])
        self.assertEqual(list(result.columns), symbols)

    def test_full_return_cache_is_bounded(self):
        source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
        self.assertIn(
            '@st.cache_data(show_spinner=False, ttl="24h", max_entries=2)\n'
            "def get_daily_log_returns",
            source,
        )
        self.assertIn(
            '_download_close_history_batch = st.cache_data(\n'
            '    show_spinner=False,\n'
            '    ttl="24h",\n'
            '    max_entries=4096,',
            source,
        )


if __name__ == "__main__":
    unittest.main()
