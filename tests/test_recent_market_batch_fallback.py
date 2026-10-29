import ast
import unittest
from pathlib import Path

import pandas as pd


def _load_download_function():
    source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    batch_node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_download_recent_market_data_batch"
    )
    bulk_node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_download_recent_market_data_bulk"
    )

    def chunked(values, size):
        for start in range(0, len(values), size):
            yield values[start : start + size]

    namespace = {
        "pd": pd,
        "_chunked": chunked,
        "apply_price_integrity_gate": lambda prices: (prices, pd.DataFrame()),
        "PRICE_INTEGRITY_VERSION": "test-v1",
    }
    exec(
        compile(
            ast.Module(body=[batch_node, bulk_node], type_ignores=[]),
            "<batch-download>",
            "exec",
        ),
        namespace,
    )
    return namespace


class RecentMarketBatchFallbackTests(unittest.TestCase):
    def test_empty_primary_batch_retries_smaller_batches(self):
        namespace = _load_download_function()

        def download(batch, **_kwargs):
            return [] if len(batch) > 2 else list(batch)

        def extract(downloaded, _expected):
            return pd.DataFrame([[1.0] * len(downloaded)], columns=downloaded)

        namespace["_yf_download_quiet"] = download
        namespace["_extract_close_prices_frame"] = extract
        namespace["_extract_volume_frame"] = extract

        closes, volumes, diagnostics, native_recent = namespace["_download_recent_market_data_bulk"](
            ["AAA", "BBB", "CCC", "DDD", "EEE"],
            batch_size=5,
            fallback_batch_size=2,
        )

        self.assertEqual(set(closes.columns), {"AAA", "BBB", "CCC", "DDD", "EEE"})
        self.assertEqual(set(volumes.columns), set(closes.columns))
        self.assertEqual(diagnostics["requests"], 4)
        self.assertEqual(diagnostics["failed_or_empty_requests"], 1)
        self.assertEqual(diagnostics["recovered"], 5)
        self.assertEqual(set(native_recent), set(closes.columns))

    def test_reports_primary_and_retry_progress_without_affecting_result(self):
        namespace = _load_download_function()
        updates = []

        def download(batch, **_kwargs):
            return list(batch[:1]) if len(batch) > 2 else list(batch)

        def extract(downloaded, _expected):
            return pd.DataFrame([[1.0] * len(downloaded)], columns=downloaded)

        namespace["_yf_download_quiet"] = download
        namespace["_extract_close_prices_frame"] = extract
        namespace["_extract_volume_frame"] = extract

        closes, _, diagnostics, _ = namespace["_download_recent_market_data_bulk"](
            ["AAA", "BBB", "CCC", "DDD", "EEE"],
            batch_size=5,
            fallback_batch_size=2,
            _progress_callback=lambda fraction, details: updates.append(
                (fraction, details.copy())
            ),
        )

        self.assertEqual(set(closes.columns), {"AAA", "BBB", "CCC", "DDD", "EEE"})
        self.assertEqual(diagnostics["requests"], 3)
        self.assertTrue(any(item[1].get("retry") for item in updates))
        self.assertTrue(any(item[1].get("validating_integrity") for item in updates))
        self.assertEqual(updates[-1][0], 1.0)


if __name__ == "__main__":
    unittest.main()
