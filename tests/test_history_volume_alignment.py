import ast
import unittest
from pathlib import Path

import numpy as np
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


class HistoryVolumeAlignmentTests(unittest.TestCase):
    def test_volume_only_symbols_cannot_break_history_filtering(self):
        select = _load_function(
            "_select_drop_bottom_tickers_fallback",
            {
                "pd": pd,
                "np": np,
                "_percent_drop_count": lambda total, pct, minimum: min(
                    int(np.floor(total * pct)), total - minimum
                ),
            },
        )
        history = pd.DataFrame(
            {"AAA": [1.0, 2.0, 3.0], "BBB": [1.0, 2.0, 3.0]}
        )
        volume = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 120.0],
                "BBB": [200.0, 210.0, 220.0],
                "BJUL": [300.0, 310.0, 320.0],
                "IWM": [400.0, 410.0, 420.0],
            }
        )

        kept, dropped, _ = select(
            history,
            drop_bottom_pct=0.5,
            min_tickers_to_keep=1,
            volume_history=volume,
        )

        self.assertTrue(set(kept.index).issubset(history.columns))
        self.assertTrue(set(dropped.index).issubset(history.columns))

    def test_current_statistics_ignore_holdings_absent_from_return_matrix(self):
        stats_calls = []

        def portfolio_stats(weights, returns):
            stats_calls.append((np.asarray(weights), list(returns.columns)))
            return {"assets": list(returns.columns)}

        compare = _load_function(
            "portfolio_stats_comparison",
            {"pd": pd, "np": np, "portfolio_stats": portfolio_stats},
        )
        allocation = pd.DataFrame(
            {
                "Yahoo Ticker": ["AAA", "BJUL", "IWM"],
                "Weight": [0.5, 0.25, 0.25],
            }
        )
        returns = pd.DataFrame({"AAA": [0.01, -0.01], "CANDIDATE": [0.02, 0.01]})

        current, optimal = compare(allocation, returns, np.array([0.5, 0.5]))

        self.assertEqual(current["assets"], ["AAA"])
        self.assertEqual(optimal["assets"], ["AAA", "CANDIDATE"])
        self.assertEqual(len(stats_calls), 2)


if __name__ == "__main__":
    unittest.main()
