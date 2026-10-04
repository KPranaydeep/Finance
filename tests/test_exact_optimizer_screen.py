import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_screen():
    source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "screen_log_returns_for_exact_optimizer"
    )
    namespace = {
        "np": np,
        "pd": pd,
        "DEFAULT_EXACT_OPTIMIZER_ASSET_CAP": 300,
        "TRADING_DAYS_PER_YEAR": 252,
        "RISK_FREE_RATE_ANNUAL": 0.112,
    }
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "portfolio_rebalancer_database.py", "exec"), namespace)
    return namespace["screen_log_returns_for_exact_optimizer"]


class ExactOptimizerScreenTests(unittest.TestCase):
    def test_screen_caps_solver_assets_and_protects_owned_holdings(self):
        rng = np.random.default_rng(42)
        columns = [f"TICKER{i:03d}" for i in range(500)]
        returns = pd.DataFrame(
            rng.normal(0.0004, 0.01, size=(400, len(columns))),
            columns=columns,
        )
        returns["TICKER499"] = rng.normal(-0.001, 0.02, size=400)

        selected, report = _load_screen()(
            returns,
            maximum_assets=100,
            owned_tickers=("TICKER499",),
        )

        self.assertEqual(selected.shape[1], 100)
        self.assertIn("TICKER499", selected.columns)
        self.assertEqual(report["eligible_assets"], 500)
        self.assertEqual(report["selected_assets"], 100)
        self.assertEqual(report["owned_assets_protected"], 1)

    def test_screen_leaves_smaller_matrix_unchanged(self):
        returns = pd.DataFrame({"AAA": [0.01, -0.01], "BBB": [0.02, 0.01]})

        selected, report = _load_screen()(returns, maximum_assets=300)

        pd.testing.assert_frame_equal(selected, returns)
        self.assertEqual(report["method"], "all-assets-fit")


if __name__ == "__main__":
    unittest.main()
