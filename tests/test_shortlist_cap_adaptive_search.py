import ast
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_search_function(namespace):
    source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "search_universal_shortlist_caps"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "portfolio_rebalancer_database.py", "exec"), namespace)
    return namespace["search_universal_shortlist_caps"]


class AdaptiveShortlistCapSearchTests(unittest.TestCase):
    def test_search_uses_big_jumps_and_reaches_natural_upper_endpoint(self):
        state = {"cap": None}

        def build_current_allocation_from_db(_owner):
            return pd.DataFrame({"Yahoo Ticker": ["OWNED.NS"]}), []

        def extend_allocation_with_universal_candidates(frame, *, maximum_candidates):
            state["cap"] = int(maximum_candidates)
            return frame, ["CANDIDATE"], {
                "eligible": 1100,
                "shortlisted": min(int(maximum_candidates), 1100),
            }

        def run_portfolio_analysis_multi(*_args, **_kwargs):
            cap = state["cap"]
            trading_days = {400: 700, 800: 400, 1100: 200}[cap]
            returns = pd.DataFrame(
                np.zeros((trading_days, 2)), columns=["OWNED.NS", "CANDIDATE"]
            )
            stats = {
                "Annual Return": cap / 10000.0,
                "Annual Volatility": 0.15,
                "Historical ES 95% (1 Session)": 0.02,
                "Block-Bootstrap ES 95% (20 Sessions)": 0.06,
                "Sharpe Ratio": 1.2,
            }
            return np.array([0.5, 0.5]), returns, None, stats, None

        search = _load_search_function(
            {
                "time": time,
                "np": np,
                "build_current_allocation_from_db": build_current_allocation_from_db,
                "extend_allocation_with_universal_candidates": (
                    extend_allocation_with_universal_candidates
                ),
                "run_portfolio_analysis_multi": run_portfolio_analysis_multi,
            }
        )

        rows, invalid_rows = search(
            "owner",
            starting_cap=400,
            max_dd=-0.2,
            target_volatility=None,
            drop_bottom_pct=0.2,
            history_buffer_days=30,
            redundancy_corr_threshold=0.8,
        )

        self.assertEqual(invalid_rows, [])
        self.assertEqual([row["Shortlist cap"] for row in rows], [400, 800, 1100])
        self.assertEqual(rows[0]["Next adaptive jump"], 400)
        self.assertEqual(rows[1]["Next adaptive jump"], 600)
        self.assertEqual(rows[-1]["Status"], "Below 252-session floor")
        self.assertEqual(rows[-1]["Next adaptive jump"], 0)

    def test_fifteen_minute_runtime_brake_halves_next_adaptive_jump(self):
        state = {"cap": None}

        class Clock:
            def __init__(self):
                self.values = iter([0.0, 901.0, 902.0, 903.0])

            def monotonic(self):
                return next(self.values)

        def build_current_allocation_from_db(_owner):
            return pd.DataFrame({"Yahoo Ticker": ["OWNED.NS"]}), []

        def extend_allocation_with_universal_candidates(frame, *, maximum_candidates):
            state["cap"] = int(maximum_candidates)
            return frame, [], {"eligible": 600, "shortlisted": int(maximum_candidates)}

        def run_portfolio_analysis_multi(*_args, **_kwargs):
            returns = pd.DataFrame(np.zeros((700, 1)), columns=["OWNED.NS"])
            stats = {
                "Annual Return": 0.30,
                "Annual Volatility": 0.15,
                "Historical ES 95% (1 Session)": 0.02,
                "Block-Bootstrap ES 95% (20 Sessions)": 0.06,
                "Sharpe Ratio": 1.2,
            }
            return np.array([1.0]), returns, None, stats, None

        search = _load_search_function(
            {
                "time": Clock(),
                "np": np,
                "build_current_allocation_from_db": build_current_allocation_from_db,
                "extend_allocation_with_universal_candidates": (
                    extend_allocation_with_universal_candidates
                ),
                "run_portfolio_analysis_multi": run_portfolio_analysis_multi,
            }
        )

        rows, _ = search(
            "owner",
            starting_cap=400,
            maximum_cap=600,
            adaptive=True,
            step=50,
            max_dd=-0.2,
            target_volatility=None,
            drop_bottom_pct=0.2,
            history_buffer_days=30,
            redundancy_corr_threshold=0.8,
        )

        self.assertEqual([row["Shortlist cap"] for row in rows], [400, 600])
        self.assertEqual(rows[0]["Next adaptive jump"], 200)
        self.assertEqual(rows[0]["Runtime brake"], "Applied")

    def test_fixed_search_honours_user_range_and_fifty_resolution(self):
        state = {"cap": None}

        def build_current_allocation_from_db(_owner):
            return pd.DataFrame({"Yahoo Ticker": ["OWNED.NS"]}), []

        def extend_allocation_with_universal_candidates(frame, *, maximum_candidates):
            state["cap"] = int(maximum_candidates)
            return frame, [], {"eligible": 5000, "shortlisted": int(maximum_candidates)}

        def run_portfolio_analysis_multi(*_args, **_kwargs):
            returns = pd.DataFrame(np.zeros((300, 1)), columns=["OWNED.NS"])
            stats = {
                "Annual Return": state["cap"] / 10000.0,
                "Annual Volatility": 0.15,
                "Historical ES 95% (1 Session)": 0.02,
                "Block-Bootstrap ES 95% (20 Sessions)": 0.06,
                "Sharpe Ratio": 1.2,
            }
            return np.array([1.0]), returns, None, stats, None

        search = _load_search_function(
            {
                "time": time,
                "np": np,
                "build_current_allocation_from_db": build_current_allocation_from_db,
                "extend_allocation_with_universal_candidates": (
                    extend_allocation_with_universal_candidates
                ),
                "run_portfolio_analysis_multi": run_portfolio_analysis_multi,
            }
        )

        rows, _ = search(
            "owner",
            starting_cap=400,
            maximum_cap=550,
            adaptive=False,
            step=50,
            max_dd=-0.2,
            target_volatility=None,
            drop_bottom_pct=0.2,
            history_buffer_days=30,
            redundancy_corr_threshold=0.8,
        )

        self.assertEqual([row["Shortlist cap"] for row in rows], [400, 450, 500, 550])


if __name__ == "__main__":
    unittest.main()
