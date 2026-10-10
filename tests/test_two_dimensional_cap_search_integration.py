import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_search(namespace):
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


def test_each_shortlist_prepares_history_once_then_solves_every_asset_limit():
    state = {"cap": None, "prepared": 0, "smaller": []}

    def build_current_allocation_from_db(_owner):
        return pd.DataFrame(
            {
                "Yahoo Ticker": ["OWNED.NS"],
                "Quantity": [1.0],
                "Weight": [1.0],
            }
        ), []

    def extend(frame, *, maximum_candidates, owner=None):
        state["cap"] = int(maximum_candidates)
        return frame, [], {"eligible": 10_000, "shortlisted": int(maximum_candidates)}

    def prepare(*_args, **kwargs):
        state["prepared"] += 1
        assert kwargs["exact_optimizer_asset_cap"] == 400
        returns = pd.DataFrame(
            np.zeros((300, 400)),
            columns=[f"T{index}" for index in range(400)],
        )
        stats = {
            "Annual Return": state["cap"] / 10_000,
            "Annual Volatility": 0.15,
            "Historical ES 95% (1 Session)": 0.02,
            "Block-Bootstrap ES 95% (20 Sessions)": 0.06,
            "Sharpe Ratio": 1.0,
        }
        return np.ones(400) / 400, returns, None, stats, {
            "exact_optimizer_screen": {"selected_assets": 400}
        }

    def solve(prepared, _allocation, *, maximum_assets, **_kwargs):
        state["smaller"].append((state["cap"], int(maximum_assets)))
        selected = prepared.iloc[:, : int(maximum_assets)]
        stats = {
            "Annual Return": state["cap"] / 10_000 + maximum_assets / 100_000,
            "Annual Volatility": 0.15,
            "Historical ES 95% (1 Session)": 0.02,
            "Block-Bootstrap ES 95% (20 Sessions)": 0.06,
            "Sharpe Ratio": 1.0,
        }
        return (
            np.ones(maximum_assets) / maximum_assets,
            selected,
            None,
            stats,
            {"selected_assets": maximum_assets},
        )

    search = load_search(
        {
            "time": time,
            "np": np,
            "build_current_allocation_from_db": build_current_allocation_from_db,
            "extend_allocation_with_universal_candidates": extend,
            "run_portfolio_analysis_multi": prepare,
            "optimize_prepared_return_matrix": solve,
        }
    )
    rows, _ = search(
        "owner",
        starting_cap=8500,
        maximum_cap=8550,
        step=50,
        adaptive=False,
        exact_optimizer_asset_caps=(300, 350, 400),
        max_dd=-0.2,
        target_volatility=None,
        drop_bottom_pct=0.2,
        history_buffer_days=30,
        redundancy_corr_threshold=0.8,
    )

    assert state["prepared"] == 2
    assert state["smaller"] == [
        (8500, 300),
        (8500, 350),
        (8550, 300),
        (8550, 350),
    ]
    assert [(row["Shortlist cap"], row["Maximum assets"]) for row in rows] == [
        (8500, 300),
        (8500, 350),
        (8500, 400),
        (8550, 300),
        (8550, 350),
        (8550, 400),
    ]
    assert all(row["Status"] == "Feasible" for row in rows)
