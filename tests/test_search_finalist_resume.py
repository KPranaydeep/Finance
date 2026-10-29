import ast
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


SOURCE_PATH = Path("portfolio_rebalancer_database.py")


def _load_validator(missing_caps):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_search_finalists_walk_forward"
    )
    holdings = pd.DataFrame(
        {"Yahoo Ticker": ["TEST"], "Weight": [1.0], "Currency": ["USD"]}
    )
    rebuilt = []
    returns = pd.DataFrame(
        {"TEST": np.full(420, 0.001)},
        index=pd.date_range("2024-01-01", periods=420, freq="B"),
    )

    def build_current_allocation_from_db(_owner):
        return holdings.copy(), []

    def extend_allocation_with_universal_candidates(frame, *, maximum_candidates, owner):
        rebuilt.append(int(maximum_candidates))
        return frame, [], {}

    def run_portfolio_analysis_multi(*args, **kwargs):
        return None, returns.copy(), None, None, {}

    def optimize_prepared_return_matrix(train_returns, *_args, **_kwargs):
        return np.array([1.0]), train_returns, None, None, {}

    namespace = {
        "time": time,
        "np": np,
        "pd": pd,
        "TRADING_DAYS_PER_YEAR": 252,
        "build_current_allocation_from_db": build_current_allocation_from_db,
        "extend_allocation_with_universal_candidates": extend_allocation_with_universal_candidates,
        "run_portfolio_analysis_multi": run_portfolio_analysis_multi,
        "optimize_prepared_return_matrix": optimize_prepared_return_matrix,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<validator>", "exec"), namespace)
    return namespace["validate_search_finalists_walk_forward"], rebuilt, returns


def test_resume_rebuilds_missing_matrix_for_top_finalist_instead_of_skipping_it():
    validate, rebuilt, returns = _load_validator([])
    rows = [
        {"Status": "Feasible", "Shortlist cap": 9000, "Maximum assets": 200,
         "Annual Return": 0.30},
        {"Status": "Feasible", "Shortlist cap": 8500, "Maximum assets": 150,
         "Annual Return": 0.20},
    ]
    cached_holdings = pd.DataFrame(
        {"Yahoo Ticker": ["TEST"], "Weight": [1.0], "Currency": ["USD"]}
    )

    matrix_cache = {8500: (cached_holdings, returns.copy(), {})}
    result = validate(
        "user",
        rows,
        max_dd=0.2,
        target_volatility=None,
        drop_bottom_pct=0.2,
        history_buffer_days=0,
        redundancy_corr_threshold=0.8,
        prepared_matrices=matrix_cache,
    )

    assert rebuilt == [9000]
    assert 9000 in matrix_cache
    assert [row["Shortlist cap"] for row in result["finalists"]] == [9000, 8500]
    assert result["finalists"][0]["Status"] == "Passed"
    assert result["finalists"][1]["Status"] == "Passed"
    assert result["pairs_without_reusable_matrix"] == 0


def test_final_search_handoff_solves_from_prepared_matrix_without_history_reload():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "solve_prepared_search_finalist"
    )
    selected_returns = pd.DataFrame(
        {"TEST": np.full(10, 0.001)},
        index=pd.date_range("2026-01-01", periods=10, freq="B"),
    )
    calls = []

    def solve(matrix, allocation, **kwargs):
        calls.append((matrix, allocation, kwargs))
        return np.array([1.0]), selected_returns, {}, {"Annual Return": 0.25}, {
            "selected_assets": 1
        }

    namespace = {
        "time": time,
        "np": np,
        "pd": pd,
        "optimize_prepared_return_matrix": solve,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<prepared-solve>", "exec"), namespace)

    prepared_matrix = pd.DataFrame({"TEST": [0.01, 0.02]})
    allocation = pd.DataFrame(
        {"Yahoo Ticker": ["TEST"], "Weight": [1.0], "Quantity": [1]}
    )
    prepared_meta = {
        "universal_preselection": {"cap": 9000},
        "pipeline_timings_seconds": {"history": 12.0},
    }
    weights, output_returns, _, stats, meta = namespace[
        "solve_prepared_search_finalist"
    ](
        (allocation, prepared_matrix, prepared_meta),
        allocation,
        maximum_assets=1,
        max_dd=0.2,
        target_volatility=None,
    )

    assert len(calls) == 1
    assert calls[0][0] is prepared_matrix
    assert calls[0][2]["maximum_assets"] == 1
    assert weights.tolist() == [1.0]
    assert output_returns is selected_returns
    assert stats["Annual Return"] == 0.25
    assert meta["universal_preselection"]["cap"] == 9000
    assert "final_optimizer_solver" in meta["pipeline_timings_seconds"]

