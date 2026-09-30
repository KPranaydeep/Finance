"""Near-duplicate reduction must not manufacture exits from owned holdings."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_selector():
    source = Path(__file__).resolve().parents[1] / "portfolio_rebalancer_database.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "select_redundant_tickers"
    )
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["select_redundant_tickers"]


def return_frame():
    rng = np.random.default_rng(42)
    factor = rng.normal(0, .01, 80)
    return pd.DataFrame({
        "OWNED": factor,
        "LIQUID_CLONE": factor,
        "SECOND_OWNED": factor,
        "DIVERSIFIER": rng.normal(0, .01, 80),
    })


def test_owned_holding_survives_more_liquid_near_duplicate():
    select = load_selector()
    kept, report = select(
        return_frame()[["OWNED", "LIQUID_CLONE", "DIVERSIFIER"]],
        pd.Series({"OWNED": 10, "LIQUID_CLONE": 10_000,
                   "DIVERSIFIER": 100}),
        correlation_threshold=.80,
        aggregation_days=1,
        owned_tickers=("owned",),
    )
    assert "OWNED" in kept
    assert "LIQUID_CLONE" not in kept
    row = report.loc[report["Dropped"] == "LIQUID_CLONE"].iloc[0]
    assert row["Kept Instead"] == "OWNED"


def test_multiple_owned_duplicates_are_all_preserved():
    select = load_selector()
    kept, report = select(
        return_frame(),
        pd.Series({"OWNED": 10, "SECOND_OWNED": 20,
                   "LIQUID_CLONE": 10_000, "DIVERSIFIER": 100}),
        correlation_threshold=.80,
        aggregation_days=1,
        owned_tickers=("OWNED", "SECOND_OWNED"),
    )
    assert {"OWNED", "SECOND_OWNED", "DIVERSIFIER"}.issubset(kept)
    assert "LIQUID_CLONE" not in kept
    assert not report["Dropped"].isin(["OWNED", "SECOND_OWNED"]).any()


def test_candidate_only_cluster_keeps_highest_volume_security():
    select = load_selector()
    kept, report = select(
        return_frame()[["OWNED", "LIQUID_CLONE", "DIVERSIFIER"]],
        pd.Series({"OWNED": 10, "LIQUID_CLONE": 10_000,
                   "DIVERSIFIER": 100}),
        correlation_threshold=.80,
        aggregation_days=1,
    )
    assert "LIQUID_CLONE" in kept
    assert "OWNED" not in kept
    row = report.loc[report["Dropped"] == "OWNED"].iloc[0]
    assert row["Kept Instead"] == "LIQUID_CLONE"
