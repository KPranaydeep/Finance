import ast
from pathlib import Path

import numpy as np
import pandas as pd


def load_function(name):
    source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "pd": pd}
    exec(compile(module, "portfolio_rebalancer_database.py", "exec"), namespace)
    return namespace[name]


def row(cap, annual_return, es=0.06, status="Feasible"):
    return {
        "Shortlist cap": cap,
        "Annual Return": annual_return,
        "Block-Bootstrap ES 95% (30 Sessions)": es,
        "Status": status,
    }


def test_refinement_bisects_left_and_right_intervals_around_best():
    choose = load_function("local_refinement_cap_pair")
    results = [row(3000, 0.40), row(3500, 0.50), row(4200, 0.45)]

    assert choose(
        results,
        minimum_cap=2500,
        maximum_cap=5000,
        minimum_step=50,
    ) == [3250, 3850]


def test_refinement_moves_with_new_best_and_reduces_interval():
    choose = load_function("local_refinement_cap_pair")
    results = [
        row(3000, 0.40),
        row(3250, 0.44),
        row(3500, 0.50),
        row(3850, 0.53),
        row(4200, 0.45),
    ]

    assert choose(
        results,
        minimum_cap=2500,
        maximum_cap=5000,
        minimum_step=50,
    ) == [3700, 4050]


def test_material_gain_requires_return_and_es_guard():
    material = load_function("refinement_is_material")
    baseline = row(3500, 0.50, es=0.06)

    assert material(baseline, [row(3600, 0.515, es=0.064)])
    assert not material(baseline, [row(3600, 0.505, es=0.060)])
    assert not material(baseline, [row(3600, 0.520, es=0.071)])
