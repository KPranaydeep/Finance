import pytest

from two_dimensional_search import (
    best_feasible_result,
    inclusive_values,
    missing_grid_pairs,
    search_grid,
)


def test_requested_shortlist_range_is_inclusive_at_fifty_resolution():
    values = inclusive_values(8500, 9400, 50, minimum_step=50)
    assert values == tuple(range(8500, 9401, 50))
    assert len(values) == 19


def test_grid_crosses_every_cap_with_every_exact_asset_limit():
    pairs = search_grid((8500, 8550), (300, 350, 400))
    assert pairs == (
        (8500, 300),
        (8500, 350),
        (8500, 400),
        (8550, 300),
        (8550, 350),
        (8550, 400),
    )


def test_resume_skips_only_completed_pairs():
    missing = missing_grid_pairs(
        (8500, 8550),
        (300, 350),
        [{"Shortlist cap": 8500, "Maximum assets": 300}],
    )
    assert missing == ((8500, 350), (8550, 300), (8550, 350))


def test_best_result_uses_return_then_smaller_solver_and_shortlist_on_tie():
    rows = [
        {"Shortlist cap": 8500, "Maximum assets": 400, "Status": "Feasible", "Annual Return": 0.5},
        {"Shortlist cap": 8550, "Maximum assets": 350, "Status": "Feasible", "Annual Return": 0.5},
        {"Shortlist cap": 8500, "Maximum assets": 350, "Status": "Feasible", "Annual Return": 0.5},
        {"Shortlist cap": 9400, "Maximum assets": 500, "Status": "Solver failed", "Annual Return": 0.9},
    ]
    assert best_feasible_result(rows) is rows[2]


def test_invalid_range_is_rejected():
    with pytest.raises(ValueError):
        inclusive_values(9400, 8500, 50, minimum_step=50)
