import pytest

from two_dimensional_search import (
    best_feasible_result,
    inclusive_values,
    missing_grid_pairs,
    neighbouring_grid_values,
    search_convergence_summary,
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


def test_neighbours_bracket_a_saved_leader_that_is_off_the_current_grid():
    assert neighbouring_grid_values((8300, 8400, 8500), 8350) == (8300, 8400)


def test_neighbours_remain_adjacent_when_the_leader_is_on_the_grid():
    assert neighbouring_grid_values((8300, 8400, 8500), 8400) == (8300, 8500)


def test_neighbours_use_the_nearest_boundary_for_an_outside_leader():
    assert neighbouring_grid_values((8300, 8400, 8500), 8200) == (8300,)
    assert neighbouring_grid_values((8300, 8400, 8500), 8600) == (8500,)


def test_convergence_preserves_but_ignores_saved_points_off_current_grid():
    rows = [
        {
            "Shortlist cap": 8350,
            "Maximum assets": 550,
            "Status": "Feasible",
            "Annual Return": 0.70,
        },
        {
            "Shortlist cap": 8300,
            "Maximum assets": 550,
            "Status": "Feasible",
            "Annual Return": 0.68,
        },
    ]

    summary = search_convergence_summary(
        rows,
        shortlist_caps=(8300, 8400),
        maximum_assets_values=(550, 600),
    )

    assert summary["raw_peak"] is rows[1]
    assert summary["off_grid_completed"] == 1
    assert summary["completed"] == 1
