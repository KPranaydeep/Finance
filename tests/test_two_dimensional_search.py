import pytest

from two_dimensional_search import (
    adaptive_anchor_values,
    best_feasible_result,
    coarse_to_fine_values,
    inclusive_values,
    local_refinement_values,
    missing_grid_pairs,
    robust_feasible_result,
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


def test_complete_order_covers_extremes_then_bisects_without_omissions():
    values = tuple(range(100, 451, 50))
    ordered = coarse_to_fine_values(values)
    assert ordered[:3] == (450, 100, 250)
    assert set(ordered) == set(values)
    assert len(ordered) == len(values)


def test_adaptive_anchors_cover_range_and_refine_around_current_best():
    values = tuple(range(100, 1801, 50))
    anchors = adaptive_anchor_values(values, target_points=8)
    assert anchors[0] == 1800
    assert 100 in anchors
    assert len(anchors) == 8
    neighbors = local_refinement_values(values, anchors, 1100, radius=2)
    assert all(value not in anchors for value in neighbors)
    assert set(neighbors).issubset({1000, 1050, 1150, 1200})


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


def test_robust_result_uses_downside_risk_inside_return_plateau():
    rows = [
        {
            "Shortlist cap": 8650,
            "Maximum assets": 1100,
            "Status": "Feasible",
            "Annual Return": 0.9231,
            "Block-Bootstrap ES 95% (20 Sessions)": 0.08,
            "Historical ES 95% (1 Session)": 0.03,
            "Annual Volatility": 0.16,
        },
        {
            "Shortlist cap": 8650,
            "Maximum assets": 1250,
            "Status": "Feasible",
            "Annual Return": 0.9236,
            "Block-Bootstrap ES 95% (20 Sessions)": 0.09,
            "Historical ES 95% (1 Session)": 0.03,
            "Annual Volatility": 0.16,
        },
        {
            "Shortlist cap": 8650,
            "Maximum assets": 1000,
            "Status": "Feasible",
            "Annual Return": 0.9225,
            "Block-Bootstrap ES 95% (20 Sessions)": 0.06,
            "Historical ES 95% (1 Session)": 0.02,
            "Annual Volatility": 0.15,
        },
    ]
    # 1,000 lies 0.11 percentage points below the raw peak, so the default
    # plateau excludes it and selects the safer 1,100 result.
    assert robust_feasible_result(rows) is rows[0]
    # A wider user-selected tolerance admits 1,000 and lets risk decide.
    assert robust_feasible_result(rows, return_tolerance=0.0015) is rows[2]


def test_convergence_reports_incomplete_then_boundary_then_interior():
    caps = (8500, 8550, 8600)
    assets = (1000, 1050, 1100)
    partial = [
        {
            "Shortlist cap": 8500,
            "Maximum assets": 1000,
            "Status": "Feasible",
            "Annual Return": 0.8,
        }
    ]
    assert search_convergence_summary(partial, caps, assets)["state"] == "boundary_limited"

    full = [
        {
            "Shortlist cap": cap,
            "Maximum assets": asset,
            "Status": "Feasible",
            "Annual Return": 0.9 - abs(cap - 8550) / 100_000 - abs(asset - 1050) / 10_000,
        }
        for cap, asset in search_grid(caps, assets)
    ]
    summary = search_convergence_summary(full, caps, assets)
    assert summary["state"] == "global_verified"
    assert summary["boundary_axes"] == ()

    full[-1]["Annual Return"] = 0.95
    summary = search_convergence_summary(full, caps, assets)
    assert summary["state"] == "global_verified"
    assert summary["boundary_axes"]
    assert summary["boundary_axes"] == ("shortlist cap", "Maximum assets")


def test_adaptive_search_reports_local_convergence_after_all_neighbours():
    caps = (8500, 8550, 8600, 8650, 8700)
    assets = (1000, 1050, 1100, 1150, 1200)
    rows = [
        {
            "Shortlist cap": cap,
            "Maximum assets": asset,
            "Status": "Feasible",
            "Annual Return": 0.90 if (cap, asset) == (8600, 1100) else 0.89,
        }
        for cap in (8550, 8600, 8650)
        for asset in (1050, 1100, 1150)
    ]
    summary = search_convergence_summary(rows, caps, assets)
    assert summary["state"] == "locally_converged"
    assert summary["neighbourhood_complete"] is True
