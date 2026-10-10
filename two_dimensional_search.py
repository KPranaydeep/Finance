"""Pure helpers for exhaustive shortlist-cap × exact-asset searches."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def inclusive_values(start: int, through: int, step: int, *, minimum_step: int) -> tuple[int, ...]:
    """Return an aligned inclusive range with an explicit minimum resolution."""
    start = int(start)
    through = int(through)
    step = max(int(step), int(minimum_step))
    if through < start:
        raise ValueError("Search range end must be greater than or equal to its start.")
    values = list(range(start, through + 1, step))
    if values[-1] != through:
        values.append(through)
    return tuple(dict.fromkeys(values))


def search_grid(
    shortlist_caps: Iterable[int],
    maximum_assets_values: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    """Return the deterministic Cartesian search grid."""
    caps = sorted({int(value) for value in shortlist_caps})
    asset_limits = sorted({int(value) for value in maximum_assets_values})
    if not caps or not asset_limits:
        raise ValueError("Both search dimensions require at least one value.")
    if caps[0] < 1 or asset_limits[0] < 1:
        raise ValueError("Search values must be positive integers.")
    return tuple((cap, asset_limit) for cap in caps for asset_limit in asset_limits)


def coarse_to_fine_values(values: Iterable[int]) -> tuple[int, ...]:
    """Order a complete integer grid for useful broad coverage as early as possible."""
    ordered_values = sorted({int(value) for value in values})
    if not ordered_values:
        return ()
    emitted = []
    seen = set()

    def emit(index):
        value = ordered_values[index]
        if value not in seen:
            seen.add(value)
            emitted.append(value)

    emit(len(ordered_values) - 1)  # Prepared by the history pass, so expose it first.
    emit(0)
    intervals = [(0, len(ordered_values) - 1)]
    while intervals:
        intervals.sort(key=lambda pair: pair[1] - pair[0], reverse=True)
        lower, upper = intervals.pop(0)
        if upper - lower <= 1:
            continue
        midpoint = (lower + upper) // 2
        emit(midpoint)
        intervals.extend(((lower, midpoint), (midpoint, upper)))
    return tuple(emitted)


def adaptive_anchor_values(
    values: Iterable[int],
    *,
    target_points: int = 8,
) -> tuple[int, ...]:
    """Return broad anchors for a bounded anytime search of one dimension."""
    ordered_values = sorted({int(value) for value in values})
    if len(ordered_values) <= max(int(target_points), 2):
        return coarse_to_fine_values(ordered_values)
    intervals = max(int(target_points) - 1, 1)
    indices = {
        round(index * (len(ordered_values) - 1) / intervals)
        for index in range(intervals + 1)
    }
    anchors = [ordered_values[index] for index in sorted(indices)]
    return tuple(
        value for value in coarse_to_fine_values(anchors)
    )


def local_refinement_values(
    values: Iterable[int],
    evaluated: Iterable[int],
    best_value: int,
    *,
    radius: int = 2,
) -> tuple[int, ...]:
    """Return unevaluated grid neighbors around the current best value."""
    ordered_values = sorted({int(value) for value in values})
    if not ordered_values or int(best_value) not in ordered_values:
        return ()
    completed = {int(value) for value in evaluated}
    center = ordered_values.index(int(best_value))
    candidates = []
    for distance in range(1, max(int(radius), 1) + 1):
        for index in (center - distance, center + distance):
            if 0 <= index < len(ordered_values):
                value = ordered_values[index]
                if value not in completed and value not in candidates:
                    candidates.append(value)
    return tuple(candidates)


def missing_grid_pairs(
    shortlist_caps: Iterable[int],
    maximum_assets_values: Iterable[int],
    prior_results: Sequence[Mapping[str, object]] = (),
) -> tuple[tuple[int, int], ...]:
    """Return only combinations not already present in resumable results."""
    completed = {
        (int(row["Shortlist cap"]), int(row["Maximum assets"]))
        for row in prior_results
        if row.get("Shortlist cap") is not None
        and row.get("Maximum assets") is not None
    }
    return tuple(
        pair
        for pair in search_grid(shortlist_caps, maximum_assets_values)
        if pair not in completed
    )


def best_feasible_result(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Select the highest-return feasible pair with deterministic tie-breaks."""
    best = None
    best_key = None
    for row in rows:
        if str(row.get("Status")) != "Feasible":
            continue
        try:
            annual_return = float(row["Annual Return"])
            cap = int(row["Shortlist cap"])
            maximum_assets = int(row["Maximum assets"])
        except (KeyError, TypeError, ValueError):
            continue
        if annual_return != annual_return:
            continue
        key = (annual_return, -maximum_assets, -cap)
        if best_key is None or key > best_key:
            best = row
            best_key = key
    return best


def robust_feasible_result(
    rows: Sequence[Mapping[str, object]],
    *,
    return_tolerance: float = 0.001,
) -> Mapping[str, object] | None:
    """Choose a lower-risk, simpler result from the near-maximum plateau.

    ``return_tolerance`` is expressed as a decimal annual return. The default
    0.001 therefore treats results within 0.10 percentage points of the raw
    maximum as economically equivalent. Missing risk values sort last.
    """
    feasible = []
    for row in rows:
        if str(row.get("Status")) != "Feasible":
            continue
        try:
            annual_return = float(row["Annual Return"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(annual_return):
            continue
        feasible.append((row, annual_return))
    if not feasible:
        return None

    peak_return = max(value for _, value in feasible)
    plateau = [
        row for row, value in feasible
        if value >= peak_return - max(float(return_tolerance), 0.0)
    ]

    def finite_or_infinity(row: Mapping[str, object], key: str) -> float:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    return min(
        plateau,
        key=lambda row: (
            finite_or_infinity(row, "Block-Bootstrap ES 95% (20 Sessions)"),
            finite_or_infinity(row, "Historical ES 95% (1 Session)"),
            finite_or_infinity(row, "Annual Volatility"),
            int(row.get("Maximum assets") or 0),
            int(row.get("Shortlist cap") or 0),
            finite_or_infinity(row, "Elapsed seconds"),
        ),
    )


def search_convergence_summary(
    rows: Sequence[Mapping[str, object]],
    shortlist_caps: Iterable[int],
    maximum_assets_values: Iterable[int],
    *,
    return_tolerance: float = 0.001,
) -> dict[str, object]:
    """Describe grid completion and whether the raw peak is boundary-limited."""
    grid = search_grid(shortlist_caps, maximum_assets_values)
    completed = {
        (int(row["Shortlist cap"]), int(row["Maximum assets"]))
        for row in rows
        if row.get("Shortlist cap") is not None
        and row.get("Maximum assets") is not None
    }
    peak = best_feasible_result(rows)
    robust = robust_feasible_result(rows, return_tolerance=return_tolerance)
    complete = all(pair in completed for pair in grid)
    if peak is None:
        return {
            "state": "no_feasible_result",
            "complete": complete,
            "completed": len(completed.intersection(grid)),
            "total": len(grid),
            "boundary_axes": (),
            "raw_peak": None,
            "robust_choice": robust,
        }

    caps = sorted({pair[0] for pair in grid})
    assets = sorted({pair[1] for pair in grid})
    peak_cap = int(peak["Shortlist cap"])
    peak_assets = int(peak["Maximum assets"])
    boundary_axes = []
    if len(caps) > 1 and peak_cap in {caps[0], caps[-1]}:
        boundary_axes.append("shortlist cap")
    if len(assets) > 1 and peak_assets in {assets[0], assets[-1]}:
        boundary_axes.append("Maximum assets")

    cap_index = caps.index(peak_cap)
    asset_index = assets.index(peak_assets)
    neighbours = {
        (caps[x_index], assets[y_index])
        for x_index in range(max(0, cap_index - 1), min(len(caps), cap_index + 2))
        for y_index in range(max(0, asset_index - 1), min(len(assets), asset_index + 2))
        if (x_index, y_index) != (cap_index, asset_index)
    }
    neighbourhood_complete = bool(neighbours) and neighbours.issubset(completed)

    if complete:
        state = "global_verified"
    elif boundary_axes:
        state = "boundary_limited"
    elif neighbourhood_complete:
        state = "locally_converged"
    else:
        state = "incomplete"
    return {
        "state": state,
        "complete": complete,
        "completed": len(completed.intersection(grid)),
        "total": len(grid),
        "boundary_axes": tuple(boundary_axes),
        "neighbourhood_complete": neighbourhood_complete,
        "neighbour_pairs": tuple(sorted(neighbours)),
        "raw_peak": peak,
        "robust_choice": robust,
    }
