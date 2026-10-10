"""Pure helpers for exhaustive shortlist-cap × exact-asset searches."""

from __future__ import annotations

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
