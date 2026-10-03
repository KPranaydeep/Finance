"""Rerun-safe timing primitives for an optimization lifecycle."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def start_run_timer(*, monotonic_now: float | None = None,
                    wall_now: datetime | None = None,
                    estimated_total_seconds: float | None = None) -> dict:
    monotonic_now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    wall_now = wall_now or datetime.now(timezone.utc)
    return {
        "status": "running",
        "started_at": wall_now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "monotonic_start": monotonic_now,
        "elapsed_seconds": None,
        "finish_event": None,
        "finished_at": None,
        "stage": "Starting",
        "progress": 0.0,
        "estimated_total_seconds": (
            max(float(estimated_total_seconds), 1.0)
            if estimated_total_seconds is not None else None
        ),
        "eta_seconds": (
            max(float(estimated_total_seconds), 1.0)
            if estimated_total_seconds is not None else None
        ),
    }


def estimate_run_seconds(
    previous_elapsed_seconds: float | None,
    previous_workload: int | None,
    current_workload: int | None,
    fallback_seconds: float = 360.0,
) -> float:
    """Estimate total runtime, scaling only the broad-universe portion by workload."""
    fallback = max(float(fallback_seconds), 30.0)
    if previous_elapsed_seconds is None or float(previous_elapsed_seconds) <= 0:
        return fallback
    previous = float(previous_elapsed_seconds)
    if not previous_workload or not current_workload:
        return previous
    ratio = max(min(float(current_workload) / float(previous_workload), 3.0), 0.25)
    # Full-history optimization is capped, while broad preselection scales with
    # the universe. Treat 35% as workload-sensitive and 65% as fixed/capped work.
    return max(previous * (0.65 + 0.35 * ratio), 30.0)


def update_run_stage(
    timer: dict,
    stage: str,
    progress: float,
    *,
    estimated_total_seconds: float | None = None,
    monotonic_now: float | None = None,
) -> dict:
    """Return a running timer with an adaptive remaining-time estimate."""
    result = dict(timer or {})
    if result.get("status") != "running":
        return result
    now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    elapsed = max(now - float(result["monotonic_start"]), 0.0)
    bounded_progress = min(max(float(progress), 0.0), 0.99)
    total = estimated_total_seconds
    if total is None:
        total = result.get("estimated_total_seconds")
    total = max(float(total), elapsed + 1.0) if total is not None else None
    baseline_remaining = max(total - elapsed, 0.0) if total is not None else None
    pace_remaining = (
        elapsed * (1.0 - bounded_progress) / bounded_progress
        if bounded_progress > 0 and elapsed > 0 else None
    )
    if baseline_remaining is None:
        remaining = pace_remaining
    elif pace_remaining is None:
        remaining = baseline_remaining
    else:
        remaining = 0.65 * baseline_remaining + 0.35 * pace_remaining
    result.update(
        {
            "stage": str(stage),
            "progress": bounded_progress,
            "estimated_total_seconds": total,
            "eta_seconds": max(float(remaining), 0.0) if remaining is not None else None,
        }
    )
    return result


def finish_run_timer(timer: dict, finish_event: str, *,
                     monotonic_now: float | None = None,
                     wall_now: datetime | None = None) -> dict:
    result = dict(timer or {})
    if result.get("status") != "running":
        return result
    monotonic_now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    wall_now = wall_now or datetime.now(timezone.utc)
    started = float(result["monotonic_start"])
    result.update({
        "status": "finished",
        "elapsed_seconds": max(monotonic_now - started, 0.0),
        "finish_event": str(finish_event),
        "finished_at": wall_now.astimezone(timezone.utc).isoformat(timespec="seconds"),
    })
    return result


def abort_run_timer(timer: dict, reason: str, *,
                    monotonic_now: float | None = None,
                    wall_now: datetime | None = None) -> dict:
    result = dict(timer or {})
    if result.get("status") != "running":
        return result
    monotonic_now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    wall_now = wall_now or datetime.now(timezone.utc)
    started = float(result["monotonic_start"])
    result.update({
        "status": "stopped_without_plan",
        "elapsed_seconds": max(monotonic_now - started, 0.0),
        "stop_reason": str(reason),
        "finished_at": wall_now.astimezone(timezone.utc).isoformat(timespec="seconds"),
    })
    return result


def serializable_timer(timer: dict | None) -> dict | None:
    if not timer:
        return None
    return {key: value for key, value in timer.items() if key != "monotonic_start"}


def format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "Running"
    seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours >= 1:
        return f"{int(hours)} h {int(minutes):02d} m {remaining_seconds:04.1f} s"
    if minutes >= 1:
        return f"{int(minutes)} m {remaining_seconds:04.1f} s"
    return f"{remaining_seconds:.1f} s"
