"""Rerun-safe timing primitives for an optimization-to-download lifecycle."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def start_run_timer(*, monotonic_now: float | None = None,
                    wall_now: datetime | None = None) -> dict:
    monotonic_now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    wall_now = wall_now or datetime.now(timezone.utc)
    return {
        "status": "running",
        "started_at": wall_now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "monotonic_start": monotonic_now,
        "elapsed_seconds": None,
        "download_file": None,
        "finished_at": None,
    }


def finish_run_timer(timer: dict, download_file: str, *,
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
        "download_file": str(download_file),
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
        "status": "stopped_without_download",
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
