"""Profile-scoped risk controls and last attempt, separate from saved analysis."""
import json
from datetime import datetime, timezone

from allocation_policy_config import validate_allocation_policy


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS allocation_policy_workspace (
        owner TEXT PRIMARY KEY, controls_json TEXT, attempt_json TEXT, updated_at TEXT NOT NULL)""")


def load_workspace(conn, owner):
    _schema(conn)
    row = conn.execute("SELECT controls_json, attempt_json, updated_at FROM allocation_policy_workspace WHERE owner=?", (str(owner),)).fetchone()
    if row is None:
        return {}
    result = {"updated_at": row[2]}
    for key, raw in zip(("controls", "attempt"), row[:2]):
        try:
            value = json.loads(raw or "null")
            if isinstance(value, dict):
                result[key] = value
        except (ValueError, TypeError):
            pass
    return result


def save_controls(conn, owner, controls):
    controls = validate_allocation_policy(controls)
    _schema(conn)
    conn.execute("""INSERT INTO allocation_policy_workspace(owner,controls_json,updated_at) VALUES(?,?,?)
        ON CONFLICT(owner) DO UPDATE SET controls_json=excluded.controls_json,updated_at=excluded.updated_at""",
        (str(owner), json.dumps(controls, allow_nan=False), datetime.now(timezone.utc).isoformat()))
    conn.commit()


def save_attempt(conn, owner, report):
    _schema(conn)
    report = dict(report, recorded_at=datetime.now(timezone.utc).isoformat())
    conn.execute("""INSERT INTO allocation_policy_workspace(owner,attempt_json,updated_at) VALUES(?,?,?)
        ON CONFLICT(owner) DO UPDATE SET attempt_json=excluded.attempt_json,updated_at=excluded.updated_at""",
        (str(owner), json.dumps(report, default=str, allow_nan=False), datetime.now(timezone.utc).isoformat()))
    conn.commit()


def failure_summary(report):
    status = report.get("status", "unavailable")
    category = {
        "unavailable": "Allocation inputs unavailable",
        "infeasible": "No feasible allocation under these limits",
        "time_limit": "Allocation time budget reached",
        "failed_recheck": "Allocation failed its final checks",
    }.get(status, "No allocation passed all checks")
    reason = report.get("reason") or report.get("solver_message") or "Detailed cause was not recorded. No particular constraint has been diagnosed."
    return f"{category}: {reason}"


def capacity_log_summary(report):
    """Only aggregate counts/weights belong in the run log, never private names."""
    capacity = report.get("capacity") or {}
    keys = ("passed", "frozen_count", "conflict_count", "frozen_conflict_count",
            "minimum_weight", "position_cap_sum", "capacity_shortfall", "minimum_excess")
    return {key: capacity[key] for key in keys if key in capacity}


def stage_label(stage):
    return {
        "policy_validation": "validating risk settings",
        "current_allocation_inputs": "checking owned holdings and their history",
        "joint_scenario_inputs": "building complete portfolio scenarios",
        "position_and_frozen_capacity": "checking position limits and frozen holdings",
        "joint_tail_group_cost_solve": "solving joint tail loss, concentration and trading costs",
        "candidate_recheck": "checking the resulting allocation",
        "saved_weight_execution_recheck": "checking the resized draft at saved weights",
    }.get(stage, "not recorded")
