"""Durable checkpoints for the long-running two-dimensional optimizer search."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from two_dimensional_search import best_feasible_result


CHECKPOINT_DDL = """
CREATE TABLE IF NOT EXISTS shortlist_search_checkpoints (
    owner TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    status TEXT NOT NULL,
    analysis_cutoff TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, config_hash)
)
"""


def canonical_config_json(config):
    return json.dumps(dict(config or {}), sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config):
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def research_scope_hash(config):
    """Hash search settings while ignoring snapshot date and warm-start coords."""
    stable = dict(config or {})
    for key in (
        "snapshot_date_ist",
        "starting_cap",
        "starting_asset_limit",
        "skip_recent_best_search",
    ):
        stable.pop(key, None)
    return hashlib.sha256(canonical_config_json(stable).encode("utf-8")).hexdigest()


def ensure_checkpoint_schema(conn):
    conn.execute(CHECKPOINT_DDL)


def _analysis_cutoff(results):
    values = [str(row.get("Analysis cutoff")).strip() for row in (results or []) if row.get("Analysis cutoff")]
    return max(values) if values else None


def save_checkpoint(conn, owner, config, results, *, status="running"):
    ensure_checkpoint_schema(conn)
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    payload = [dict(row) for row in (results or [])]
    signature = config_hash(config)
    conn.execute(
        """
        INSERT INTO shortlist_search_checkpoints
            (owner, config_hash, config_json, results_json, status, analysis_cutoff, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner, config_hash) DO UPDATE SET
            config_json = excluded.config_json,
            results_json = excluded.results_json,
            status = excluded.status,
            analysis_cutoff = excluded.analysis_cutoff,
            updated_at = excluded.updated_at
        """,
        (str(owner), signature, canonical_config_json(config), json.dumps(payload, default=str),
         str(status), _analysis_cutoff(payload), now),
    )
    conn.commit()
    return signature


def _usable_result_rows(results):
    """Keep legacy rows with no Y, but reject coordinates that cannot be rendered."""
    usable = []
    for result in results:
        if not isinstance(result, dict) or result.get("Shortlist cap") is None:
            continue
        try:
            for key in ("Shortlist cap", "Maximum assets"):
                value = result.get(key)
                if value is not None and (int(value) <= 0 or float(value) != int(value)):
                    raise ValueError("Invalid search coordinate")
        except (ValueError, TypeError, OverflowError):
            continue
        usable.append(result)
    return usable


def load_latest_checkpoint(conn, owner):
    ensure_checkpoint_schema(conn)
    row = conn.execute(
        """SELECT config_hash, config_json, results_json, status, analysis_cutoff, updated_at
           FROM shortlist_search_checkpoints WHERE owner = ?
           ORDER BY updated_at DESC LIMIT 1""",
        (str(owner),),
    ).fetchone()
    if row is None:
        return None
    try:
        config = json.loads(row["config_json"])
        results = json.loads(row["results_json"])
    except (ValueError, TypeError):
        return None  # Retain the stored evidence; never rewrite a corrupt payload.
    if not isinstance(config, dict) or not isinstance(results, list):
        return None
    return {
        "config_hash": str(row["config_hash"]),
        "config": config,
        "results": _usable_result_rows(results),
        "status": str(row["status"]),
        "analysis_cutoff": row["analysis_cutoff"],
        "updated_at": str(row["updated_at"]),
    }


def load_latest_checkpoint_for_research_scope(conn, owner, config):
    """Load the newest completed checkpoint with matching research settings."""
    ensure_checkpoint_schema(conn)
    rows = conn.execute(
        """SELECT config_hash, config_json, results_json, status, analysis_cutoff, updated_at
           FROM shortlist_search_checkpoints WHERE owner = ?
           ORDER BY updated_at DESC""",
        (str(owner),),
    ).fetchall()
    target_scope = research_scope_hash(config)
    for row in rows:
        try:
            saved_config = json.loads(row["config_json"])
            results = json.loads(row["results_json"])
        except (ValueError, TypeError):
            return None
        if not isinstance(saved_config, dict) or not isinstance(results, list):
            return None
        if len(_usable_result_rows(results)) != len(results):
            return None
        if research_scope_hash(saved_config) != target_scope:
            continue
        if str(row["status"]) == "running":
            # Prefer resuming this incomplete search over replacing it with a
            # one-pair refresh after an interruption.
            return None
        if str(row["status"]) != "complete" or not results:
            continue
        if not any(
            isinstance(result.get("Search skipper context"), dict)
            for result in results
        ):
            # A newer completed search without a skipper checkpoint must
            # invalidate an older leader; otherwise we could skip after a
            # recent full scan that failed to saturate.
            return None
        return {
            "config_hash": str(row["config_hash"]),
            "config": saved_config,
            "results": results,
            "status": str(row["status"]),
            "analysis_cutoff": row["analysis_cutoff"],
            "updated_at": str(row["updated_at"]),
        }
    return None


def load_best_saved_result(conn, owner):
    """Return the highest-return feasible pair across all saved searches."""
    ensure_checkpoint_schema(conn)
    stored_rows = conn.execute(
        """
        SELECT results_json
        FROM shortlist_search_checkpoints
        WHERE owner = ?
        """,
        (str(owner),),
    ).fetchall()
    candidates = []
    for stored in stored_rows:
        try:
            results = json.loads(stored["results_json"])
        except (ValueError, TypeError):
            continue
        if isinstance(results, list):
            candidate = best_feasible_result(results)
            if candidate is not None:
                candidates.append(candidate)
    best = best_feasible_result(candidates)
    return dict(best) if best is not None else None


def delete_checkpoints(conn, owner):
    ensure_checkpoint_schema(conn)
    conn.execute("DELETE FROM shortlist_search_checkpoints WHERE owner = ?", (str(owner),))
    conn.commit()
