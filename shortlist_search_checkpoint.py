"""Durable checkpoints for the long-running two-dimensional optimizer search."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


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
    return json.dumps(
        dict(config or {}),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def config_hash(config):
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def ensure_checkpoint_schema(conn):
    conn.execute(CHECKPOINT_DDL)


def _analysis_cutoff(results):
    values = [
        str(row.get("Analysis cutoff")).strip()
        for row in (results or [])
        if row.get("Analysis cutoff")
    ]
    return max(values) if values else None


def save_checkpoint(conn, owner, config, results, *, status="running"):
    """Atomically replace one compatible search checkpoint."""
    ensure_checkpoint_schema(conn)
    now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    payload = [dict(row) for row in (results or [])]
    signature = config_hash(config)
    conn.execute(
        """
        INSERT INTO shortlist_search_checkpoints
            (owner, config_hash, config_json, results_json, status,
             analysis_cutoff, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner, config_hash) DO UPDATE SET
            config_json = excluded.config_json,
            results_json = excluded.results_json,
            status = excluded.status,
            analysis_cutoff = excluded.analysis_cutoff,
            updated_at = excluded.updated_at
        """,
        (
            str(owner),
            signature,
            canonical_config_json(config),
            json.dumps(payload, default=str),
            str(status),
            _analysis_cutoff(payload),
            now,
        ),
    )
    conn.commit()
    return signature


def load_latest_checkpoint(conn, owner):
    ensure_checkpoint_schema(conn)
    row = conn.execute(
        """
        SELECT config_hash, config_json, results_json, status,
               analysis_cutoff, updated_at
        FROM shortlist_search_checkpoints
        WHERE owner = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (str(owner),),
    ).fetchone()
    if row is None:
        return None
    return {
        "config_hash": str(row["config_hash"]),
        "config": json.loads(row["config_json"]),
        "results": json.loads(row["results_json"]),
        "status": str(row["status"]),
        "analysis_cutoff": row["analysis_cutoff"],
        "updated_at": str(row["updated_at"]),
    }


def delete_checkpoints(conn, owner):
    ensure_checkpoint_schema(conn)
    conn.execute(
        "DELETE FROM shortlist_search_checkpoints WHERE owner = ?",
        (str(owner),),
    )
    conn.commit()
