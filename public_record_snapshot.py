"""Verified, static public-record snapshots for quota-free public reads."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

from public_portfolio_publications import verify_trust_audit


SNAPSHOT_SCHEMA = "public-portfolio-record-snapshot"
SNAPSHOT_VERSION = 1
DEFAULT_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/KPranaydeep/Finance/"
    "public-snapshot/public_record_snapshot.json"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def build_snapshot(
    record: dict[str, Any],
    *,
    basket_id: str,
    review_events: list[dict[str, Any]] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a checksum-protected snapshot only from a verified trust chain."""
    normalized_record = _json_value(record)
    audit_ok, audit_message = verify_trust_audit(
        normalized_record.get("audit") or [], basket_id
    )
    if not audit_ok:
        raise ValueError(f"UNVERIFIED_PUBLIC_RECORD: {audit_message}")
    if not normalized_record.get("current"):
        raise ValueError("PUBLICATION_NOT_FOUND")

    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "schema_version": SNAPSHOT_VERSION,
        "basket_id": basket_id,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "record": normalized_record,
        "review_events": _json_value(review_events or []),
    }
    return {**payload, "payload_sha256": _digest(payload)}


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    target = Path(path)
    target.write_text(
        json.dumps(snapshot, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_datetime(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value


def _restore_record_types(record: dict[str, Any]) -> dict[str, Any]:
    restored = dict(record)
    for collection in ("publications", "active_publications"):
        restored[collection] = [dict(row) for row in restored.get(collection) or []]
        for row in restored[collection]:
            for key in ("as_of", "published_at", "created_at"):
                if key in row:
                    row[key] = _parse_datetime(row[key])
    if restored.get("current"):
        restored["current"] = dict(restored["current"])
        for key in ("as_of", "published_at", "created_at"):
            if key in restored["current"]:
                restored["current"][key] = _parse_datetime(restored["current"][key])
    restored["cash_flows"] = [dict(row) for row in restored.get("cash_flows") or []]
    for row in restored["cash_flows"]:
        if "event_at" in row:
            row["event_at"] = _parse_datetime(row["event_at"])
    return restored


def verify_snapshot(snapshot: dict[str, Any], basket_id: str) -> dict[str, Any]:
    """Verify envelope checksum, basket scope, publication and trust audit."""
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("SNAPSHOT_SCHEMA_INVALID")
    if int(snapshot.get("schema_version") or 0) != SNAPSHOT_VERSION:
        raise ValueError("SNAPSHOT_VERSION_UNSUPPORTED")
    if snapshot.get("basket_id") != basket_id:
        raise ValueError("SNAPSHOT_BASKET_MISMATCH")
    supplied_hash = snapshot.get("payload_sha256")
    payload = {key: value for key, value in snapshot.items() if key != "payload_sha256"}
    if not supplied_hash or supplied_hash != _digest(payload):
        raise ValueError("SNAPSHOT_CHECKSUM_INVALID")

    record = payload.get("record") or {}
    audit_ok, audit_message = verify_trust_audit(record.get("audit") or [], basket_id)
    if not audit_ok:
        raise ValueError(f"SNAPSHOT_TRUST_INVALID: {audit_message}")
    if not record.get("current"):
        raise ValueError("SNAPSHOT_PUBLICATION_MISSING")

    restored = _restore_record_types(record)
    restored["review_events"] = payload.get("review_events") or []
    restored["snapshot_generated_at"] = payload.get("generated_at")
    restored["record_source"] = "verified_snapshot"
    return restored


def load_snapshot(
    basket_id: str,
    *,
    url: str | None = None,
    local_path: str | Path | None = None,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    """Load and verify a local snapshot or the public snapshot branch."""
    if local_path:
        raw = Path(local_path).read_text(encoding="utf-8")
    else:
        snapshot_url = url or os.getenv("PUBLIC_RECORD_SNAPSHOT_URL") or DEFAULT_SNAPSHOT_URL
        response = requests.get(snapshot_url, timeout=timeout_seconds)
        response.raise_for_status()
        raw = response.text
    return verify_snapshot(json.loads(raw), basket_id)
