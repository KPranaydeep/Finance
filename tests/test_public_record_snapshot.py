from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from public_portfolio_trust import fingerprint
from public_record_snapshot import (
    build_snapshot,
    load_snapshot,
    verify_snapshot,
    write_snapshot,
)


def sample_record():
    basket_id = "PUBLIC-01"
    audit_payload = {"basket_id": basket_id, "event": "PUBLISHED"}
    audit = [{
        "basket_id": basket_id,
        "sequence_number": 1,
        "previous_hash": "",
        "payload_json": audit_payload,
        "event_hash": fingerprint({"previous_hash": "", "event": audit_payload}),
    }]
    published_at = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
    current = {
        "basket_id": basket_id,
        "publication_id": "PUB-TEST",
        "portfolio_version": 10,
        "as_of": published_at,
        "published_at": published_at,
    }
    return {
        "basket": {"basket_id": basket_id},
        "current": current,
        "publications": [current],
        "active_publications": [current],
        "constituents": [{"ticker": "TEST.NS", "target_weight": 1.0}],
        "publication_positions": [],
        "forecasts": [],
        "nav": [],
        "cash_flows": [],
        "audit": audit,
    }


def test_snapshot_round_trip_is_verified_and_restores_dates(tmp_path):
    snapshot = build_snapshot(
        sample_record(),
        basket_id="PUBLIC-01",
        review_events=[{"seq": 1, "kind": "HEARTBEAT", "payload": {}}],
    )
    path = tmp_path / "snapshot.json"
    write_snapshot(path, snapshot)

    restored = load_snapshot("PUBLIC-01", local_path=path)

    assert restored["record_source"] == "verified_snapshot"
    assert restored["current"]["as_of"].tzinfo is not None
    assert restored["review_events"][0]["kind"] == "HEARTBEAT"


def test_snapshot_rejects_tampering():
    snapshot = build_snapshot(sample_record(), basket_id="PUBLIC-01")
    snapshot["record"]["current"]["portfolio_version"] = 999

    with pytest.raises(ValueError, match="SNAPSHOT_CHECKSUM_INVALID"):
        verify_snapshot(snapshot, "PUBLIC-01")


def test_snapshot_json_contains_no_python_specific_values():
    snapshot = build_snapshot(sample_record(), basket_id="PUBLIC-01")
    encoded = json.dumps(snapshot, allow_nan=False)
    assert "PUB-TEST" in encoded
