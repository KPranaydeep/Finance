from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from unittest.mock import Mock

from public_portfolio_trust import fingerprint
from public_record_snapshot import (
    build_snapshot,
    load_snapshot,
    verify_snapshot,
    write_snapshot,
    DEFAULT_SNAPSHOT_URL,
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


def test_default_branch_uses_shared_bounded_cdn_cache_window(monkeypatch):
    response = Mock(text=json.dumps(build_snapshot(sample_record(), basket_id="PUBLIC-01")))
    get = Mock(return_value=response)
    monkeypatch.setattr("public_record_snapshot.requests.get", get)
    monkeypatch.setattr("public_record_snapshot.time.time", lambda: 900.0)
    monkeypatch.delenv("PUBLIC_RECORD_SNAPSHOT_URL", raising=False)
    result = load_snapshot("PUBLIC-01")
    get.assert_called_once_with(DEFAULT_SNAPSHOT_URL, timeout=8.0,
                                params={"snapshot_window": 3})
    assert result["record_source"] == "verified_snapshot"


def test_custom_snapshot_url_is_preserved_without_extra_query_parameters(monkeypatch):
    response = Mock(text=json.dumps(build_snapshot(sample_record(), basket_id="PUBLIC-01")))
    get = Mock(return_value=response)
    monkeypatch.setattr("public_record_snapshot.requests.get", get)
    url = "https://example.invalid/snapshot?signature=test-only"
    load_snapshot("PUBLIC-01", url=url)
    get.assert_called_once_with(url, timeout=8.0)
