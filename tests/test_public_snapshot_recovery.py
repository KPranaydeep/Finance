from contextlib import nullcontext
from unittest.mock import MagicMock

import export_public_record_snapshot as exporter
from public_card_feed import load_public_record_from_database


def test_export_uses_consistent_read_only_transaction(monkeypatch, tmp_path):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.transaction.return_value = nullcontext()
    load = MagicMock(return_value={"current": {"publication_id": "PUB-1"}})
    build = MagicMock(return_value={"verified": True})
    write = MagicMock()
    monkeypatch.setattr(exporter, "get_public_basket_database_url", lambda: "private-test-url")
    monkeypatch.setattr(exporter, "connect_public_basket_db", lambda url: conn)
    monkeypatch.setattr(exporter, "load_public_record_from_database", load)
    monkeypatch.setattr(exporter, "read_review_events", lambda connection, basket: [])
    monkeypatch.setattr(exporter, "build_snapshot", build)
    monkeypatch.setattr(exporter, "write_snapshot", write)
    output = str(tmp_path / "snapshot.json")
    monkeypatch.setattr("sys.argv", ["export", "--output", output])

    assert exporter.main() == 0
    conn.execute.assert_called_once_with(
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    )
    load.assert_called_once_with(conn, "PUBLIC-01", initialize_schema=False)
    write.assert_called_once_with(output, {"verified": True})


def test_read_only_feed_does_not_initialize_trust_schema():
    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

        def fetchall(self):
            return []

    class ReadOnlyConnection:
        def execute(self, sql, params=None):
            assert sql.lstrip().upper().startswith("SELECT"), sql
            if "FROM public_baskets" in sql:
                return Result({"basket_id": "PUBLIC-01"})
            if "to_regclass" in sql:
                return Result({"relation": None})
            return Result()

    record = load_public_record_from_database(
        ReadOnlyConnection(), "PUBLIC-01", initialize_schema=False
    )
    assert record["basket"]["basket_id"] == "PUBLIC-01"
    assert record["current"] is None
    assert record["audit"] == []
