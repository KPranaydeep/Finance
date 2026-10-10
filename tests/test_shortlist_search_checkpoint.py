import sqlite3

from shortlist_search_checkpoint import (
    delete_checkpoints,
    load_best_saved_result,
    load_latest_checkpoint,
    save_checkpoint,
)


def _connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_checkpoint_round_trip_and_latest_replacement():
    conn = _connection()
    config = {
        "starting_cap": 7700,
        "maximum_cap": 9400,
        "exact_optimizer_asset_caps": [600, 650, 700],
        "snapshot_date_ist": "2026-10-10",
    }
    first = [{
        "Shortlist cap": 7700,
        "Maximum assets": 600,
        "Annual Return": 0.81,
        "Status": "Feasible",
        "Analysis cutoff": "2026-10-09",
    }]
    save_checkpoint(conn, "owner", config, first, status="running")
    restored = load_latest_checkpoint(conn, "owner")
    assert restored["config"] == config
    assert restored["results"] == first
    assert restored["analysis_cutoff"] == "2026-10-09"
    assert restored["status"] == "running"

    second = first + [{
        "Shortlist cap": 7750,
        "Maximum assets": 650,
        "Annual Return": 0.83,
        "Status": "Feasible",
        "Analysis cutoff": "2026-10-09",
    }]
    save_checkpoint(conn, "owner", config, second, status="complete")
    restored = load_latest_checkpoint(conn, "owner")
    assert len(restored["results"]) == 2
    assert restored["status"] == "complete"


def test_delete_removes_only_selected_owner():
    conn = _connection()
    config = {"snapshot_date_ist": "2026-10-10"}
    save_checkpoint(conn, "one", config, [], status="running")
    save_checkpoint(conn, "two", config, [], status="running")
    delete_checkpoints(conn, "one")
    assert load_latest_checkpoint(conn, "one") is None
    assert load_latest_checkpoint(conn, "two") is not None


def test_best_saved_result_spans_checkpoints_and_ignores_infeasible_rows():
    conn = _connection()
    save_checkpoint(
        conn,
        "owner",
        {"snapshot_date_ist": "2026-10-09"},
        [
            {
                "Shortlist cap": 8650,
                "Maximum assets": 1250,
                "Annual Return": 0.92,
                "Status": "Feasible",
            },
            {
                "Shortlist cap": 9000,
                "Maximum assets": 2000,
                "Annual Return": 0.99,
                "Status": "Failed",
            },
        ],
        status="complete",
    )
    save_checkpoint(
        conn,
        "owner",
        {"snapshot_date_ist": "2026-10-10"},
        [{
            "Shortlist cap": 8800,
            "Maximum assets": 3000,
            "Annual Return": 0.88,
            "Status": "Feasible",
        }],
        status="running",
    )

    best = load_best_saved_result(conn, "owner")
    assert (best["Shortlist cap"], best["Maximum assets"]) == (8650, 1250)
    assert load_best_saved_result(conn, "another-owner") is None
