import ast
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shortlist_search_checkpoint import (
    load_latest_checkpoint,
    save_checkpoint,
)


SOURCE_PATH = Path("portfolio_rebalancer_database.py")


def test_search_checkpoint_survives_database_reopen(tmp_path):
    db_path = tmp_path / "search.sqlite"
    rows = [
        {
            "Search phase": "Alternating X gradient",
            "Shortlist cap": 8600,
            "Maximum assets": 650,
            "Annual Return": 0.9226,
            "Status": "Feasible",
            "Search saturated": True,
        }
    ]
    with sqlite3.connect(db_path) as conn:
        save_checkpoint(conn, "test-user", {"snapshot": "2026-10-27"}, rows, status="complete")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        restored = load_latest_checkpoint(conn, "test-user")

    assert restored["status"] == "complete"
    assert restored["results"] == rows
    assert restored["analysis_cutoff"] is None


def _load_saved_analysis_functions(db_path):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name)
                and target.id == "LATEST_ANALYSIS_DDL"
                for target in node.targets
            ):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "make_json_safe",
            "save_latest_analysis",
            "load_latest_analysis",
        }:
            selected.append(node)

    @contextmanager
    def get_db_connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    namespace = {
        "json": json,
        "sqlite3": sqlite3,
        "np": np,
        "pd": pd,
        "datetime": datetime,
        "get_db_connection": get_db_connection,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<saved-analysis>", "exec"), namespace)
    return namespace["save_latest_analysis"], namespace["load_latest_analysis"]


def test_saved_analysis_survives_restart_with_json_safe_values(tmp_path):
    save, load = _load_saved_analysis_functions(tmp_path / "analysis.sqlite")
    payload = {
        "saved_at": "2026-10-27T12:00:00+00:00",
        "optimal_stats": {"Annual Return": np.float64(0.68)},
        "security_scenario_ranking": [{"ticker": "STLN", "score": np.float64(2.5)}],
    }

    save(payload, "test-user")
    restored = load("test-user")

    assert restored["saved_at"] == payload["saved_at"]
    assert restored["optimal_stats"]["Annual Return"] == 0.68
    assert restored["security_scenario_ranking"][0]["ticker"] == "STLN"
    assert restored["security_scenario_ranking"][0]["score"] == 2.5

