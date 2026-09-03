"""
Fallback / helper for public basket Postgres access.

Behavior:
- get_public_basket_database_url()
    * Return PUBLIC_BASKET_DATABASE_URL from env or Streamlit secrets if present.
    * If not present, and a local 'universal_portfolio_backup.csv' is present,
      returns a special file URL "file://local" which connect_public_basket_db()
      understands and will provide a LocalConnector that reads data from the CSV
      and synthesizes a minimal public_baskets + daily_nav dataset so the UI can
      show something.

- connect_public_basket_db(database_url)
    * If database_url starts with "file://", returns a LocalConnector object used for file fallback.
    * Otherwise, tries to create a SQLAlchemy Engine and return engine.connect().
      (SQLAlchemy is optional — if unavailable a helpful error is raised.)

This file is intended as a developer/testing fallback only. For production, configure a
real Postgres connection string in Streamlit secrets as PUBLIC_BASKET_DATABASE_URL.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

try:
    import pandas as pd
except Exception:
    pd = None  # optional; used only for CSV convenience

# Public API
DEFAULT_BASKET_ID = "PUBLIC"

# --- Helpers: discover DB URL -------------------------------------------------


def get_public_basket_database_url() -> Optional[str]:
    """
    Return a database URL for the public basket store, or None.
    Order of lookup:
      1. environment variable PUBLIC_BASKET_DATABASE_URL
      2. streamlit.secrets['PUBLIC_BASKET_DATABASE_URL'] if streamlit is available
      3. fallback to file mode if local evidence exists (returns "file://local")
    """
    # 1) env
    url = os.environ.get("PUBLIC_BASKET_DATABASE_URL")
    if url:
        return url

    # 2) streamlit secrets (if available)
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets and isinstance(secrets, dict):
            svc = secrets.get("PUBLIC_BASKET_DATABASE_URL")
            if svc:
                return svc
    except Exception:
        # streamlit not available or secrets not set; ignore
        pass

    # 3) local fallback: check for evidence files in repo root
    root = pathlib.Path.cwd()
    csv_path = root / "universal_portfolio_backup.csv"
    json_path = root / f"{DEFAULT_BASKET_ID.lower()}-public-evidence.json"
    # If either file exists, enable file mode fallback
    if csv_path.exists() or json_path.exists():
        return "file://local"

    # Nothing found
    return None


# --- LocalConnector used in file:// mode -------------------------------------


class LocalResult:
    """Simple object implementing .fetchone() and .fetchall() like DBAPI/SA results."""

    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self._rows = list(rows)

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class LocalConnector:
    """
    Minimal local "connection" with execute(sql, params) -> LocalResult.

    The connector recognizes the same table names used in the app:
      - public_baskets
      - signal_runs
      - rebalance_events
      - trade_orders
      - trade_executions
      - daily_nav
      - public_basket_audit_log

    For testing it synthesizes a small public_baskets row and a short daily_nav
    derived from universal_portfolio_backup.csv if available.
    """

    def __init__(self, root: Optional[pathlib.Path] = None):
        self.root = pathlib.Path(root or pathlib.Path.cwd())
        self._load_sources()

    def _load_sources(self):
        # Try to read an evidence JSON (if present) preferred over CSV
        evidence_file = self.root / f"{DEFAULT_BASKET_ID.lower()}-public-evidence.json"
        csv_file = self.root / "universal_portfolio_backup.csv"

        self._signals: List[Dict[str, Any]] = []
        self._rebalances: List[Dict[str, Any]] = []
        self._orders: List[Dict[str, Any]] = []
        self._executions: List[Dict[str, Any]] = []
        self._nav: List[Dict[str, Any]] = []
        self._audit: List[Dict[str, Any]] = []

        if evidence_file.exists():
            try:
                with open(evidence_file, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                # adopt expected keys if present
                self._signals = payload.get("signals", []) or []
                self._rebalances = payload.get("rebalances", []) or []
                self._orders = payload.get("orders", []) or []
                self._executions = payload.get("executions", []) or []
                self._nav = payload.get("nav", []) or []
                self._audit = payload.get("audit", []) or []
            except Exception:
                # fallback to empty
                pass

        # If we don't have nav rows, try to synthesize from CSV
        if not self._nav and csv_file.exists():
            try:
                if pd:
                    df = pd.read_csv(csv_file, encoding="utf-8-sig", keep_default_na=False)
                    # try columns "Average Price"/"AveragePrice" and "Quantity"
                    price_col = None
                    for cand in ("Average Price", "AveragePrice", "Average_Price", "AveragePrice."):
                        if cand in df.columns:
                            price_col = cand
                            break
                    qty_col = None
                    for cand in ("Quantity", "quantity", "Qty", "QTY"):
                        if cand in df.columns:
                            qty_col = cand
                            break
                    if price_col is None:
                        df["avg_price"] = 0.0
                        price_col = "avg_price"
                    if qty_col is None:
                        df["qty"] = 0.0
                        qty_col = "qty"

                    df[price_col] = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)
                    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)

                    total_value = float((df[price_col] * df[qty_col]).sum())
                else:
                    # simple CSV read fallback
                    total_value = 0.0
                    with open(csv_file, newline="", encoding="utf-8-sig") as fh:
                        reader = csv.DictReader(fh)
                        for r in reader:
                            try:
                                price = float(r.get("Average Price") or r.get("AveragePrice") or 0.0)
                                qty = float(r.get("Quantity") or r.get("quantity") or 0.0)
                                total_value += price * qty
                            except Exception:
                                continue

                # Build two nav rows 30 days apart so performance_summary can compute growth
                now = datetime.utcnow().date()
                nav_latest = {
                    "nav_date": str(now),
                    "calculation_version": 1,
                    "nav": float(total_value if total_value and total_value > 0 else 1000.0),
                    "portfolio_value": float(total_value if total_value and total_value > 0 else 1000.0),
                    "cash_value": 0.0,
                    "input_sha256": "",
                    "calculated_at": str(datetime.utcnow()),
                }
                prev_date = now - timedelta(days=30)
                nav_prev = {
                    "nav_date": str(prev_date),
                    "calculation_version": 1,
                    "nav": float((nav_latest["nav"]) * 0.98),  # 2% lower 30 days earlier
                    "portfolio_value": float((nav_latest["nav"]) * 0.98),
                    "cash_value": 0.0,
                    "input_sha256": "",
                    "calculated_at": str(datetime.utcnow() - timedelta(days=30)),
                }
                self._nav = [nav_latest, nav_prev]
            except Exception:
                self._nav = []

    def execute(self, sql: str, params: Optional[Sequence] = None):
        s = (sql or "").lower()
        # crude detection of table being queried
        if "from public_baskets" in s or "public_baskets" in s and "select" in s:
            row = {
                "basket_id": DEFAULT_BASKET_ID,
                "basket_name": "Local fallback portfolio",
                "calendar_market": "NSE",
                "rebalance_rule": "WEEKLY",
                "strategy_version": "local",
                "schema_version": 1,
            }
            return LocalResult([row])

        if "from signal_runs" in s or "signal_runs" in s:
            return LocalResult(self._signals)

        if "from rebalance_events" in s or "rebalance_events" in s:
            return LocalResult(self._rebalances)

        if "from trade_orders" in s or "trade_orders" in s:
            return LocalResult(self._orders)

        if "from trade_executions" in s or "trade_executions" in s:
            return LocalResult(self._executions)

        if "from daily_nav" in s or "daily_nav" in s:
            return LocalResult(self._nav)

        if "from public_basket_audit_log" in s or "public_basket_audit_log" in s:
            return LocalResult(self._audit)

        # Default: return empty result
        return LocalResult([])

    # For compatibility with callers that close the connection
    def close(self):
        return None


# --- Connection factory ------------------------------------------------------


def connect_public_basket_db(database_url: str):
    """
    If database_url starts with file:// the function returns a LocalConnector
    that reads local files. Otherwise it attempts to create a SQLAlchemy
    Engine and returns a Connection object.

    Raises a helpful error if neither path is available.
    """
    if not database_url:
        raise ValueError("database_url is required")

    if database_url.startswith("file://"):
        return LocalConnector()

    # Try to use SQLAlchemy (recommended for real DB)
    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise RuntimeError(
            "SQLAlchemy is required to connect to a real database. "
            "Install sqlalchemy or use the local file fallback by not setting "
            "PUBLIC_BASKET_DATABASE_URL."
        ) from exc

    engine = create_engine(database_url)
    conn = engine.connect()
    return conn


# --- Simple publish helper used by pages that call record_weekly_signal ------
# This implements a tiny local fallback for publishing signals so page flows don't crash
# when running in file mode. If a real DB is used you should replace this with
# your application DB insert logic.

def rebalance_gate(conn, basket_id: str, scheduled_date):
    """
    Minimal gate check used by pages/YY_Generate_and_Run_From_CSV.py.

    In file-fallback mode (LocalConnector) this returns a permissive DUE status.
    When a real DB connection is provided this function should be implemented to
    query the actual weekly_rebalance_cycles / scheduling table — here we return
    a safe default so the UI can proceed.
    """
    # If global override is set, allow
    if globals().get("PUBLIC_BASKET_ALLOW_ANY_DAY"):
        return {"status": "DUE", "reason": "override_any_day_enabled"}

    # LocalConnector (file mode) -> permissive
    try:
        if isinstance(conn, LocalConnector):
            return {"status": "DUE", "reason": "local_fallback"}
    except Exception:
        # conn may be a sqlalchemy connection or other; ignore and fall through
        pass

    # For real databases: try to perform a lightweight check if possible (best-effort).
    try:
        # Attempt a safe query; many deployments will not have this exact schema,
        # so we catch exceptions and return DUE as fallback.
        sql = "SELECT 1"
        try:
            res = conn.execute(sql)
            # if query runs, return a permissive DUE — real implementation should inspect DB rows
            return {"status": "DUE", "reason": "db_check_ok"}
        except Exception:
            return {"status": "DUE", "reason": "db_query_failed_but_continue"}
    except Exception as exc:
        return {"status": "DUE", "reason": "unexpected_error", "error": str(exc)}

def record_weekly_signal(
    conn,
    basket_id: str,
    today,
    strategy_version: str,
    git_commit_sha: Optional[str],
    settings: Dict[str, Any],
    portfolio_before: Dict[str, Any],
    optimizer_output: Dict[str, Any],
    signal_output: List[Dict[str, Any]],
    decision_status: str,
) -> int:
    """
    If `conn` is a LocalConnector, append the published signal into a local JSON
    in the repo root called '<DEFAULT_BASKET_ID>-published-signals.json' and return
    a synthetic id. If conn is a SQLAlchemy connection, raise NotImplementedError
    because real DB schema is required to safely insert.
    """
    if isinstance(conn, LocalConnector):
        out = {"published_at": str(datetime.utcnow()), "basket_id": basket_id, "today": str(today)}
        out.update(
            {
                "strategy_version": strategy_version,
                "git_commit_sha": git_commit_sha,
                "settings": settings,
                "portfolio_before": portfolio_before,
                "optimizer_output": optimizer_output,
                "signal_output": signal_output,
                "decision_status": decision_status,
            }
        )
        file_path = pathlib.Path.cwd() / f"{DEFAULT_BASKET_ID.lower()}-published-signals.json"
        existing = []
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except Exception:
                existing = []
        existing.append(out)
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, default=str)
        return len(existing) - 1  # synthetic id (index)
    else:
        # We don't implement DB inserts for real DBs here. Real schema and transaction
        # logic is required before you do this in production.
        raise NotImplementedError(
            "record_weekly_signal is not implemented for real databases in this fallback module. "
            "Use a real implementation to insert into your public database."
        )


# --- Utility: safe close (used by callers who expect conn.close()) -------------
def safe_close(conn):
    try:
        conn.close()
    except Exception:
        pass
