"""Staged, percentile-capped cleaner for the shared security universe."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd


JOB_DDL = """
CREATE TABLE IF NOT EXISTS universal_cleaner_jobs (
    job_id TEXT PRIMARY KEY,
    clusters_json TEXT NOT NULL,
    criteria_json TEXT NOT NULL,
    removal_percentile REAL NOT NULL,
    include_insider INTEGER NOT NULL,
    status TEXT NOT NULL,
    total_symbols INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    applied_at TEXT,
    note TEXT
)
"""

ITEM_DDL = """
CREATE TABLE IF NOT EXISTS universal_cleaner_items (
    job_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    yahoo_ticker TEXT NOT NULL,
    exchange TEXT,
    currency TEXT,
    protected INTEGER NOT NULL DEFAULT 0,
    history_status TEXT NOT NULL DEFAULT 'pending',
    insider_status TEXT NOT NULL DEFAULT 'not_selected',
    latest_price REAL,
    dma25 REAL,
    dma50 REAL,
    dma200 REAL,
    bearish_stack INTEGER NOT NULL DEFAULT 0,
    technical_score REAL,
    insider_net_sale_ratio REAL,
    insider_sale_value REAL,
    final_score REAL,
    proposed_remove INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, symbol),
    FOREIGN KEY (job_id) REFERENCES universal_cleaner_jobs(job_id)
)
"""


def ensure_cleaner_schema(conn: sqlite3.Connection) -> None:
    conn.execute(JOB_DDL)
    conn.execute(ITEM_DDL)
    job_columns = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(universal_cleaner_jobs)").fetchall()
    }
    if "criteria_json" not in job_columns:
        conn.execute("ALTER TABLE universal_cleaner_jobs ADD COLUMN criteria_json TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cleaner_history "
        "ON universal_cleaner_items(job_id, history_status, symbol)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cleaner_insider "
        "ON universal_cleaner_items(job_id, insider_status, symbol)"
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cluster_key(exchange, currency) -> str:
    return f"{str(exchange or 'Unknown').strip()} · {str(currency or 'Unknown').strip()}"


def available_clusters(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return []
    keys = {
        cluster_key(row.Exchange, row.Currency)
        for row in frame[["Exchange", "Currency"]].itertuples(index=False)
    }
    return sorted(keys)


def score_price_history(series: pd.Series, as_of=None, stale_days=15) -> dict:
    prices = pd.to_numeric(series, errors="coerce").dropna()
    if prices.empty:
        return {"history_status": "no_price"}
    latest = float(prices.iloc[-1])
    if not math.isfinite(latest) or latest <= 0:
        return {"history_status": "no_price"}
    if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index):
        last_observation = pd.Timestamp(prices.index[-1])
        reference = pd.Timestamp(as_of or datetime.now(timezone.utc))
        if last_observation.tzinfo is None:
            last_observation = last_observation.tz_localize("UTC")
        else:
            last_observation = last_observation.tz_convert("UTC")
        if reference.tzinfo is None:
            reference = reference.tz_localize("UTC")
        else:
            reference = reference.tz_convert("UTC")
        if last_observation < reference - pd.Timedelta(days=int(stale_days)):
            return {
                "history_status": "no_price",
                "latest_price": latest,
                "reason": (
                    "Latest usable price is stale ("
                    f"{last_observation.date().isoformat()}); review listing status manually."
                ),
            }
    if len(prices) < 200:
        return {
            "history_status": "insufficient_history",
            "latest_price": latest,
            "reason": f"Only {len(prices)} valid sessions; 200 required for the full trend test.",
        }
    dma25 = float(prices.tail(25).mean())
    dma50 = float(prices.tail(50).mean())
    dma200 = float(prices.tail(200).mean())
    below25 = latest < dma25
    below50 = latest < dma50
    below200 = latest < dma200
    bearish_stack = latest < dma25 < dma50 < dma200
    gap200 = max((dma200 - latest) / dma200, 0.0) if dma200 > 0 else 0.0
    technical_score = (
        0.15 * float(below25)
        + 0.25 * float(below50)
        + 0.40 * float(below200)
        + 0.20 * min(gap200 / 0.30, 1.0)
    )
    return {
        "history_status": "ready",
        "latest_price": latest,
        "dma25": dma25,
        "dma50": dma50,
        "dma200": dma200,
        "bearish_stack": int(bearish_stack),
        "technical_score": float(technical_score),
    }


def insider_sale_signal(transactions: pd.DataFrame, as_of=None, lookback_days=180) -> dict:
    """Measure management open-market sales; missing/ambiguous data stays neutral."""
    if transactions is None or transactions.empty:
        return {"status": "unavailable"}
    frame = transactions.copy()
    required = {"Text", "Position", "Start Date"}
    if not required.issubset(frame.columns):
        return {"status": "unavailable"}
    dates = pd.to_datetime(frame["Start Date"], errors="coerce", utc=True)
    reference = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    cutoff = reference - pd.Timedelta(days=int(lookback_days))
    position = frame["Position"].fillna("").astype(str).str.lower()
    is_management = position.str.contains(
        r"officer|director|chief|ceo|cfo|president|counsel|treasurer",
        regex=True,
    )
    recent = frame.loc[dates.ge(cutoff) & dates.le(reference) & is_management].copy()
    if recent.empty:
        return {"status": "available", "net_sale_ratio": 0.0, "sale_value": 0.0}
    text = recent["Text"].fillna("").astype(str).str.lower()
    if "Value" in recent.columns:
        values = pd.to_numeric(recent["Value"], errors="coerce").fillna(0).abs()
    else:
        values = pd.Series(0.0, index=recent.index)
    sale_value = float(values[text.str.contains(r"\bsale\b", regex=True)].sum())
    purchase_value = float(values[text.str.contains(r"\bpurchase\b", regex=True)].sum())
    gross = sale_value + purchase_value
    ratio = max(sale_value - purchase_value, 0.0) / gross if gross > 0 else 0.0
    return {
        "status": "available",
        "net_sale_ratio": float(ratio),
        "sale_value": sale_value,
    }


def prepare_cleaner_job(
    conn: sqlite3.Connection,
    universal_owner: str,
    selected_clusters: list[str],
    removal_percentile: float,
    include_insider: bool,
    criteria: list[str] | None = None,
) -> dict:
    ensure_cleaner_schema(conn)
    percentile = float(removal_percentile)
    if not 1 <= percentile <= 20:
        raise ValueError("Removal percentile must be between 1% and 20%.")
    valid_criteria = {"unavailable", "bearish", "management"}
    selected_criteria = valid_criteria if criteria is None else set(criteria)
    if not selected_criteria or not selected_criteria.issubset(valid_criteria):
        raise ValueError("Choose at least one supported cleaner criterion.")
    include_insider = bool(include_insider and "management" in selected_criteria)
    rows = conn.execute(
        """
        SELECT symbol, yahoo_ticker, exchange, currency
        FROM master_holdings WHERE owner = ? ORDER BY symbol
        """,
        (universal_owner,),
    ).fetchall()
    selected = set(selected_clusters or [])
    selected_rows = [
        row for row in rows
        if not selected or cluster_key(row["exchange"], row["currency"]) in selected
    ]
    if not selected_rows:
        raise ValueError("No Universal Portfolio symbols match the selected clusters.")
    held_tickers = {
        str(row["yahoo_ticker"] or "").upper()
        for row in conn.execute(
            """
            SELECT yahoo_ticker FROM master_holdings
            WHERE owner <> ? AND quantity > 0 AND yahoo_ticker IS NOT NULL
            """,
            (universal_owner,),
        ).fetchall()
    }
    job_id = "CLEAN-" + uuid.uuid4().hex.upper()
    now = _now()
    conn.execute("BEGIN")
    try:
        conn.execute(
            """
            INSERT INTO universal_cleaner_jobs
                (job_id, clusters_json, criteria_json, removal_percentile, include_insider,
                 status, total_symbols, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'prepared', ?, ?, ?)
            """,
            (
                job_id,
                json.dumps(sorted(selected)),
                json.dumps(sorted(selected_criteria)),
                percentile,
                int(bool(include_insider)),
                len(selected_rows),
                now,
                now,
            ),
        )
        for row in selected_rows:
            ticker = str(row["yahoo_ticker"] or row["symbol"]).upper()
            conn.execute(
                """
                INSERT INTO universal_cleaner_items
                    (job_id, symbol, yahoo_ticker, exchange, currency, protected, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    row["symbol"],
                    ticker,
                    row["exchange"],
                    row["currency"],
                    int(ticker in held_tickers),
                    now,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return get_cleaner_job(conn, job_id)


def get_cleaner_job(conn: sqlite3.Connection, job_id: str) -> dict:
    ensure_cleaner_schema(conn)
    row = conn.execute(
        "SELECT * FROM universal_cleaner_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise ValueError("Cleaner job was not found.")
    result = dict(row)
    result["clusters"] = json.loads(result.pop("clusters_json"))
    raw_criteria = result.pop("criteria_json", None)
    result["criteria"] = (
        json.loads(raw_criteria)
        if raw_criteria
        else ["unavailable", "bearish", "management"]
    )
    counts = {
        "history_pending": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND history_status='pending'",
            (job_id,),
        ).fetchone()[0],
        "history_ready": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND history_status='ready'",
            (job_id,),
        ).fetchone()[0],
        "no_price": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND history_status='no_price'",
            (job_id,),
        ).fetchone()[0],
        "insufficient_history": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND history_status='insufficient_history'",
            (job_id,),
        ).fetchone()[0],
        "insider_pending": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND insider_status='pending'",
            (job_id,),
        ).fetchone()[0],
        "protected": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND protected=1",
            (job_id,),
        ).fetchone()[0],
        "proposed": conn.execute(
            "SELECT COUNT(*) FROM universal_cleaner_items WHERE job_id=? AND proposed_remove=1",
            (job_id,),
        ).fetchone()[0],
    }
    result["counts"] = {key: int(value) for key, value in counts.items()}
    return result


def latest_cleaner_job(conn: sqlite3.Connection) -> dict | None:
    ensure_cleaner_schema(conn)
    row = conn.execute(
        "SELECT job_id FROM universal_cleaner_jobs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return get_cleaner_job(conn, row["job_id"]) if row is not None else None


def pending_history_items(conn, job_id, limit=60) -> list[dict]:
    rows = conn.execute(
        """
        SELECT symbol, yahoo_ticker FROM universal_cleaner_items
        WHERE job_id=? AND history_status='pending' ORDER BY symbol LIMIT ?
        """,
        (job_id, int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]


def record_history_batch(conn, job_id, results: dict[str, dict]) -> dict:
    now = _now()
    conn.execute("BEGIN")
    try:
        for symbol, metrics in results.items():
            conn.execute(
                """
                UPDATE universal_cleaner_items
                SET history_status=?, latest_price=?, dma25=?, dma50=?, dma200=?,
                    bearish_stack=?, technical_score=?, reason=?, updated_at=?
                WHERE job_id=? AND symbol=? AND history_status='pending'
                """,
                (
                    metrics.get("history_status", "no_price"),
                    metrics.get("latest_price"),
                    metrics.get("dma25"),
                    metrics.get("dma50"),
                    metrics.get("dma200"),
                    int(metrics.get("bearish_stack", 0)),
                    metrics.get("technical_score"),
                    metrics.get("reason"),
                    now,
                    job_id,
                    symbol,
                ),
            )
        conn.execute(
            "UPDATE universal_cleaner_jobs SET status='scanning_history', updated_at=? WHERE job_id=?",
            (now, job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return get_cleaner_job(conn, job_id)


def prepare_insider_shortlist(conn, job_id, max_symbols=200) -> int:
    job = get_cleaner_job(conn, job_id)
    if not job["include_insider"]:
        return 0
    quota = max(1, int(math.ceil(job["total_symbols"] * job["removal_percentile"] / 100)))
    shortlist_size = min(max_symbols, max(quota * 2, 25))
    rows = conn.execute(
        """
        SELECT symbol FROM universal_cleaner_items
        WHERE job_id=? AND protected=0 AND history_status='ready'
          AND UPPER(COALESCE(currency,''))='USD'
          AND yahoo_ticker NOT LIKE '%.%'
        ORDER BY COALESCE(technical_score,0) DESC, symbol
        LIMIT ?
        """,
        (job_id, shortlist_size),
    ).fetchall()
    now = _now()
    conn.execute("BEGIN")
    try:
        for row in rows:
            conn.execute(
                "UPDATE universal_cleaner_items SET insider_status='pending', updated_at=? "
                "WHERE job_id=? AND symbol=? AND insider_status='not_selected'",
                (now, job_id, row["symbol"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


def pending_insider_items(conn, job_id, limit=1) -> list[dict]:
    rows = conn.execute(
        """
        SELECT symbol, yahoo_ticker FROM universal_cleaner_items
        WHERE job_id=? AND insider_status='pending' ORDER BY symbol LIMIT ?
        """,
        (job_id, int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]


def record_insider_signal(conn, job_id, symbol, signal) -> None:
    available = signal.get("status") == "available"
    conn.execute(
        """
        UPDATE universal_cleaner_items
        SET insider_status=?, insider_net_sale_ratio=?, insider_sale_value=?, updated_at=?
        WHERE job_id=? AND symbol=?
        """,
        (
            "ready" if available else "unavailable",
            signal.get("net_sale_ratio") if available else None,
            signal.get("sale_value") if available else None,
            _now(),
            job_id,
            symbol,
        ),
    )
    conn.commit()


def finalize_cleaner_job(conn, job_id, min_history_coverage=0.80) -> dict:
    job = get_cleaner_job(conn, job_id)
    if job["counts"]["history_pending"] or job["counts"]["insider_pending"]:
        raise ValueError("Cleaner scan is incomplete.")
    frame = pd.read_sql_query(
        "SELECT * FROM universal_cleaner_items WHERE job_id=?",
        conn,
        params=(job_id,),
    )
    coverage = float(frame["history_status"].isin(["ready", "insufficient_history"]).mean())
    if coverage < min_history_coverage:
        note = (
            f"Only {coverage:.1%} of selected symbols returned usable history. "
            "No removal proposal was produced. Retry when market data is available."
        )
        conn.execute(
            "UPDATE universal_cleaner_jobs SET status='blocked', note=?, updated_at=? WHERE job_id=?",
            (note, _now(), job_id),
        )
        conn.commit()
        return get_cleaner_job(conn, job_id)

    insider_ratio = pd.to_numeric(
        frame["insider_net_sale_ratio"], errors="coerce"
    ).astype(float)
    observed = insider_ratio.dropna()
    if observed.empty:
        insider_pct = pd.Series(0.0, index=frame.index)
    else:
        insider_pct = insider_ratio.rank(pct=True).fillna(0.0)
    frame["insider_percentile"] = insider_pct
    frame["final_score"] = (
        frame["technical_score"].fillna(0.0)
        + 0.35 * frame["insider_percentile"]
        + 2.0 * frame["history_status"].eq("no_price").astype(float)
    )
    qualifies = pd.Series(False, index=frame.index)
    if "unavailable" in job["criteria"]:
        qualifies |= frame["history_status"].eq("no_price")
    if "bearish" in job["criteria"]:
        qualifies |= frame["bearish_stack"].eq(1)
    if "management" in job["criteria"]:
        qualifies |= (
            insider_ratio.fillna(0.0).gt(0)
            & frame["insider_percentile"].ge(0.80)
        )
    qualifies &= frame["protected"].eq(0)
    eligible_count = int(frame["protected"].eq(0).sum())
    quota = int(math.floor(eligible_count * job["removal_percentile"] / 100))
    if quota <= 0 and qualifies.any():
        quota = 1
    selected = (
        frame.loc[qualifies]
        .sort_values(
            ["final_score", "technical_score", "symbol"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(quota)
    )
    selected_symbols = set(selected["symbol"])
    score_by_symbol = frame.set_index("symbol")["final_score"].to_dict()
    now = _now()
    conn.execute("BEGIN")
    try:
        for row in frame.itertuples(index=False):
            reasons = []
            if row.history_status == "no_price":
                reasons.append("No usable one-year price history")
            if int(row.bearish_stack or 0):
                reasons.append("Price < 25-DMA < 50-DMA < 200-DMA")
            if pd.notna(row.insider_net_sale_ratio) and row.insider_net_sale_ratio > 0:
                reasons.append("Recent management net selling")
            reason = "; ".join(reasons) or row.reason
            conn.execute(
                """
                UPDATE universal_cleaner_items
                SET final_score=?, proposed_remove=?, reason=?, updated_at=?
                WHERE job_id=? AND symbol=?
                """,
                (
                    float(score_by_symbol[row.symbol]),
                    int(row.symbol in selected_symbols),
                    reason,
                    now,
                    job_id,
                    row.symbol,
                ),
            )
        conn.execute(
            "UPDATE universal_cleaner_jobs SET status='ready', note=?, updated_at=? WHERE job_id=?",
            (
                f"Proposed {len(selected_symbols)} removals, capped at "
                f"{job['removal_percentile']:.1f}% of unprotected selected symbols.",
                now,
                job_id,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return get_cleaner_job(conn, job_id)


def cleaner_preview_frame(conn, job_id) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT symbol AS Symbol, exchange AS Exchange, currency AS Currency,
               latest_price AS Price, dma25 AS "25-DMA", dma50 AS "50-DMA",
               dma200 AS "200-DMA", insider_net_sale_ratio AS "Management net-sale ratio",
               final_score AS "Removal score", reason AS Reason
        FROM universal_cleaner_items
        WHERE job_id=? AND proposed_remove=1
        ORDER BY final_score DESC, symbol
        """,
        conn,
        params=(job_id,),
    )


def cleaner_audit_frame(conn, job_id) -> pd.DataFrame:
    """Return a complete, human-reviewable snapshot of a cleaner job."""
    return pd.read_sql_query(
        """
        SELECT symbol AS Symbol, exchange AS Exchange, currency AS Currency,
               CASE WHEN protected=1 THEN 'Protected holding'
                    WHEN proposed_remove=1 THEN 'Proposed removal'
                    ELSE 'Keep' END AS Decision,
               history_status AS "Price-history status", latest_price AS Price,
               dma25 AS "25-DMA", dma50 AS "50-DMA", dma200 AS "200-DMA",
               insider_status AS "Management-data status",
               insider_net_sale_ratio AS "Management net-sale ratio",
               final_score AS "Removal score", reason AS Reason
        FROM universal_cleaner_items
        WHERE job_id=?
        ORDER BY proposed_remove DESC, protected DESC, final_score DESC, symbol
        """,
        conn,
        params=(job_id,),
    )


def apply_cleaner_job(conn, job_id, universal_owner) -> dict:
    job = get_cleaner_job(conn, job_id)
    if job["status"] != "ready":
        raise ValueError("Cleaner proposal is not ready to apply.")
    rows = conn.execute(
        "SELECT symbol, yahoo_ticker FROM universal_cleaner_items "
        "WHERE job_id=? AND proposed_remove=1",
        (job_id,),
    ).fetchall()
    protected_now = {
        str(row["yahoo_ticker"] or "").upper()
        for row in conn.execute(
            "SELECT yahoo_ticker FROM master_holdings "
            "WHERE owner<>? AND quantity>0 AND yahoo_ticker IS NOT NULL",
            (universal_owner,),
        ).fetchall()
    }
    removed = []
    skipped = []
    conn.execute("BEGIN")
    try:
        for row in rows:
            if str(row["yahoo_ticker"] or "").upper() in protected_now:
                skipped.append(row["symbol"])
                continue
            cursor = conn.execute(
                "DELETE FROM master_holdings WHERE owner=? AND symbol=?",
                (universal_owner, row["symbol"]),
            )
            if cursor.rowcount:
                removed.append(row["symbol"])
        now = _now()
        conn.execute(
            "UPDATE universal_cleaner_jobs SET status='applied', applied_at=?, updated_at=? WHERE job_id=?",
            (now, now, job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"removed": removed, "protected_at_apply": skipped}
