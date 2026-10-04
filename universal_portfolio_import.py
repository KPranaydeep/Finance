"""Checkpointed CSV imports for the shared Universal Portfolio."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd


IMPORT_SCHEMA_VERSION = "universal-import-v2-market-cap"

JOB_DDL = """
CREATE TABLE IF NOT EXISTS universal_import_jobs (
    job_id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    file_name TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'merge',
    status TEXT NOT NULL,
    source_rows INTEGER NOT NULL,
    unique_symbols INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    applied_at TEXT
)
"""

ITEM_DDL = """
CREATE TABLE IF NOT EXISTS universal_import_items (
    job_id TEXT NOT NULL,
    input_order INTEGER NOT NULL,
    source_symbol TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    country TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    yahoo_ticker TEXT,
    exchange TEXT,
    currency TEXT,
    market_cap_millions REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_id, source_symbol),
    FOREIGN KEY (job_id) REFERENCES universal_import_jobs(job_id)
)
"""

ITEM_COLUMNS = [
    "Symbol",
    "Name",
    "Status",
    "Attempts",
    "Reason",
    "Yahoo ticker",
    "Exchange",
    "Currency",
    "Market cap (millions)",
]

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,31}$")


def ensure_import_schema(conn: sqlite3.Connection) -> None:
    conn.execute(JOB_DDL)
    conn.execute(ITEM_DDL)
    job_columns = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(universal_import_jobs)").fetchall()
    }
    if "mode" not in job_columns:
        conn.execute(
            "ALTER TABLE universal_import_jobs ADD COLUMN mode TEXT NOT NULL DEFAULT 'merge'"
        )
    item_columns = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(universal_import_items)").fetchall()
    }
    if "market_cap_millions" not in item_columns:
        conn.execute(
            "ALTER TABLE universal_import_items ADD COLUMN market_cap_millions REAL"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_universal_import_source "
        "ON universal_import_jobs(source_hash, updated_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_universal_import_pending "
        "ON universal_import_items(job_id, status, input_order)"
    )


def _normalized_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _find_column(columns, aliases, required=False):
    lookup = {_normalized_header(column): column for column in columns}
    for alias in aliases:
        match = lookup.get(_normalized_header(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(
            "The CSV needs a Symbol or Ticker column. No matching column was found."
        )
    return None


def parse_universal_source_csv(content: bytes, max_rows: int = 20_000) -> tuple[pd.DataFrame, dict]:
    """Normalize a third-party symbol CSV without executing or trusting its values."""
    if not content:
        raise ValueError("The uploaded CSV is empty.")
    if len(content) > 15 * 1024 * 1024:
        raise ValueError("The CSV is larger than the 15 MB safety limit.")

    last_error = None
    frame = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            frame = pd.read_csv(
                io.BytesIO(content),
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if frame is None:
        raise ValueError(f"The CSV encoding could not be read: {last_error}")
    if len(frame) > max_rows:
        raise ValueError(f"The CSV exceeds the {max_rows:,}-row safety limit.")

    symbol_col = _find_column(
        frame.columns, ("symbol", "ticker", "ticker symbol", "stock code"), required=True
    )
    name_col = _find_column(frame.columns, ("name", "company", "company name"))
    country_col = _find_column(frame.columns, ("country", "listing country"))
    market_cap_col = _find_column(
        frame.columns,
        (
            "marketCapitalizationMln",
            "market capitalization mln",
            "market cap millions",
            "market capitalization",
            "market cap",
            "marketcap",
        ),
    )

    source_rows = len(frame)
    normalized = pd.DataFrame()
    normalized["source_symbol"] = (
        frame[symbol_col].astype(str).str.strip().str.upper()
    )
    normalized["stock_name"] = (
        frame[name_col].astype(str).str.strip()
        if name_col is not None
        else normalized["source_symbol"]
    )
    normalized["country"] = (
        frame[country_col].astype(str).str.strip()
        if country_col is not None
        else ""
    )
    if market_cap_col is not None:
        cleaned_market_cap = (
            frame[market_cap_col]
            .astype(str)
            .str.replace(r"[^0-9eE+\-.]", "", regex=True)
        )
        normalized["market_cap_millions"] = pd.to_numeric(
            cleaned_market_cap, errors="coerce"
        )
        normalized.loc[
            normalized["market_cap_millions"] <= 0, "market_cap_millions"
        ] = pd.NA
    else:
        normalized["market_cap_millions"] = pd.NA
    normalized = normalized[normalized["source_symbol"].ne("")].copy()
    duplicate_count = int(normalized["source_symbol"].duplicated().sum())
    normalized = normalized.drop_duplicates("source_symbol", keep="first").reset_index(drop=True)
    normalized["input_order"] = normalized.index.astype(int)
    normalized["valid_symbol"] = normalized["source_symbol"].map(
        lambda value: bool(_SYMBOL_RE.fullmatch(value))
    )
    normalized["stock_name"] = normalized["stock_name"].where(
        normalized["stock_name"].ne(""), normalized["source_symbol"]
    )
    return normalized, {
        "source_rows": int(source_rows),
        "unique_symbols": int(len(normalized)),
        "duplicates_removed": duplicate_count,
        "blank_symbols_removed": int(source_rows - len(normalized) - duplicate_count),
    }


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_india_listing(row) -> bool:
    ticker = str(row["yahoo_ticker"] or row["symbol"] or "").strip().upper()
    exchange = str(row["exchange"] or "").strip().upper()
    currency = str(row["currency"] or "").strip().upper()
    return (
        ticker.endswith((".NS", ".BO"))
        or exchange in {"NSE", "NSI", "BSE", "BOM"}
        or currency == "INR"
    )


def prepare_import_job(
    conn: sqlite3.Connection,
    content: bytes,
    file_name: str,
    owner: str,
    mode: str = "merge",
) -> tuple[str, bool, dict]:
    """Create a durable staging job, or resume the same source file idempotently."""
    ensure_import_schema(conn)
    mode = str(mode or "merge").strip().lower()
    if mode not in {"merge", "replace_overseas"}:
        raise ValueError("Unsupported Universal Portfolio import mode.")
    source_hash = hashlib.sha256(
        IMPORT_SCHEMA_VERSION.encode("utf-8") + b"\0" + content
    ).hexdigest()
    prior = conn.execute(
        """
        SELECT job_id FROM universal_import_jobs
        WHERE source_hash = ? AND COALESCE(mode, 'merge') = ?
        ORDER BY updated_at DESC LIMIT 1
        """,
        (source_hash, mode),
    ).fetchone()
    if prior is not None:
        job_id = str(prior["job_id"] if hasattr(prior, "keys") else prior[0])
        return job_id, True, get_import_job(conn, job_id)

    normalized, parse_stats = parse_universal_source_csv(content)
    existing_rows = conn.execute(
        "SELECT symbol, yahoo_ticker, exchange, currency "
        "FROM master_holdings WHERE owner = ?",
        (owner,),
    ).fetchall()
    by_symbol = {}
    by_ticker = {}
    for row in existing_rows:
        symbol = str(row["symbol"] or "").strip().upper()
        ticker = str(row["yahoo_ticker"] or "").strip().upper()
        if symbol:
            by_symbol[symbol] = row
        if ticker:
            by_ticker[ticker] = row

    job_id = "UNI-" + uuid.uuid4().hex.upper()
    now = _utc_now_text()
    conn.execute("BEGIN")
    try:
        conn.execute(
            """
            INSERT INTO universal_import_jobs
                (job_id, source_hash, file_name, mode, status, source_rows,
                 unique_symbols, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'prepared', ?, ?, ?, ?)
            """,
            (
                job_id,
                source_hash,
                str(file_name or "universal-source.csv")[:255],
                mode,
                parse_stats["source_rows"],
                parse_stats["unique_symbols"],
                now,
                now,
            ),
        )
        for row in normalized.itertuples(index=False):
            source_symbol = str(row.source_symbol)
            existing_row = by_symbol.get(source_symbol)
            existing_ticker = (
                str(existing_row["yahoo_ticker"] or "").strip().upper()
                if existing_row is not None
                else ""
            )
            ticker_row = by_ticker.get(source_symbol)
            if not bool(row.valid_symbol):
                status = "rejected"
                reason = "Symbol format is not supported."
            elif mode == "replace_overseas" and existing_row is not None:
                if _is_india_listing(existing_row):
                    status = "conflict"
                    reason = "Symbol key belongs to an India-listed security; preserved."
                else:
                    status = "pending"
                    reason = None
            elif mode == "replace_overseas" and ticker_row is not None:
                if _is_india_listing(ticker_row):
                    status = "conflict"
                    reason = "Ticker belongs to an India-listed security; preserved."
                else:
                    status = "pending"
                    reason = None
            elif source_symbol in by_ticker or existing_ticker == source_symbol:
                status = "existing"
                reason = "Already present in the Universal Portfolio."
            elif existing_ticker:
                status = "conflict"
                reason = (
                    f"Symbol key already belongs to {existing_ticker}; skipped to avoid "
                    "mixing exchanges."
                )
            else:
                status = "pending"
                reason = None
            conn.execute(
                """
                INSERT INTO universal_import_items
                    (job_id, input_order, source_symbol, stock_name, country,
                     status, reason, yahoo_ticker, market_cap_millions, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    int(row.input_order),
                    source_symbol,
                    str(row.stock_name)[:300],
                    str(row.country)[:100],
                    status,
                    reason,
                    source_symbol,
                    (
                        float(row.market_cap_millions)
                        if pd.notna(row.market_cap_millions)
                        else None
                    ),
                    now,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    result = get_import_job(conn, job_id)
    result["parse_stats"] = parse_stats
    return job_id, False, result


def get_import_job(conn: sqlite3.Connection, job_id: str) -> dict:
    ensure_import_schema(conn)
    job = conn.execute(
        "SELECT * FROM universal_import_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if job is None:
        raise ValueError("Universal import job was not found.")
    values = dict(job)
    counts = {
        str(row["status"]): int(row["count"])
        for row in conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM universal_import_items WHERE job_id = ? GROUP BY status
            """,
            (job_id,),
        ).fetchall()
    }
    values["counts"] = counts
    values["processed"] = int(values["unique_symbols"] - counts.get("pending", 0))
    values["progress"] = (
        values["processed"] / values["unique_symbols"]
        if values["unique_symbols"]
        else 1.0
    )
    return values


def latest_import_job(conn: sqlite3.Connection, mode: str = "merge") -> dict | None:
    ensure_import_schema(conn)
    row = conn.execute(
        "SELECT job_id FROM universal_import_jobs "
        "WHERE COALESCE(mode, 'merge')=? ORDER BY updated_at DESC LIMIT 1",
        (str(mode or "merge").strip().lower(),),
    ).fetchone()
    if row is None:
        return None
    return get_import_job(conn, str(row["job_id"]))


def pending_import_items(conn: sqlite3.Connection, job_id: str, limit: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT source_symbol, stock_name, country, attempts
        FROM universal_import_items
        WHERE job_id = ? AND status = 'pending'
        ORDER BY attempts, input_order LIMIT ?
        """,
        (job_id, int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]


def record_validation_batch(
    conn: sqlite3.Connection,
    job_id: str,
    attempted_symbols: list[str],
    active_metadata: dict[str, dict],
    max_attempts: int = 3,
) -> dict:
    """Checkpoint one validation batch; unverified rows get bounded retries."""
    now = _utc_now_text()
    conn.execute("BEGIN")
    try:
        for symbol in attempted_symbols:
            metadata = active_metadata.get(symbol)
            if metadata is not None:
                conn.execute(
                    """
                    UPDATE universal_import_items
                    SET status = 'verified', attempts = attempts + 1,
                        reason = 'Recent usable Yahoo price verified.',
                        yahoo_ticker = ?, exchange = ?, currency = ?, updated_at = ?
                    WHERE job_id = ? AND source_symbol = ? AND status = 'pending'
                    """,
                    (
                        str(metadata.get("yahoo_ticker") or symbol).upper(),
                        str(metadata.get("exchange") or "US/Global"),
                        str(metadata.get("currency") or "USD"),
                        now,
                        job_id,
                        symbol,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE universal_import_items
                    SET attempts = attempts + 1,
                        status = CASE WHEN attempts + 1 >= ? THEN 'unresolved' ELSE 'pending' END,
                        reason = CASE WHEN attempts + 1 >= ?
                            THEN 'No recent usable Yahoo price after repeated validation; omitted safely.'
                            ELSE 'No recent usable Yahoo price yet; queued for another pass.' END,
                        updated_at = ?
                    WHERE job_id = ? AND source_symbol = ? AND status = 'pending'
                    """,
                    (max_attempts, max_attempts, now, job_id, symbol),
                )
        pending = int(
            conn.execute(
                "SELECT COUNT(*) FROM universal_import_items "
                "WHERE job_id = ? AND status = 'pending'",
                (job_id,),
            ).fetchone()[0]
        )
        conn.execute(
            "UPDATE universal_import_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            ("ready" if pending == 0 else "validating", now, job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return get_import_job(conn, job_id)


def apply_verified_import(conn: sqlite3.Connection, job_id: str, owner: str) -> dict:
    """Atomically merge verified rows; never delete or overwrite existing rows."""
    job = get_import_job(conn, job_id)
    if job["counts"].get("pending", 0):
        raise ValueError("Validation is incomplete. Resume it before applying additions.")
    rows = conn.execute(
        """
        SELECT source_symbol, stock_name, yahoo_ticker, exchange, currency,
               market_cap_millions
        FROM universal_import_items
        WHERE job_id = ? AND status = 'verified' ORDER BY input_order
        """,
        (job_id,),
    ).fetchall()
    now = _utc_now_text()
    added = 0
    already_present = 0
    conn.execute("BEGIN")
    try:
        for row in rows:
            before = conn.total_changes
            conn.execute(
                """
                INSERT INTO master_holdings
                    (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                     quantity, average_price, market_cap_millions, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
                ON CONFLICT(owner, symbol) DO NOTHING
                """,
                (
                    owner,
                    row["source_symbol"],
                    row["stock_name"],
                    row["yahoo_ticker"],
                    row["exchange"],
                    row["currency"],
                    row["market_cap_millions"],
                    now,
                    now,
                ),
            )
            if conn.total_changes > before:
                added += 1
                item_status = "added"
                reason = "Added to the Universal Portfolio."
            else:
                already_present += 1
                item_status = "existing"
                reason = "Already present when the staged import was applied."
            conn.execute(
                """
                UPDATE universal_import_items SET status = ?, reason = ?, updated_at = ?
                WHERE job_id = ? AND source_symbol = ?
                """,
                (item_status, reason, now, job_id, row["source_symbol"]),
            )
        conn.execute(
            """
            UPDATE universal_import_jobs
            SET status = 'applied', updated_at = ?, applied_at = ? WHERE job_id = ?
            """,
            (now, now, job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {
        "added": added,
        "already_present": already_present,
        "job": get_import_job(conn, job_id),
    }


def apply_verified_overseas_replacement(
    conn: sqlite3.Connection,
    job_id: str,
    owner: str,
    min_verified_ratio: float = 0.50,
) -> dict:
    """Atomically retain India listings and replace every overseas universe row."""
    job = get_import_job(conn, job_id)
    if job.get("mode") != "replace_overseas":
        raise ValueError("This staged job is not an overseas replacement.")
    if job["counts"].get("pending", 0):
        raise ValueError("Validation is incomplete. Resume it before replacing overseas rows.")
    verified_count = int(job["counts"].get("verified", 0))
    verified_ratio = verified_count / max(int(job["unique_symbols"]), 1)
    if verified_count == 0 or verified_ratio < float(min_verified_ratio):
        raise ValueError(
            f"Only {verified_ratio:.1%} of uploaded symbols were verified. "
            "The overseas universe was not changed."
        )
    verified_rows = conn.execute(
        """
        SELECT source_symbol, stock_name, yahoo_ticker, exchange, currency,
               market_cap_millions
        FROM universal_import_items
        WHERE job_id=? AND status='verified' ORDER BY input_order
        """,
        (job_id,),
    ).fetchall()
    current_rows = conn.execute(
        """
        SELECT symbol, yahoo_ticker, exchange, currency
        FROM master_holdings WHERE owner=?
        """,
        (owner,),
    ).fetchall()
    overseas_symbols = [
        str(row["symbol"]) for row in current_rows if not _is_india_listing(row)
    ]
    retained_india = len(current_rows) - len(overseas_symbols)
    now = _utc_now_text()
    added = 0
    conn.execute("BEGIN")
    try:
        for symbol in overseas_symbols:
            conn.execute(
                "DELETE FROM master_holdings WHERE owner=? AND symbol=?",
                (owner, symbol),
            )
        for row in verified_rows:
            before = conn.total_changes
            conn.execute(
                """
                INSERT INTO master_holdings
                    (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                     quantity, average_price, market_cap_millions, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
                ON CONFLICT(owner, symbol) DO NOTHING
                """,
                (
                    owner,
                    row["source_symbol"],
                    row["stock_name"],
                    row["yahoo_ticker"],
                    row["exchange"],
                    row["currency"],
                    row["market_cap_millions"],
                    now,
                    now,
                ),
            )
            if conn.total_changes > before:
                added += 1
            conn.execute(
                """
                UPDATE universal_import_items
                SET status='added', reason='Added by overseas replacement.', updated_at=?
                WHERE job_id=? AND source_symbol=?
                """,
                (now, job_id, row["source_symbol"]),
            )
        conn.execute(
            """
            UPDATE universal_import_jobs
            SET status='applied', updated_at=?, applied_at=? WHERE job_id=?
            """,
            (now, now, job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {
        "removed_overseas": len(overseas_symbols),
        "added_overseas": added,
        "retained_india": retained_india,
        "verified_ratio": verified_ratio,
        "job": get_import_job(conn, job_id),
    }


def import_items_frame(conn: sqlite3.Connection, job_id: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT source_symbol AS Symbol, stock_name AS Name, status AS Status,
               attempts AS Attempts, reason AS Reason, yahoo_ticker AS "Yahoo ticker",
               exchange AS Exchange, currency AS Currency,
               market_cap_millions AS "Market cap (millions)"
        FROM universal_import_items WHERE job_id = ? ORDER BY input_order
        """,
        conn,
        params=(job_id,),
    )
    return frame.reindex(columns=ITEM_COLUMNS)


def safe_report_csv(frame: pd.DataFrame) -> bytes:
    """Export audit rows while neutralizing spreadsheet formula prefixes."""
    output = frame.copy()
    for column in output.select_dtypes(include="object").columns:
        output[column] = output[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
            else value
        )
    buffer = io.StringIO(newline="")
    output.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
    return buffer.getvalue().encode("utf-8-sig")
