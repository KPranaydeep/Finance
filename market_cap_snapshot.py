"""Safe, explicit refresh of market-cap snapshots for existing universe rows."""

from __future__ import annotations

import io
import json
import random
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


def refresh_market_cap_sample(
    conn: sqlite3.Connection,
    rows: pd.DataFrame,
    *,
    owner: str,
    sample_fraction: float = 0.01,
    max_age_days: int = 90,
    as_of: date | str | None = None,
    progress_callback=None,
    random_seed: int | None = None,
    excluded_tickers: set[str] | None = None,
) -> dict:
    """Refresh a bounded sample of missing/stale caps from Yahoo fast_info.

    This is deliberately a small opportunistic repair, not a replacement for
    the quarterly verified snapshot. Existing fresh values are never touched;
    failures are returned as diagnostics and do not stop optimization.
    """
    if rows is None or rows.empty or not 0 < float(sample_fraction) <= 1:
        return {"eligible": 0, "attempted": 0, "updated": 0, "failed": 0, "failures": []}
    frame = rows.copy()
    ticker_col = "Yahoo Ticker" if "Yahoo Ticker" in frame.columns else "yahoo_ticker"
    if ticker_col not in frame.columns:
        return {"eligible": 0, "attempted": 0, "updated": 0, "failed": 0, "failures": []}
    cap_col = "Market Cap Millions" if "Market Cap Millions" in frame.columns else "market_cap_millions"
    asof_col = "Market Cap As Of" if "Market Cap As Of" in frame.columns else "market_cap_as_of"
    caps = pd.to_numeric(frame.get(cap_col), errors="coerce") if cap_col in frame else pd.Series(index=frame.index, dtype=float)
    dates = pd.to_datetime(frame.get(asof_col), errors="coerce", utc=True) if asof_col in frame else pd.Series(index=frame.index, dtype="datetime64[ns, UTC]")
    today = pd.Timestamp(as_of or pd.Timestamp.now(tz="Asia/Kolkata").date(), tz="UTC").normalize()
    cutoff = today - pd.Timedelta(days=max(int(max_age_days), 0))
    needs_refresh = caps.le(0) | dates.isna() | dates.lt(cutoff) | dates.gt(today)
    candidates = frame.loc[needs_refresh].copy()
    candidates["_ticker"] = candidates[ticker_col].astype(str).str.strip().str.upper()
    candidates = candidates[candidates["_ticker"].ne("")].drop_duplicates("_ticker")
    eligible = int(len(candidates))
    limit = max(1, int(np.ceil(eligible * float(sample_fraction)))) if eligible else 0
    excluded = {str(value).upper() for value in (excluded_tickers or set())}
    candidates = candidates[~candidates["_ticker"].isin(excluded)]
    sampler = random.Random(random_seed)
    selected = candidates.iloc[sampler.sample(range(len(candidates)), min(limit, len(candidates)))] if len(candidates) else candidates
    failures = []
    updates = []
    try:
        import yfinance as yf
    except Exception as exc:
        return {"eligible": eligible, "attempted": 0, "attempted_tickers": [], "updated": 0, "failed": 0, "failures": [type(exc).__name__]}
    selected_tickers = selected["_ticker"].tolist()
    for index, ticker in enumerate(selected_tickers, start=1):
        try:
            fast_info = yf.Ticker(ticker).fast_info
            value = None
            if hasattr(fast_info, "get"):
                # yfinance exposes this field as camelCase in FastInfo.
                value = fast_info.get("marketCap")
                if value is None:
                    value = fast_info.get("market_cap")
            value = float(value) / 1_000_000 if value is not None else 0.0
            if value <= 0:
                raise ValueError("no usable market cap")
            updates.append((value, "Yahoo Finance fast_info", today.date().isoformat(), datetime.now(timezone.utc).isoformat(timespec="seconds"), owner, ticker))
        except Exception as exc:
            failures.append(f"{ticker}:{type(exc).__name__}")
        if progress_callback is not None:
            try:
                progress_callback(index, len(selected_tickers), ticker)
            except Exception:
                pass
    ensure_market_cap_metadata_schema(conn)
    for row in updates:
        conn.execute(
            """UPDATE master_holdings
               SET market_cap_millions=?, market_cap_source=?, market_cap_as_of=?, market_cap_imported_at=?
               WHERE owner=? AND UPPER(TRIM(yahoo_ticker))=?""",
            row,
        )
    conn.commit()
    return {"eligible": eligible, "attempted": int(len(selected)), "attempted_tickers": selected_tickers, "updated": int(len(updates)), "failed": int(len(failures)), "failures": failures[:10], "source": "Yahoo Finance fast_info", "as_of": today.date().isoformat()}


def refresh_market_cap_until_stalled(
    conn: sqlite3.Connection,
    rows: pd.DataFrame,
    *,
    owner: str,
    sample_fraction: float = 0.01,
    max_age_days: int = 90,
    max_rounds: int = 20,
    max_failure_fraction: float = 0.40,
    progress_callback=None,
    checkpoint_path: str | Path | None = None,
    random_seed: int | None = None,
) -> dict:
    """Run bounded 1% refresh batches until no new values can be obtained."""
    checkpoint = Path(checkpoint_path or (Path.home() / ".portfolio_rebalancer" / "market_cap_refresh_checkpoint.json"))
    if random_seed is None:
        random_seed = time.time_ns() % (2**31 - 1)
    completed_tickers: set[str] = set()
    if checkpoint.exists():
        try:
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            completed_tickers = {str(value).upper() for value in saved.get("attempted_tickers", [])}
        except Exception:
            completed_tickers = set()
    aggregate = {"rounds": 0, "eligible": 0, "attempted": 0, "updated": 0, "failed": 0, "failures": [], "resumed": bool(completed_tickers)}
    current_rows = rows
    for round_number in range(1, max(int(max_rounds), 1) + 1):
        result = refresh_market_cap_sample(
            conn,
            current_rows,
            owner=owner,
            sample_fraction=sample_fraction,
            max_age_days=max_age_days,
            progress_callback=(
                (lambda done, total, ticker, r=round_number: progress_callback(r, max_rounds, done, total, ticker))
                if progress_callback is not None else None
            ),
            random_seed=(int(random_seed or 0) + round_number),
            excluded_tickers=completed_tickers,
        )
        aggregate["rounds"] = round_number
        for key in ("eligible", "attempted", "updated", "failed"):
            aggregate[key] += int(result.get(key, 0) or 0)
        aggregate["failures"].extend(result.get("failures") or [])
        completed_tickers.update(result.get("attempted_tickers") or [])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"attempted_tickers": sorted(completed_tickers), "updated_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        batch_attempted = int(result.get("attempted", 0) or 0)
        batch_failed = int(result.get("failed", 0) or 0)
        if batch_attempted and (batch_failed / batch_attempted) > float(max_failure_fraction):
            aggregate["stopped_reason"] = "failure_threshold"
            break
        if not result.get("updated"):
            aggregate["stopped_reason"] = "no_new_values"
            break
        current_rows = pd.read_sql_query(
            "SELECT symbol AS Symbol, yahoo_ticker AS \"Yahoo Ticker\", market_cap_millions AS \"Market Cap Millions\", market_cap_as_of AS \"Market Cap As Of\" FROM master_holdings WHERE owner=?",
            conn,
            params=(owner,),
        )
    else:
        aggregate["stopped_reason"] = "round_limit"
    aggregate["failures"] = aggregate["failures"][:20]
    if aggregate.get("stopped_reason") == "no_new_values":
        try:
            checkpoint.unlink(missing_ok=True)
        except Exception:
            pass
    aggregate["as_of"] = pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()
    return aggregate


def ensure_market_cap_metadata_schema(conn: sqlite3.Connection) -> None:
    """Add nullable provenance columns without changing existing holdings."""
    columns = {
        str(row[1]).lower()
        for row in conn.execute("PRAGMA table_info(master_holdings)").fetchall()
    }
    additions = {
        "market_cap_source": "ALTER TABLE master_holdings ADD COLUMN market_cap_source TEXT",
        "market_cap_as_of": "ALTER TABLE master_holdings ADD COLUMN market_cap_as_of TEXT",
        "market_cap_imported_at": "ALTER TABLE master_holdings ADD COLUMN market_cap_imported_at TEXT",
    }
    for name, statement in additions.items():
        if name not in columns:
            conn.execute(statement)


def prepare_market_cap_snapshot(
    content: bytes,
    master_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Match a ticker+market-cap CSV to existing holdings by exact Yahoo ticker.

    Matching only exact Yahoo ticker values prevents same-name/symbol listings on
    different exchanges from silently receiving one another's market cap.
    """
    if not content:
        raise ValueError("Upload a non-empty CSV snapshot.")
    if len(content) > 15 * 1024 * 1024:
        raise ValueError("The CSV is larger than the 15 MB safety limit.")
    frame = None
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding=encoding, dtype=str, keep_default_na=False)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if frame is None:
        raise ValueError(f"The CSV encoding could not be read: {last_error}")
    if len(frame) > 20_000:
        raise ValueError("The CSV exceeds the 20,000-row safety limit.")

    from universal_portfolio_import import _find_column

    ticker_col = _find_column(
        frame.columns, ("yahoo ticker", "yahoo symbol", "ticker", "symbol", "ticker symbol"), required=True
    )
    cap_col = _find_column(
        frame.columns,
        ("marketCapitalizationMln", "market cap millions", "market capitalization mln"),
    )
    if cap_col is None:
        raise ValueError(
            "This CSV needs a market-cap column explicitly expressed in millions "
            "(for example, marketCapitalizationMln)."
        )

    incoming = pd.DataFrame({
        "CSV ticker": frame[ticker_col].astype(str).str.strip().str.upper(),
        "Market Cap Millions": pd.to_numeric(
            frame[cap_col].astype(str).str.replace(r"[^0-9eE+\-.]", "", regex=True),
            errors="coerce",
        ),
    })
    incoming = incoming[incoming["CSV ticker"].ne("")].copy()
    incoming.loc[incoming["Market Cap Millions"] <= 0, "Market Cap Millions"] = pd.NA
    duplicate_tickers = set(
        incoming.loc[incoming["CSV ticker"].duplicated(keep=False), "CSV ticker"]
    )

    if master_rows is None or master_rows.empty:
        incoming["Status"] = "No holdings to match"
        incoming["Symbol"] = ""
        incoming["Yahoo Ticker"] = ""
        return incoming.reset_index(drop=True), {
            "uploaded": int(len(incoming)), "matched": 0, "unmatched": int(len(incoming)),
            "missing_cap": int(incoming["Market Cap Millions"].isna().sum()),
        }

    master = master_rows.copy()
    master["_ticker_key"] = master["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    ticker_counts = master.loc[master["_ticker_key"].ne("")].groupby("_ticker_key").size().to_dict()
    by_ticker = master.drop_duplicates("_ticker_key", keep=False).set_index("_ticker_key")
    result = []
    for row in incoming.itertuples(index=False, name=None):
        ticker = str(row[0])
        cap = row[1]
        count = int(ticker_counts.get(ticker, 0))
        if ticker in duplicate_tickers:
            status, symbol, yahoo_ticker = "Duplicate ticker in CSV", "", ""
        elif count == 1:
            matched = by_ticker.loc[ticker]
            status = "Ready to update" if pd.notna(cap) else "Matched; no usable cap"
            symbol = str(matched["Symbol"])
            yahoo_ticker = str(matched["Yahoo Ticker"])
        elif count > 1:
            status, symbol, yahoo_ticker = "Ambiguous Yahoo ticker", "", ""
        else:
            status, symbol, yahoo_ticker = "No exact Yahoo-ticker match", "", ""
        result.append({
            "CSV ticker": ticker,
            "Symbol": symbol,
            "Yahoo Ticker": yahoo_ticker,
            "Market Cap Millions": cap,
            "Status": status,
        })
    preview = pd.DataFrame(result)
    counts = {
        "uploaded": int(len(preview)),
        "matched": int(preview["Yahoo Ticker"].ne("").sum()),
        "ready": int(preview["Status"].eq("Ready to update").sum()),
        "unmatched": int(preview["Status"].eq("No exact Yahoo-ticker match").sum()),
        "ambiguous": int(preview["Status"].eq("Ambiguous Yahoo ticker").sum()),
        "duplicate_source": int(preview["Status"].eq("Duplicate ticker in CSV").sum()),
        "missing_cap": int(preview["Market Cap Millions"].isna().sum()),
    }
    return preview, counts


def apply_market_cap_snapshot(
    conn: sqlite3.Connection,
    preview: pd.DataFrame,
    *,
    owner: str,
    source: str,
    as_of: date | str,
) -> dict:
    """Apply only matched positive caps; do not alter holdings or clear other data."""
    source = str(source or "").strip()[:160]
    if not source:
        raise ValueError("Enter the market-cap data source/provider.")
    snapshot_date = pd.Timestamp(as_of).date()
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    if snapshot_date > today:
        raise ValueError("The market-cap as-of date cannot be in the future.")
    ensure_market_cap_metadata_schema(conn)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ready = preview.loc[
        preview["Status"].eq("Ready to update"),
        ["Yahoo Ticker", "Market Cap Millions"],
    ]
    updated = 0
    conn.execute("BEGIN")
    try:
        for row in ready.itertuples(index=False, name=None):
            ticker, cap = str(row[0]), float(row[1])
            if cap <= 0:
                continue
            cursor = conn.execute(
                """UPDATE master_holdings
                   SET market_cap_millions=?, market_cap_source=?, market_cap_as_of=?,
                       market_cap_imported_at=?
                   WHERE owner=? AND UPPER(TRIM(yahoo_ticker))=?""",
                (cap, source, snapshot_date.isoformat(), now, owner, ticker.upper()),
            )
            updated += int(cursor.rowcount or 0)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"updated": updated, "as_of": snapshot_date.isoformat(), "source": source}


def summarize_market_cap_coverage(
    rows: pd.DataFrame,
    *,
    as_of: date | str | None = None,
    max_age_days: int = 90,
) -> dict:
    """Summarize usable, stale, and absent values without inferring freshness."""
    if rows is None or rows.empty:
        return {"total": 0, "current": 0, "stale": 0, "missing": 0, "coverage": 0.0}
    frame = rows.copy()
    caps = pd.to_numeric(frame.get("Market Cap Millions"), errors="coerce")
    dates = pd.to_datetime(frame.get("Market Cap As Of"), errors="coerce", utc=True)
    today = pd.Timestamp(
        as_of or pd.Timestamp.now(tz="Asia/Kolkata").date(), tz="UTC"
    ).normalize()
    cutoff = today - pd.Timedelta(days=max(int(max_age_days), 0))
    valid_cap = caps.gt(0)
    current = valid_cap & dates.ge(cutoff) & dates.le(today)
    stale = valid_cap & ~current
    missing = ~valid_cap
    sources = (
        frame.loc[current, "Market Cap Source"].fillna("").astype(str).str.strip()
        if "Market Cap Source" in frame
        else pd.Series(dtype=str)
    )
    return {
        "total": int(len(frame)),
        "current": int(current.sum()),
        "stale": int(stale.sum()),
        "missing": int(missing.sum()),
        "coverage": float(current.mean()) if len(frame) else 0.0,
        "oldest_current_as_of": (
            dates.loc[current].min().date().isoformat() if current.any() else None
        ),
        "newest_current_as_of": (
            dates.loc[current].max().date().isoformat() if current.any() else None
        ),
        "sources": sorted({value for value in sources if value}),
        "max_age_days": max(int(max_age_days), 0),
    }
