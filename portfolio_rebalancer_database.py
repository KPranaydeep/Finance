import io
import html
import json
import os
import re
import sqlite3
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Portfolio Rebalancer", layout="wide")

# Mobile-friendly tweaks: stack side-by-side columns vertically on narrow
# screens, enlarge tap targets, and let wide tables scroll horizontally
# instead of being clipped. No widgets/features are removed by this CSS.
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.6rem 2rem 0.6rem; }
        [data-testid="stHorizontalBlock"] { flex-direction: column; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        [data-testid="stMetric"] { padding: 0.4rem 0; }
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1.05rem !important; }
        [data-testid="stVerticalBlockBorderWrapper"] { padding: 0.75rem !important; }
    }
    button[kind="primary"], button[kind="secondary"] { min-height: 2.75rem; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_BUILD = "2026-08-13-yf-resilient-downloads-v1"

# =========================================================
# HELPERS
# =========================================================

@st.cache_data(show_spinner=False)
def load_equity_mapping():
    url = "https://raw.githubusercontent.com/KPranaydeep/Finance/refs/heads/main/EQUITY_L.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df[["ISIN NUMBER", "SYMBOL", "NAME OF COMPANY"]].rename(columns={
        "ISIN NUMBER": "ISIN",
        "SYMBOL": "Symbol",
        "NAME OF COMPANY": "Company Name"
    })


@st.cache_data(show_spinner=False)
def _download_probe(ticker):
    """Return True only when Yahoo Finance has at least one real recent price value."""
    try:
        data = _yf_download_quiet(
            ticker,
            period="10d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data is None or data.empty:
            return False

        # yfinance can occasionally return a non-empty frame whose price columns are
        # entirely NaN for an invalid symbol. Treat that as unresolved.
        if isinstance(data, pd.Series):
            return bool(pd.to_numeric(data, errors="coerce").notna().any())

        numeric = data.apply(pd.to_numeric, errors="coerce")
        return bool(numeric.notna().any().any())
    except Exception:
        return False


def normalize_portfolio_symbol(value):
    """Normalize a user-facing holding key while preserving non-India Yahoo suffixes."""
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        symbol = symbol[:-3]
    return symbol


def _infer_market_from_ticker(ticker):
    ticker = str(ticker or "").upper()
    if ticker.endswith(".NS"):
        return "NSE", "INR"
    if ticker.endswith(".BO"):
        return "BSE", "INR"
    if ticker.endswith(".L"):
        return "LSE", "GBP"
    if ticker.endswith(".TO"):
        return "TSX", "CAD"
    if ticker.endswith(".AX"):
        return "ASX", "AUD"
    if ticker.endswith(".HK"):
        return "HKEX", "HKD"
    if ticker.endswith(".T"):
        return "TSE", "JPY"
    return "US/Global", "USD"


def _read_fast_info_value(fast_info, key):
    try:
        if hasattr(fast_info, "get"):
            return fast_info.get(key)
        return getattr(fast_info, key, None)
    except Exception:
        return None


def _chunked(values, size):
    for idx in range(0, len(values), size):
        yield values[idx:idx + size]


def _is_rate_limit_error(exc):
    """Detect Yahoo Finance rate-limit errors across yfinance versions."""
    name = type(exc).__name__
    message = str(exc)
    return (
        "RateLimit" in name
        or "Too Many Requests" in message
        or "rate limited" in message.lower()
    )


def _yf_download_quiet(*args, max_retries=2, retry_backoff_seconds=3, **kwargs):
    """Call yf.download with console noise suppressed and rate-limit retries."""
    import yfinance as yf

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return yf.download(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < max_retries:
                time.sleep(retry_backoff_seconds * (attempt + 1))
                continue
            raise
    raise last_exc


def _extract_close_prices_frame(data, expected_tickers):
    """Normalize yfinance output into a Close-price DataFrame keyed by ticker."""
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.Series):
        if len(expected_tickers) == 1:
            return data.to_frame(name=expected_tickers[0]).astype(float)
        return pd.DataFrame()

    if getattr(data, "empty", True):
        return pd.DataFrame()

    frame = data
    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" in frame.columns.get_level_values(0):
            frame = frame["Close"]
        elif "Adj Close" in frame.columns.get_level_values(0):
            frame = frame["Adj Close"]
        else:
            return pd.DataFrame()
    else:
        if "Close" in frame.columns:
            if len(expected_tickers) == 1:
                frame = frame[["Close"]].rename(columns={"Close": expected_tickers[0]})
            else:
                return pd.DataFrame()

    if isinstance(frame, pd.Series):
        column_name = expected_tickers[0] if len(expected_tickers) == 1 else str(frame.name)
        frame = frame.to_frame(name=column_name)

    frame.columns = [str(col).strip().upper() for col in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame


def _download_close_prices_resilient(
    tickers,
    *,
    start=None,
    end=None,
    period=None,
    batch_size=12,
):
    """Download close-price history with chunking and per-ticker fallback."""
    unique_tickers = [
        str(t).strip().upper()
        for t in dict.fromkeys(tickers)
        if str(t).strip()
    ]
    if not unique_tickers:
        return pd.DataFrame(), {}

    failures = {}
    collected_frames = []

    for batch in _chunked(unique_tickers, batch_size):
        batch_frame = pd.DataFrame()
        try:
            kwargs = {
                "progress": False,
                "auto_adjust": True,
                "threads": False,
            }
            if period is not None:
                kwargs["period"] = period
            else:
                kwargs["start"] = start
                kwargs["end"] = end

            downloaded = _yf_download_quiet(batch, **kwargs)
            batch_frame = _extract_close_prices_frame(downloaded, batch)
        except Exception as exc:
            batch_frame = pd.DataFrame()
            batch_error = str(exc) or exc.__class__.__name__
            for ticker in batch:
                failures.setdefault(ticker, batch_error)

        recovered = set(batch_frame.columns)
        for ticker in batch:
            if ticker in recovered:
                continue
            try:
                kwargs = {
                    "progress": False,
                    "auto_adjust": True,
                    "threads": False,
                }
                if period is not None:
                    kwargs["period"] = period
                else:
                    kwargs["start"] = start
                    kwargs["end"] = end

                single_download = _yf_download_quiet(ticker, **kwargs)
                single_frame = _extract_close_prices_frame(single_download, [ticker])
                if single_frame.empty:
                    failures.setdefault(ticker, "No usable price history returned by Yahoo Finance.")
                    continue

                batch_frame = pd.concat([batch_frame, single_frame], axis=1)
                failures.pop(ticker, None)
            except Exception as exc:
                failures[ticker] = str(exc) or exc.__class__.__name__

        if not batch_frame.empty:
            collected_frames.append(batch_frame)

    if not collected_frames:
        return pd.DataFrame(), failures

    merged = pd.concat(collected_frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()].sort_index()
    return merged, failures


def _format_download_failure_message(tickers, failures, context):
    total = len(tickers)
    failed = len(failures)
    if failed == 0:
        return f"No data available for the requested symbols ({context})."

    sample = sorted(failures)[:10]
    details = ", ".join(sample)
    suffix = "" if failed <= 10 else f" (+{failed - 10} more)"
    return (
        f"No usable price data could be downloaded ({context}). "
        f"Failed symbols: {details}{suffix}. "
        "Yahoo Finance may be rate-limiting requests or the symbols may be inactive."
        f" Requested symbols: {total}, failed: {failed}."
    )


@st.cache_data(show_spinner=False)
def get_yahoo_metadata(yahoo_ticker):
    """Fetch lightweight name/exchange/currency metadata with safe fallbacks."""
    import yfinance as yf

    ticker = str(yahoo_ticker).strip().upper()
    inferred_exchange, inferred_currency = _infer_market_from_ticker(ticker)
    name = ticker
    exchange = inferred_exchange
    currency = inferred_currency

    try:
        obj = yf.Ticker(ticker)
        fast_info = obj.fast_info
        currency = _read_fast_info_value(fast_info, "currency") or currency
        exchange = (
            _read_fast_info_value(fast_info, "exchange")
            or _read_fast_info_value(fast_info, "market")
            or exchange
        )
    except Exception:
        pass

    try:
        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName") or name
        currency = info.get("currency") or currency
        exchange = info.get("exchange") or info.get("fullExchangeName") or exchange
    except Exception:
        pass

    currency = str(currency or inferred_currency).strip()
    if currency.upper() in {"GBP", "GBX"}:
        # Yahoo sometimes reports London securities in GBp/GBX (pence).
        currency = "GBp" if currency.upper() == "GBX" else currency
    else:
        currency = currency.upper()

    return {
        "yahoo_ticker": ticker,
        "stock_name": str(name or ticker).strip(),
        "exchange": str(exchange or inferred_exchange).strip(),
        "currency": currency,
    }


def resolve_yahoo_instrument(symbol, nse_company_lookup=None):
    """Resolve NSE, BSE, US and explicit Yahoo symbols into one canonical record.

    Plain symbols that exist in the official NSE equity reference prefer ``.NS``.
    Plain symbols absent from that reference prefer the unsuffixed Yahoo ticker.
    This distinction is important for symbols such as ``VT``: ``VT`` is a US ETF
    and must not be silently converted into ``VT.NS``. Explicit Yahoo suffixes are
    always tried exactly as entered.
    """
    raw = str(symbol or "").strip().upper()
    if not raw:
        return None

    nse_company_lookup = nse_company_lookup or get_nse_company_lookup()
    base = normalize_portfolio_symbol(raw)

    if base.endswith("-BE"):
        base = base[:-3].rstrip("-").strip()

    explicit = (
        raw.endswith(".NS")
        or raw.endswith(".BO")
        or ("." in raw and not raw.endswith(".NS") and not raw.endswith(".BO"))
        or raw.endswith("=X")
        or raw.startswith("^")
    )

    if explicit:
        candidates = [raw]
    elif base.isdigit():
        # Numeric Indian security codes are overwhelmingly BSE identifiers.
        candidates = [f"{base}.BO", base, f"{base}.NS"]
    elif base in nse_company_lookup:
        # The NSE reference is authoritative for ordinary Indian equity symbols.
        candidates = [f"{base}.NS", base, f"{base}.BO"]
    else:
        # Unknown-to-NSE plain symbols are treated as global Yahoo symbols first.
        # Example: VT -> VT (NYSE Arca / USD), not VT.NS.
        candidates = [base, f"{base}.NS", f"{base}.BO"]

    seen = set()
    for ticker in candidates:
        ticker = ticker.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        if not _download_probe(ticker):
            continue

        metadata = get_yahoo_metadata(ticker)
        display_symbol = normalize_portfolio_symbol(raw)
        if ticker.endswith(".NS"):
            display_symbol = ticker[:-3]
            metadata["stock_name"] = nse_company_lookup.get(
                display_symbol, metadata["stock_name"]
            )
        elif ticker.endswith(".BO"):
            display_symbol = ticker[:-3]
        elif explicit:
            display_symbol = raw
        else:
            display_symbol = base

        metadata["symbol"] = display_symbol
        return metadata

    return None


def resolve_yahoo_tickers(symbols_base):
    """Resolve user-facing holding symbols to Yahoo tickers across markets."""
    lookup = get_nse_company_lookup()
    resolved = {}
    for sym in symbols_base:
        instrument = resolve_yahoo_instrument(sym, lookup)
        if instrument is not None:
            resolved[normalize_portfolio_symbol(sym)] = instrument["yahoo_ticker"]
    return resolved

# =========================================================
# DATABASE / HOLDINGS INPUT
# =========================================================

SCRIPT_DB_PATH = Path(__file__).resolve().with_name("portfolio_holdings.db")
USER_DB_PATH = Path.home() / ".portfolio_rebalancer" / "portfolio_holdings.db"

# Honour an explicit deployment setting first. Otherwise reuse a legacy database
# beside the script when it exists; for new installations use a writable user-data
# directory instead of assuming that the deployed source folder is writable.
DEFAULT_DB_PATH = SCRIPT_DB_PATH if SCRIPT_DB_PATH.exists() else USER_DB_PATH
DB_PATH = Path(os.getenv("PORTFOLIO_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()

# Reserved owner key for the shared universal portfolio (visible/editable by everyone,
# quantity always 0 so it never counts as anyone's real holding).
UNIVERSAL_OWNER = "__universal__"

MASTER_HOLDINGS_DDL = """
CREATE TABLE IF NOT EXISTS master_holdings (
    owner TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    stock_name TEXT NOT NULL,
    yahoo_ticker TEXT,
    exchange TEXT,
    currency TEXT,
    quantity REAL NOT NULL DEFAULT 1,
    average_price REAL,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (owner, symbol)
)
"""

LATEST_ANALYSIS_DDL = """
CREATE TABLE IF NOT EXISTS latest_analysis (
    owner TEXT PRIMARY KEY,
    saved_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
)
"""


def _connect_sqlite():
    """Open SQLite with settings suitable for Streamlit reruns/concurrent sessions."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=30,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _ensure_master_holdings_schema(conn):
    """Create the table and repair older compatible schemas in place."""
    conn.execute(MASTER_HOLDINGS_DDL)
    conn.execute(LATEST_ANALYSIS_DDL)
    now = datetime.now().isoformat(timespec="seconds")

    latest_columns = {
        str(row["name"]).lower()
        for row in conn.execute("PRAGMA table_info(latest_analysis)").fetchall()
    }
    required_latest_columns = {"owner", "saved_at", "payload_json"}
    if not required_latest_columns.issubset(latest_columns):
        legacy_latest_name = (
            "latest_analysis_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        conn.execute(f'ALTER TABLE latest_analysis RENAME TO "{legacy_latest_name}"')
        conn.execute(LATEST_ANALYSIS_DDL)

        legacy_columns = {
            str(row["name"]).lower()
            for row in conn.execute(
                f'PRAGMA table_info("{legacy_latest_name}")'
            ).fetchall()
        }
        if "payload_json" in legacy_columns:
            time_column = (
                "saved_at"
                if "saved_at" in legacy_columns
                else "analyzed_at"
                if "analyzed_at" in legacy_columns
                else None
            )
            if time_column is not None:
                # Pre-multi-user installs had a single shared analysis row. Preserve
                # it under owner='' (an orphaned/legacy bucket) instead of deleting it.
                legacy_row = conn.execute(
                    f'SELECT "{time_column}", payload_json '
                    f'FROM "{legacy_latest_name}" LIMIT 1'
                ).fetchone()
                if legacy_row is not None:
                    conn.execute(
                        """
                        INSERT INTO latest_analysis (owner, saved_at, payload_json)
                        VALUES ('', ?, ?)
                        ON CONFLICT(owner) DO NOTHING
                        """,
                        (str(legacy_row[0]), legacy_row[1]),
                    )

    table_info = conn.execute("PRAGMA table_info(master_holdings)").fetchall()
    columns = {str(row["name"]).lower() for row in table_info}

    if "symbol" not in columns:
        legacy_name = "master_holdings_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        conn.execute(f'ALTER TABLE master_holdings RENAME TO "{legacy_name}"')
        conn.execute(MASTER_HOLDINGS_DDL)
        columns = {
            str(row["name"]).lower()
            for row in conn.execute("PRAGMA table_info(master_holdings)").fetchall()
        }
    elif "owner" not in columns:
        # Pre-multi-user schema used `symbol` as the sole primary key. Rebuild the
        # table with a composite (owner, symbol) key and preserve the old rows under
        # owner='' (an orphaned/legacy bucket) rather than losing them.
        legacy_name = "master_holdings_pre_owner_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        conn.execute(f'ALTER TABLE master_holdings RENAME TO "{legacy_name}"')
        conn.execute(MASTER_HOLDINGS_DDL)

        legacy_columns = {
            str(row["name"]).lower()
            for row in conn.execute(f'PRAGMA table_info("{legacy_name}")').fetchall()
        }
        column_defaults = {
            "stock_name": "symbol",
            "yahoo_ticker": "NULL",
            "exchange": "NULL",
            "currency": "NULL",
            "quantity": "1",
            "average_price": "NULL",
            "added_at": f"'{now}'",
            "updated_at": f"'{now}'",
        }
        select_exprs = [
            column if column in legacy_columns else default
            for column, default in column_defaults.items()
        ]
        conn.execute(
            f"""
            INSERT INTO master_holdings
                (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                 quantity, average_price, added_at, updated_at)
            SELECT '', symbol, {", ".join(select_exprs)}
            FROM "{legacy_name}"
            """
        )
        columns = {
            str(row["name"]).lower()
            for row in conn.execute("PRAGMA table_info(master_holdings)").fetchall()
        }

    migrations = {
        "stock_name": "ALTER TABLE master_holdings ADD COLUMN stock_name TEXT",
        "yahoo_ticker": "ALTER TABLE master_holdings ADD COLUMN yahoo_ticker TEXT",
        "exchange": "ALTER TABLE master_holdings ADD COLUMN exchange TEXT",
        "currency": "ALTER TABLE master_holdings ADD COLUMN currency TEXT",
        "quantity": "ALTER TABLE master_holdings ADD COLUMN quantity REAL DEFAULT 1",
        "average_price": "ALTER TABLE master_holdings ADD COLUMN average_price REAL",
        "added_at": "ALTER TABLE master_holdings ADD COLUMN added_at TEXT",
        "updated_at": "ALTER TABLE master_holdings ADD COLUMN updated_at TEXT",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)

    # Preserve old holdings without guessing their market. Missing metadata is
    # resolved later using the current NSE reference + Yahoo validation. This avoids
    # turning a legacy global symbol such as VT into VT.NS/NSE/INR.
    conn.execute(
        """
        UPDATE master_holdings
        SET stock_name = COALESCE(NULLIF(TRIM(stock_name), ''), symbol),
            yahoo_ticker = NULLIF(TRIM(yahoo_ticker), ''),
            exchange = NULLIF(TRIM(exchange), ''),
            currency = NULLIF(TRIM(currency), ''),
            quantity = CASE WHEN quantity IS NULL OR quantity <= 0 THEN 1 ELSE quantity END,
            added_at = COALESCE(NULLIF(TRIM(added_at), ''), ?),
            updated_at = COALESCE(NULLIF(TRIM(updated_at), ''), ?)
        """,
        (now, now),
    )
    conn.commit()

def get_db_connection():
    """Return a connection only after the required holdings schema is available."""
    conn = _connect_sqlite()
    try:
        _ensure_master_holdings_schema(conn)
        return conn
    except Exception:
        conn.close()
        raise


def init_holdings_db():
    """Initialize SQLite and recover safely from a corrupt/non-SQLite DB file.

    Returns the path of a quarantined file when recovery was required, otherwise
    returns None.
    """
    try:
        with get_db_connection():
            return None
    except sqlite3.DatabaseError as exc:
        message = str(exc).lower()
        corrupt_markers = (
            "file is not a database",
            "database disk image is malformed",
            "file is encrypted",
        )
        if not DB_PATH.exists() or not any(marker in message for marker in corrupt_markers):
            raise

        quarantine_path = DB_PATH.with_name(
            f"{DB_PATH.stem}.corrupt-{datetime.now():%Y%m%d-%H%M%S}{DB_PATH.suffix}"
        )
        DB_PATH.replace(quarantine_path)
        with get_db_connection():
            pass
        return quarantine_path


def normalize_nse_symbol(value):
    """Backward-compatible alias for the now multi-market symbol normalizer."""
    return normalize_portfolio_symbol(value)

def resolve_nse_symbol(symbol, available_symbols, allow_be_fallback=False):
    """Resolve broker-style NSE symbols against canonical NSE symbols."""
    normalized = normalize_portfolio_symbol(symbol)
    if normalized in available_symbols:
        return normalized

    if normalized.endswith("-BE"):
        base_symbol = normalized[:-3].rstrip("-").strip()
        if base_symbol and (base_symbol in available_symbols or allow_be_fallback):
            return base_symbol

    return None

def parse_symbol_input(raw_text):
    """Split user input while preserving explicit Yahoo suffixes such as .NS/.BO/.L."""
    parts = re.split(r"[,;\n]+", raw_text or "")
    symbols = []
    seen = set()

    for part in parts:
        symbol = str(part or "").strip().upper()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)

    return symbols

def get_nse_company_lookup():
    equity_map = load_equity_mapping().copy()
    equity_map["Symbol"] = equity_map["Symbol"].astype(str).str.strip().str.upper()
    equity_map["Company Name"] = equity_map["Company Name"].astype(str).str.strip()
    return dict(zip(equity_map["Symbol"], equity_map["Company Name"]))


def load_master_holdings(owner):
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                symbol AS Symbol,
                stock_name AS "Stock Name",
                yahoo_ticker AS "Yahoo Ticker",
                exchange AS Exchange,
                currency AS Currency,
                quantity AS Quantity,
                average_price AS "Average Price",
                added_at AS "Added At",
                updated_at AS "Updated At"
            FROM master_holdings
            WHERE owner = ?
            ORDER BY symbol
            """,
            conn,
            params=(owner,),
        )
    return df

def repair_master_holdings_metadata(owner):
    """Repair missing or obviously misclassified market metadata in-place.

    The first multi-market build could migrate every legacy plain symbol to
    ``<symbol>.NS / NSE / INR``. That is wrong for global symbols such as VT.
    We use the NSE equity reference as the authority: if a row claims ``.NS``
    but its symbol is not in that reference, re-resolve the plain symbol.

    Returns a list of repaired display symbols. Network failures are tolerated;
    unresolved rows are left unchanged for a later rerun.
    """
    try:
        lookup = get_nse_company_lookup()
    except Exception:
        return []

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT symbol, stock_name, yahoo_ticker, exchange, currency
            FROM master_holdings
            WHERE owner = ?
            """,
            (owner,),
        ).fetchall()

    repaired = []
    now = datetime.now().isoformat(timespec="seconds")

    with get_db_connection() as conn:
        for row in rows:
            symbol = normalize_portfolio_symbol(row["symbol"])
            ticker = str(row["yahoo_ticker"] or "").strip().upper()
            exchange = str(row["exchange"] or "").strip().upper()
            currency = _normalize_currency_code(row["currency"])

            metadata_missing = not ticker or not currency
            looks_like_bad_legacy_nse_guess = (
                symbol not in lookup
                and ticker == f"{symbol}.NS"
                and exchange in {"", "NSE", "NSI"}
                and currency in {"", "INR"}
            )

            if not metadata_missing and not looks_like_bad_legacy_nse_guess:
                continue

            instrument = resolve_yahoo_instrument(symbol, lookup)
            if instrument is None:
                continue

            new_ticker = str(instrument["yahoo_ticker"]).upper()
            new_exchange = str(instrument["exchange"] or "").strip()
            new_currency = _normalize_currency_code(instrument["currency"])
            old_name = str(row["stock_name"] or "").strip()
            new_name = str(instrument["stock_name"] or symbol).strip()
            if old_name and old_name.upper() != symbol.upper() and not looks_like_bad_legacy_nse_guess:
                new_name = old_name

            if (
                ticker == new_ticker
                and exchange == new_exchange.upper()
                and currency == new_currency
                and old_name == new_name
            ):
                continue

            conn.execute(
                """
                UPDATE master_holdings
                SET stock_name = ?, yahoo_ticker = ?, exchange = ?, currency = ?, updated_at = ?
                WHERE owner = ? AND symbol = ?
                """,
                (new_name, new_ticker, new_exchange, new_currency, now, owner, row["symbol"]),
            )
            repaired.append(symbol)

        conn.commit()

    return repaired


def holdings_backup_bytes(owner):
    """Export the complete holdings master table as a UTF-8 CSV backup."""
    holdings = load_master_holdings(owner)
    return holdings.to_csv(index=False).encode("utf-8-sig")


# Broker holdings statement column names vary a lot, so accept common aliases.
BROKER_HOLDINGS_COLUMN_ALIASES = {
    "stock name": "Stock Name",
    "instrument": "Stock Name",
    "symbol": "Stock Name",
    "isin": "ISIN",
    "quantity": "Quantity",
    "quantity available": "Quantity",
    "qty": "Quantity",
    "average buy price": "Average Buy Price",
    "average price": "Average Buy Price",
    "avg. cost": "Average Buy Price",
    "avg cost": "Average Buy Price",
    "buy value": "Buy Value",
    "closing price": "Closing Price",
    "previous closing price": "Closing Price",
    "closing value": "Closing Value",
    "unrealised p&l": "Unrealised P&L",
    "unrealized p&l": "Unrealised P&L",
}


def _detect_broker_holdings_header_row(raw_df, max_scan_rows=25):
    """Locate the header row by scanning for ISIN + Quantity column labels.

    This tolerates broker exports that don't always place headers on row 11.
    """
    required_tokens = {"isin", "quantity"}
    for row_idx in range(min(max_scan_rows, len(raw_df))):
        row_values = {
            str(v).strip().lower() for v in raw_df.iloc[row_idx].tolist() if pd.notna(v)
        }
        if required_tokens.issubset(row_values):
            return row_idx
    return None


def _read_broker_holdings_excel(uploaded_file):
    """Parse a broker holdings statement (.xlsx) with Stock Name/ISIN/Quantity/
    Average Buy Price/Buy Value/Closing Price/Closing Value/Unrealised P&L columns.

    The header row is auto-detected; it defaults to row 11 (index 10) when it
    cannot be located, matching the layout described by the user.
    """
    if uploaded_file is None:
        raise ValueError("Choose a broker holdings Excel file first.")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected broker holdings file is empty.")

    try:
        preview_df = pd.read_excel(io.BytesIO(raw), header=None, nrows=25)
    except ImportError as exc:
        raise ValueError(
            "Reading .xlsx files requires the 'openpyxl' package. Install it with "
            "`pip install openpyxl` and retry."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read the Excel file: {exc}") from exc

    header_row = _detect_broker_holdings_header_row(preview_df)
    if header_row is None:
        header_row = 10

    try:
        df = pd.read_excel(io.BytesIO(raw), header=header_row)
    except Exception as exc:
        raise ValueError(
            f"Could not read the Excel file with header row {header_row + 1}: {exc}"
        ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    renamed = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in BROKER_HOLDINGS_COLUMN_ALIASES:
            renamed[col] = BROKER_HOLDINGS_COLUMN_ALIASES[key]
    df = df.rename(columns=renamed)

    required = ["ISIN", "Quantity", "Average Buy Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The broker holdings file is missing required columns: " + ", ".join(missing)
        )

    if "Stock Name" not in df.columns:
        df["Stock Name"] = df["ISIN"]

    df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()
    df["Stock Name"] = df["Stock Name"].fillna("").astype(str).str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Buy Price"] = pd.to_numeric(df["Average Buy Price"], errors="coerce")

    df = df[df["ISIN"].str.match(r"^IN[A-Z0-9]{10}$", na=False)].copy()
    df = df[df["Quantity"].notna() & (df["Quantity"] > 0)]
    df = df[df["Average Buy Price"].notna() & (df["Average Buy Price"] > 0)]

    if df.empty:
        raise ValueError(
            "No valid holdings rows (with ISIN, Quantity, Average Buy Price) were found."
        )

    def _combine(group):
        total_qty = group["Quantity"].sum()
        weighted_price = (group["Quantity"] * group["Average Buy Price"]).sum() / total_qty
        return pd.Series({
            "Stock Name": group["Stock Name"].iloc[0],
            "Quantity": total_qty,
            "Average Buy Price": weighted_price,
        })

    df = df.groupby("ISIN", as_index=False).apply(_combine).reset_index(drop=True)
    return df[["ISIN", "Stock Name", "Quantity", "Average Buy Price"]]


def import_broker_holdings_excel(uploaded_file, owner, mode="merge"):
    """Resolve broker holdings by ISIN and upsert them into master_holdings.

    Returns (imported_row_count, unresolved_isins).
    """
    cleaned = _read_broker_holdings_excel(uploaded_file)
    normalized_mode = str(mode or "merge").strip().lower()
    if normalized_mode not in {"replace", "merge"}:
        raise ValueError("Import mode must be either 'replace' or 'merge'.")

    equity_map = load_equity_mapping().copy()
    equity_map["ISIN"] = equity_map["ISIN"].astype(str).str.strip().str.upper()
    isin_to_symbol = dict(
        zip(equity_map["ISIN"], equity_map["Symbol"].astype(str).str.strip().str.upper())
    )
    nse_lookup = get_nse_company_lookup()

    resolved_rows = []
    unresolved_isins = []

    for _, row in cleaned.iterrows():
        isin = row["ISIN"]
        symbol_guess = isin_to_symbol.get(isin)
        instrument = resolve_yahoo_instrument(symbol_guess, nse_lookup) if symbol_guess else None
        if instrument is None:
            unresolved_isins.append(f"{isin} ({row['Stock Name']})")
            continue

        resolved_rows.append({
            "Symbol": instrument["symbol"],
            "Stock Name": instrument["stock_name"],
            "Yahoo Ticker": instrument["yahoo_ticker"],
            "Exchange": instrument["exchange"],
            "Currency": _normalize_currency_code(instrument["currency"]),
            "Quantity": float(row["Quantity"]),
            "Average Price": float(row["Average Buy Price"]),
        })

    if not resolved_rows:
        raise ValueError(
            "None of the ISINs in the broker holdings file could be resolved. Unresolved: "
            + ", ".join(unresolved_isins[:10])
        )

    resolved_df = pd.DataFrame(resolved_rows)
    if resolved_df["Symbol"].duplicated().any():
        def _combine_symbol(group):
            total_qty = group["Quantity"].sum()
            weighted_price = (group["Quantity"] * group["Average Price"]).sum() / total_qty
            first = group.iloc[0]
            return pd.Series({
                "Stock Name": first["Stock Name"],
                "Yahoo Ticker": first["Yahoo Ticker"],
                "Exchange": first["Exchange"],
                "Currency": first["Currency"],
                "Quantity": total_qty,
                "Average Price": weighted_price,
            })

        resolved_df = (
            resolved_df.groupby("Symbol", as_index=False).apply(_combine_symbol).reset_index(drop=True)
        )

    now = datetime.now().isoformat(timespec="seconds")
    records = [
        (
            owner, row["Symbol"], row["Stock Name"], row["Yahoo Ticker"], row["Exchange"],
            row["Currency"], float(row["Quantity"]), float(row["Average Price"]), now, now,
        )
        for _, row in resolved_df.iterrows()
    ]

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            if normalized_mode == "replace":
                conn.execute("DELETE FROM master_holdings WHERE owner = ?", (owner,))

            conn.executemany(
                """
                INSERT INTO master_holdings
                    (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                     quantity, average_price, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner, symbol) DO UPDATE SET
                    stock_name = excluded.stock_name,
                    yahoo_ticker = excluded.yahoo_ticker,
                    exchange = excluded.exchange,
                    currency = excluded.currency,
                    quantity = excluded.quantity,
                    average_price = excluded.average_price,
                    updated_at = excluded.updated_at
                """,
                records,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return len(records), unresolved_isins


def _read_holdings_backup(uploaded_file):
    """Read and validate a holdings CSV uploaded through Streamlit."""
    if uploaded_file is None:
        raise ValueError("Choose a holdings backup CSV file first.")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected holdings backup is empty.")

    try:
        backup_df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read the holdings CSV backup: {exc}") from exc

    aliases = {
        "symbol": "Symbol",
        "stock name": "Stock Name",
        "stock_name": "Stock Name",
        "yahoo ticker": "Yahoo Ticker",
        "yahoo_ticker": "Yahoo Ticker",
        "exchange": "Exchange",
        "currency": "Currency",
        "quantity": "Quantity",
        "average price": "Average Price",
        "average_price": "Average Price",
        "added at": "Added At",
        "added_at": "Added At",
        "updated at": "Updated At",
        "updated_at": "Updated At",
    }
    renamed = {}
    for column in backup_df.columns:
        normalized = str(column).strip().lower()
        if normalized in aliases:
            renamed[column] = aliases[normalized]
    backup_df = backup_df.rename(columns=renamed)

    required = ["Symbol", "Quantity", "Average Price"]
    missing = [column for column in required if column not in backup_df.columns]
    if missing:
        raise ValueError(
            "The holdings backup is missing required columns: " + ", ".join(missing)
        )

    for column, default in (
        ("Stock Name", None),
        ("Yahoo Ticker", None),
        ("Exchange", None),
        ("Currency", None),
        ("Added At", None),
        ("Updated At", None),
    ):
        if column not in backup_df.columns:
            backup_df[column] = default

    cleaned = backup_df[
        [
            "Symbol", "Stock Name", "Yahoo Ticker", "Exchange", "Currency",
            "Quantity", "Average Price", "Added At", "Updated At"
        ]
    ].copy()
    cleaned["Symbol"] = cleaned["Symbol"].map(normalize_portfolio_symbol)
    cleaned["Stock Name"] = cleaned["Stock Name"].fillna("").astype(str).str.strip()
    cleaned["Yahoo Ticker"] = cleaned["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    cleaned["Exchange"] = cleaned["Exchange"].fillna("").astype(str).str.strip()
    cleaned["Currency"] = cleaned["Currency"].fillna("").astype(str).str.strip()
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["Average Price"] = pd.to_numeric(cleaned["Average Price"], errors="coerce")

    if cleaned.empty:
        raise ValueError("The holdings backup does not contain any rows.")
    if (cleaned["Symbol"] == "").any():
        raise ValueError("Every backup row must contain a valid Symbol.")
    if cleaned["Symbol"].duplicated().any():
        duplicates = sorted(
            cleaned.loc[cleaned["Symbol"].duplicated(keep=False), "Symbol"].unique()
        )
        raise ValueError(
            "The holdings backup contains duplicate symbols: " + ", ".join(duplicates)
        )
    if cleaned["Quantity"].isna().any() or (cleaned["Quantity"] <= 0).any():
        raise ValueError("Quantity must be greater than zero for every restored holding.")
    invalid_prices = cleaned["Average Price"].notna() & (cleaned["Average Price"] <= 0)
    if invalid_prices.any():
        raise ValueError(
            "Average Price must be blank or greater than zero for every restored holding."
        )

    lookup = get_nse_company_lookup()
    resolved_rows = []
    for _, row in cleaned.iterrows():
        yahoo_ticker = row["Yahoo Ticker"]
        exchange = row["Exchange"]
        currency = row["Currency"]
        stock_name = row["Stock Name"]

        symbol = row["Symbol"]
        suspicious_nse_guess = (
            symbol not in lookup
            and str(yahoo_ticker or "").upper() == f"{symbol}.NS"
            and str(exchange or "").upper() in {"", "NSE", "NSI"}
            and _normalize_currency_code(currency) in {"", "INR"}
        )

        if not yahoo_ticker or not currency or suspicious_nse_guess:
            instrument = resolve_yahoo_instrument(symbol, lookup)
            if instrument is not None:
                if suspicious_nse_guess:
                    yahoo_ticker = instrument["yahoo_ticker"]
                    exchange = instrument["exchange"]
                    currency = instrument["currency"]
                    stock_name = instrument["stock_name"]
                else:
                    yahoo_ticker = yahoo_ticker or instrument["yahoo_ticker"]
                    exchange = exchange or instrument["exchange"]
                    currency = currency or instrument["currency"]
                    stock_name = stock_name or instrument["stock_name"]
            # If Yahoo is temporarily unavailable, keep metadata blank rather than
            # manufacturing an NSE classification that may be financially wrong.

        resolved_rows.append((
            stock_name or symbol,
            str(yahoo_ticker or "").strip().upper(),
            str(exchange or "").strip(),
            _normalize_currency_code(currency),
        ))

    cleaned[["Stock Name", "Yahoo Ticker", "Exchange", "Currency"]] = pd.DataFrame(
        resolved_rows, index=cleaned.index
    )
    return cleaned

def restore_holdings_backup(uploaded_file, owner, mode="merge"):
    """Restore holdings from CSV using replace or merge/update semantics."""
    cleaned = _read_holdings_backup(uploaded_file)
    normalized_mode = str(mode or "merge").strip().lower()
    if normalized_mode not in {"replace", "merge"}:
        raise ValueError("Restore mode must be either 'replace' or 'merge'.")

    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for _, row in cleaned.iterrows():
        added_at = str(row["Added At"]).strip() if pd.notna(row["Added At"]) else now
        updated_at = str(row["Updated At"]).strip() if pd.notna(row["Updated At"]) else now
        records.append(
            (
                owner, row["Symbol"], row["Stock Name"], row["Yahoo Ticker"],
                row["Exchange"], row["Currency"], float(row["Quantity"]),
                float(row["Average Price"]) if pd.notna(row["Average Price"]) else None,
                added_at or now, updated_at or now,
            )
        )

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            if normalized_mode == "replace":
                conn.execute("DELETE FROM master_holdings WHERE owner = ?", (owner,))

            conn.executemany(
                """
                INSERT INTO master_holdings
                    (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                     quantity, average_price, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner, symbol) DO UPDATE SET
                    stock_name = excluded.stock_name,
                    yahoo_ticker = excluded.yahoo_ticker,
                    exchange = excluded.exchange,
                    currency = excluded.currency,
                    quantity = excluded.quantity,
                    average_price = excluded.average_price,
                    added_at = excluded.added_at,
                    updated_at = excluded.updated_at
                """,
                records,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return len(records)

def get_unique_holdings_count(owner):
    """Return the current distinct holding count directly from SQLite."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT UPPER(TRIM(symbol))) AS unique_count
            FROM master_holdings
            WHERE owner = ?
              AND symbol IS NOT NULL
              AND TRIM(symbol) <> ''
            """,
            (owner,),
        ).fetchone()
    return int(row["unique_count"] or 0)


def render_live_holdings_banner(placeholder, owner):
    """Render the live count without taking down the whole app on a DB error."""
    try:
        unique_count = get_unique_holdings_count(owner)
    except sqlite3.Error as exc:
        placeholder.error(
            "Holdings database is unavailable. "
            f"SQLite reported: {exc}. Active database path: {DB_PATH}"
        )
        return 0

    placeholder.markdown(
        f"""
        <div style="
            border: 2px solid #22c55e;
            border-radius: 14px;
            padding: 16px 20px;
            margin: 10px 0 18px 0;
            background: rgba(34, 197, 94, 0.12);
            text-align: center;
        ">
            <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; color: #16a34a;">
                ● LIVE DATABASE STATUS
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; margin-top: 4px;">
                Current unique holdings count: {unique_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return unique_count



def make_json_safe(value):
    """Recursively convert pandas, NumPy, and datetime values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def save_latest_analysis(payload, owner):
    """Persist the latest successful analysis as one row per owner."""
    if not isinstance(payload, dict):
        raise ValueError("Analysis payload must be a dictionary.")

    safe_payload = make_json_safe(payload)
    saved_at = str(
        safe_payload.get("saved_at")
        or datetime.now().isoformat(timespec="seconds")
    )
    safe_payload["saved_at"] = saved_at
    payload_json = json.dumps(
        safe_payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    with get_db_connection() as conn:
        conn.execute(LATEST_ANALYSIS_DDL)
        conn.execute(
            """
            INSERT INTO latest_analysis (owner, saved_at, payload_json)
            VALUES (?, ?, ?)
            ON CONFLICT(owner) DO UPDATE SET
                saved_at = excluded.saved_at,
                payload_json = excluded.payload_json
            """,
            (owner, saved_at, payload_json),
        )
        conn.commit()

    return safe_payload


def load_latest_analysis(owner):
    """Load the latest saved analysis without preventing the app from opening."""
    try:
        with get_db_connection() as conn:
            conn.execute(LATEST_ANALYSIS_DDL)
            row = conn.execute(
                "SELECT payload_json FROM latest_analysis WHERE owner = ?", (owner,)
            ).fetchone()
    except sqlite3.DatabaseError:
        return None

    if row is None:
        return None

    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def latest_analysis_backup_bytes(owner):
    """Return the latest saved analysis as UTF-8 JSON bytes, when available."""
    payload = load_latest_analysis(owner)
    if payload is None:
        return None

    return json.dumps(
        make_json_safe(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def restore_latest_analysis_backup(uploaded_file, owner):
    """Validate and restore a previously exported analysis-result JSON backup."""
    if uploaded_file is None:
        raise ValueError("Choose an analysis backup JSON file first.")

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("The selected analysis backup is empty.")

    try:
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Could not read the JSON backup: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("The analysis backup must contain one JSON object.")

    required_keys = {
        "saved_at",
        "holdings_analyzed",
        "total_invested",
        "current_stats",
        "optimal_stats",
        "rebalancing_plan",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise ValueError(
            "This is not a complete analysis backup. Missing: "
            + ", ".join(missing)
        )

    return save_latest_analysis(payload, owner)


def _format_saved_metric(metric_name, value):
    """Format a saved statistic without failing on missing or malformed values."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if metric_name == "Sharpe Ratio":
        return f"{numeric:.2f}"
    return f"{numeric:.2%}"


def render_saved_analysis(placeholder, owner):
    """Display the latest saved analysis immediately below the live banner."""
    placeholder.empty()

    with placeholder.container():
        payload = load_latest_analysis(owner)

        if payload is None:
            st.info(
                "No saved analysis yet. Run analysis once or upload an analysis backup."
            )
            return

        saved_at = str(payload.get("saved_at") or "Unknown")
        holdings_analyzed = int(payload.get("holdings_analyzed") or 0)
        total_invested = float(payload.get("total_invested") or 0.0)
        rebalancing_plan = payload.get("rebalancing_plan") or []
        executable_trade_count = int(
            payload.get("executable_trade_count")
            if payload.get("executable_trade_count") is not None
            else len(rebalancing_plan)
        )

        st.subheader("💾 Last Saved Analysis")
        st.caption(f"Saved at: {saved_at}")

        # Compact information-only summary of the market data used for the
        # saved analysis. This does not expose or restore analysis settings.
        history = payload.get("history") or {}
        valid_start_raw = history.get("valid_start")
        valid_end_raw = history.get("valid_end")
        trading_days = history.get("log_return_rows")
        assets_in_analysis = history.get("log_return_columns")

        valid_start = pd.to_datetime(valid_start_raw, errors="coerce")
        valid_end = pd.to_datetime(valid_end_raw, errors="coerce")

        if pd.notna(valid_start) and pd.notna(valid_end):
            period_text = (
                f"{valid_start.strftime('%d %b %Y')} → "
                f"{valid_end.strftime('%d %b %Y')}"
            )
            calendar_days = int((valid_end.normalize() - valid_start.normalize()).days) + 1

            analysis_parts = [f"**Analysis period:** {period_text}"]
            if trading_days is not None:
                analysis_parts.append(f"**Trading days analysed:** {int(trading_days):,}")
            else:
                analysis_parts.append(f"**Calendar days covered:** {calendar_days:,}")
            if assets_in_analysis is not None:
                analysis_parts.append(f"**Assets in return matrix:** {int(assets_in_analysis):,}")

            st.info("  |  ".join(analysis_parts))
        elif trading_days is not None or assets_in_analysis is not None:
            analysis_parts = []
            if trading_days is not None:
                analysis_parts.append(f"**Trading days analysed:** {int(trading_days):,}")
            if assets_in_analysis is not None:
                analysis_parts.append(f"**Assets in return matrix:** {int(assets_in_analysis):,}")
            st.info("  |  ".join(analysis_parts))

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Holdings analysed", holdings_analyzed)
        summary_col2.metric("Total invested", f"₹{total_invested:,.2f}")
        summary_col3.metric("Executable trades", executable_trade_count)

        backup_json = json.dumps(
            make_json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        backup_date = saved_at[:10] if saved_at and saved_at != "Unknown" else "latest"

        st.download_button(
            "Download complete analysis backup JSON",
            data=backup_json,
            file_name=f"portfolio_analysis_backup_{backup_date}.json",
            mime="application/json",
            width="stretch",
            key="download_saved_analysis_main_" + saved_at,
        )

        current_stats = payload.get("current_stats") or {}
        optimal_stats = payload.get("optimal_stats") or {}

        if current_stats or optimal_stats:
            with st.expander("Saved portfolio statistics", expanded=False):
                stats_col1, stats_col2 = st.columns(2)

                with stats_col1:
                    st.markdown("**Current Portfolio**")
                    if current_stats:
                        current_saved_df = pd.DataFrame(
                            {
                                "Metric": list(current_stats.keys()),
                                "Value": [
                                    _format_saved_metric(metric, current_stats[metric])
                                    for metric in current_stats
                                ],
                            }
                        )
                        st.dataframe(
                            current_saved_df,
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.caption("No current-portfolio statistics were saved.")

                with stats_col2:
                    st.markdown("**Optimized Portfolio**")
                    if optimal_stats:
                        optimal_saved_df = pd.DataFrame(
                            {
                                "Metric": list(optimal_stats.keys()),
                                "Value": [
                                    _format_saved_metric(metric, optimal_stats[metric])
                                    for metric in optimal_stats
                                ],
                            }
                        )
                        st.dataframe(
                            optimal_saved_df,
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.caption("No optimized-portfolio statistics were saved.")

        with st.expander("Saved rebalancing plan", expanded=True):
            if rebalancing_plan:
                saved_rebal_df = pd.DataFrame(rebalancing_plan)
                saved_rebal_df = _sort_rebalance_df_for_priority(saved_rebal_df)

                required_style_columns = {
                    "Current Weight",
                    "Optimal Weight",
                    "Action",
                    "Change",
                    "Quantity",
                    "Executable Quantity",
                    "Executable Value",
                }
                if required_style_columns.issubset(saved_rebal_df.columns):
                    st.dataframe(
                        style_rebalance_df(saved_rebal_df),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        saved_rebal_df,
                        width="stretch",
                        hide_index=True,
                    )
            else:
                st.success("The saved analysis had no executable trades.")


def _normalize_currency_code(currency):
    value = str(currency or "").strip()
    if not value:
        return ""
    if value.upper() in {"GBX", "GBPENCE", "PENCE"} or value == "GBp":
        return "GBp"
    return value.upper()


def _currency_major_and_unit_factor(currency):
    currency = _normalize_currency_code(currency)
    if currency == "GBp":
        return "GBP", 0.01
    return currency, 1.0


@st.cache_data(show_spinner=False)
def _latest_fx_rate_to_inr(currency):
    """Return INR per one quoted currency unit (including pence handling)."""
    major, unit_factor = _currency_major_and_unit_factor(currency)
    if major == "INR":
        return 1.0

    def last_price(ticker):
        data = _yf_download_quiet(
            ticker,
            period="15d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )["Close"]
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return None
            series = pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna()
        else:
            series = pd.to_numeric(data, errors="coerce").dropna()
        return float(series.iloc[-1]) if not series.empty else None

    direct = last_price(f"{major}INR=X")
    if direct is not None and np.isfinite(direct) and direct > 0:
        return float(direct) * unit_factor

    if major != "USD":
        to_usd = last_price(f"{major}USD=X")
        usd_inr = last_price("USDINR=X")
        if (
            to_usd is not None and usd_inr is not None
            and np.isfinite(to_usd) and np.isfinite(usd_inr)
            and to_usd > 0 and usd_inr > 0
        ):
            return float(to_usd) * float(usd_inr) * unit_factor

    raise ValueError(f"Could not obtain an FX rate from {currency} to INR.")


def get_fx_to_inr_map(currencies):
    result = {}
    for currency in sorted({_normalize_currency_code(c) for c in currencies if str(c or '').strip()}):
        result[currency] = _latest_fx_rate_to_inr(currency)
    return result


@st.cache_data(show_spinner=False)
def download_fx_history_to_inr(currency, start_date, end_date):
    """Return a daily series of INR per one quoted currency unit."""
    major, unit_factor = _currency_major_and_unit_factor(currency)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + timedelta(days=7)

    if major == "INR":
        idx = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
        return pd.Series(1.0, index=idx, name="INR")

    def close_series(ticker):
        try:
            data = _yf_download_quiet(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                threads=False,
            )["Close"]
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    return pd.Series(dtype=float)
                series = data.iloc[:, 0]
            else:
                series = data
            return pd.to_numeric(series, errors="coerce").dropna().sort_index()
        except Exception:
            return pd.Series(dtype=float)

    direct = close_series(f"{major}INR=X")
    if not direct.empty:
        return direct * unit_factor

    if major != "USD":
        to_usd = close_series(f"{major}USD=X")
        usd_inr = close_series("USDINR=X")
        if not to_usd.empty and not usd_inr.empty:
            joined = pd.concat([to_usd.rename("to_usd"), usd_inr.rename("usd_inr")], axis=1)
            joined = joined.sort_index().ffill().dropna()
            if not joined.empty:
                return joined["to_usd"] * joined["usd_inr"] * unit_factor

    raise ValueError(f"No usable historical FX series was found for {currency}/INR.")


def convert_price_history_to_inr(prices, ticker_currency_pairs):
    """Convert every asset price series to INR before calculating returns."""
    if prices.empty:
        return prices

    currency_map = {str(t): _normalize_currency_code(c) for t, c in ticker_currency_pairs}
    converted = prices.copy().astype(float)

    for ticker in converted.columns:
        currency = currency_map.get(str(ticker), "INR")
        if currency == "INR":
            continue
        fx = download_fx_history_to_inr(currency, converted.index.min(), converted.index.max())
        fx = fx.reindex(converted.index).ffill().bfill()
        if fx.isna().all():
            raise ValueError(f"FX history for {currency}/INR could not be aligned with {ticker}.")
        converted[ticker] = converted[ticker] * fx

    return converted


def add_symbols_to_master(symbols, owner):
    if not symbols:
        return [], [], [], []

    lookup = get_nse_company_lookup()
    instruments = []
    invalid_symbols = []
    seen_symbols = set()

    for entered_symbol in symbols:
        instrument = resolve_yahoo_instrument(entered_symbol, lookup)
        if instrument is None:
            invalid_symbols.append(entered_symbol)
            continue
        symbol = instrument["symbol"]
        if symbol not in seen_symbols:
            instruments.append(instrument)
            seen_symbols.add(symbol)

    valid_symbols = [item["symbol"] for item in instruments]
    with get_db_connection() as conn:
        existing = {
            row["symbol"]
            for row in conn.execute(
                "SELECT symbol FROM master_holdings WHERE owner = ? AND symbol IN ({})".format(
                    ",".join("?" for _ in valid_symbols)
                ),
                [owner, *valid_symbols],
            ).fetchall()
        } if valid_symbols else set()

    duplicates = [s for s in valid_symbols if s in existing]
    new_instruments = [item for item in instruments if item["symbol"] not in existing]

    ticker_price_map = {}
    if new_instruments:
        try:
            ticker_price_map = get_latest_price_map(
                tuple(item["yahoo_ticker"] for item in new_instruments)
            )
        except Exception:
            ticker_price_map = {}

    now = datetime.now().isoformat(timespec="seconds")
    added = []
    missing_initial_price = []

    with get_db_connection() as conn:
        for item in new_instruments:
            ticker = item["yahoo_ticker"]
            initial_price = ticker_price_map.get(ticker)
            if initial_price is None or not np.isfinite(initial_price) or initial_price <= 0:
                initial_price = None
                missing_initial_price.append(item["symbol"])

            conn.execute(
                """
                INSERT INTO master_holdings
                    (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                     quantity, average_price, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner, item["symbol"], item["stock_name"], ticker,
                    item["exchange"], _normalize_currency_code(item["currency"]),
                    1.0, initial_price, now, now,
                ),
            )
            added.append(item["symbol"])
        conn.commit()

    return added, duplicates, invalid_symbols, missing_initial_price


def add_symbols_to_universal(symbols):
    """Add symbols to the shared universal portfolio (quantity fixed at 0).

    This list is visible/editable by every user and never counts toward anyone's
    real holdings; it exists purely as a shared reference/watchlist that any user
    can copy into their own personal holdings.
    """
    if not symbols:
        return [], [], []

    lookup = get_nse_company_lookup()
    instruments = []
    invalid_symbols = []
    seen_symbols = set()

    for entered_symbol in symbols:
        instrument = resolve_yahoo_instrument(entered_symbol, lookup)
        if instrument is None:
            invalid_symbols.append(entered_symbol)
            continue
        symbol = instrument["symbol"]
        if symbol not in seen_symbols:
            instruments.append(instrument)
            seen_symbols.add(symbol)

    valid_symbols = [item["symbol"] for item in instruments]
    with get_db_connection() as conn:
        existing = {
            row["symbol"]
            for row in conn.execute(
                "SELECT symbol FROM master_holdings WHERE owner = ? AND symbol IN ({})".format(
                    ",".join("?" for _ in valid_symbols)
                ),
                [UNIVERSAL_OWNER, *valid_symbols],
            ).fetchall()
        } if valid_symbols else set()

    duplicates = [s for s in valid_symbols if s in existing]
    new_instruments = [item for item in instruments if item["symbol"] not in existing]

    now = datetime.now().isoformat(timespec="seconds")
    added = []

    with get_db_connection() as conn:
        for item in new_instruments:
            conn.execute(
                """
                INSERT INTO master_holdings
                    (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                     quantity, average_price, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (
                    UNIVERSAL_OWNER, item["symbol"], item["stock_name"], item["yahoo_ticker"],
                    item["exchange"], _normalize_currency_code(item["currency"]), now, now,
                ),
            )
            added.append(item["symbol"])
        conn.commit()

    return added, duplicates, invalid_symbols


def remove_symbols_from_master(symbols, owner):
    if not symbols:
        return [], []

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT symbol, yahoo_ticker FROM master_holdings WHERE owner = ?", (owner,)
        ).fetchall()

        by_symbol = {str(row["symbol"]).upper(): row["symbol"] for row in rows}
        by_ticker = {
            str(row["yahoo_ticker"] or "").upper(): row["symbol"]
            for row in rows
            if str(row["yahoo_ticker"] or "").strip()
        }

        resolved_to_remove = []
        missing = []
        seen_resolved = set()

        for entered_symbol in symbols:
            raw = str(entered_symbol or "").strip().upper()
            base = normalize_portfolio_symbol(raw)
            resolved_symbol = (
                by_ticker.get(raw)
                or by_symbol.get(raw)
                or by_symbol.get(base)
            )

            if resolved_symbol is None:
                missing.append(raw)
            elif resolved_symbol not in seen_resolved:
                resolved_to_remove.append(resolved_symbol)
                seen_resolved.add(resolved_symbol)

        if resolved_to_remove:
            conn.execute(
                "DELETE FROM master_holdings WHERE owner = ? AND symbol IN ({})".format(
                    ",".join("?" for _ in resolved_to_remove)
                ),
                [owner, *resolved_to_remove],
            )
            conn.commit()

    return resolved_to_remove, missing


def save_holding_values(edited_df, owner):
    """Persist every editable holding atomically and verify that each row exists."""
    required = ["Symbol", "Quantity", "Average Price"]
    missing_columns = [c for c in required if c not in edited_df.columns]
    if missing_columns:
        raise ValueError(f"Missing editable holding columns: {missing_columns}")

    cleaned = edited_df.copy()
    cleaned["Symbol"] = cleaned["Symbol"].map(normalize_nse_symbol)
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["Average Price"] = pd.to_numeric(cleaned["Average Price"], errors="coerce")

    if cleaned.empty:
        raise ValueError("There are no holdings to save.")
    if (cleaned["Symbol"] == "").any():
        raise ValueError("Every holding must have a valid symbol.")
    if cleaned["Symbol"].duplicated().any():
        raise ValueError("Duplicate symbols are not allowed in the master holdings table.")
    if cleaned["Quantity"].isna().any() or (cleaned["Quantity"] <= 0).any():
        raise ValueError("Quantity must be greater than zero for every holding.")
    if cleaned["Average Price"].isna().any() or (cleaned["Average Price"] <= 0).any():
        raise ValueError("Average Price must be greater than zero for every holding.")

    symbols = cleaned["Symbol"].tolist()
    placeholders = ",".join("?" for _ in symbols)
    now = datetime.now().isoformat(timespec="seconds")

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = {
                row["symbol"]
                for row in conn.execute(
                    f"SELECT symbol FROM master_holdings WHERE owner = ? AND symbol IN ({placeholders})",
                    [owner, *symbols],
                ).fetchall()
            }
            missing_symbols = sorted(set(symbols) - existing)
            if missing_symbols:
                raise ValueError(
                    "These holdings no longer exist in SQLite: "
                    + ", ".join(missing_symbols)
                )

            updated_count = 0
            for _, row in cleaned.iterrows():
                cursor = conn.execute(
                    """
                    UPDATE master_holdings
                    SET quantity = ?, average_price = ?, updated_at = ?
                    WHERE owner = ? AND symbol = ?
                    """,
                    (
                        float(row["Quantity"]),
                        float(row["Average Price"]),
                        now,
                        owner,
                        row["Symbol"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"SQLite did not update holding {row['Symbol']} exactly once."
                    )
                updated_count += cursor.rowcount

            if updated_count != len(cleaned):
                raise RuntimeError(
                    f"Expected to update {len(cleaned)} rows, but updated {updated_count}."
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return updated_count


def build_current_allocation_from_db(owner):
    df = load_master_holdings(owner)
    if df.empty:
        return pd.DataFrame(), []

    def normalize_frame(frame):
        frame = frame.copy()
        frame["Symbol"] = frame["Symbol"].map(normalize_portfolio_symbol)
        frame["Yahoo Ticker"] = frame["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
        frame["Currency"] = frame["Currency"].fillna("").map(_normalize_currency_code)
        frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
        frame["Average Price"] = pd.to_numeric(frame["Average Price"], errors="coerce")
        return frame

    df = normalize_frame(df)

    # Repair metadata lazily for rows originating from old databases/backups,
    # including the v1 multi-market bug that could classify VT as VT.NS/NSE/INR.
    lookup = get_nse_company_lookup()
    suspicious_nse_guess = (
        ~df["Symbol"].isin(set(lookup))
        & (df["Yahoo Ticker"] == (df["Symbol"] + ".NS"))
        & df["Exchange"].fillna("").astype(str).str.upper().isin(["", "NSE", "NSI"])
        & df["Currency"].isin(["", "INR"])
    )
    rows_needing_metadata = df[
        (df["Yahoo Ticker"] == "")
        | (df["Currency"] == "")
        | suspicious_nse_guess
    ]["Symbol"].tolist()
    if rows_needing_metadata:
        with get_db_connection() as conn:
            for symbol in rows_needing_metadata:
                instrument = resolve_yahoo_instrument(symbol, lookup)
                if instrument is None:
                    continue
                conn.execute(
                    """
                    UPDATE master_holdings
                    SET stock_name = ?, yahoo_ticker = ?, exchange = ?, currency = ?, updated_at = ?
                    WHERE owner = ? AND symbol = ?
                    """,
                    (
                        instrument["stock_name"], instrument["yahoo_ticker"],
                        instrument["exchange"], _normalize_currency_code(instrument["currency"]),
                        datetime.now().isoformat(timespec="seconds"), owner, symbol,
                    ),
                )
            conn.commit()
        df = normalize_frame(load_master_holdings(owner))

    # Fetch current native-market prices for all holdings. This is used for current
    # portfolio weights; Average Price remains the editable cost-basis field.
    all_tickers = tuple(t for t in df["Yahoo Ticker"].tolist() if t)
    latest_price_map = get_latest_price_map(all_tickers) if all_tickers else {}

    needs_price = df[
        df["Average Price"].isna()
        | ~np.isfinite(df["Average Price"])
        | (df["Average Price"] <= 0)
    ].copy()
    if not needs_price.empty:
        now = datetime.now().isoformat(timespec="seconds")
        with get_db_connection() as conn:
            for _, row in needs_price.iterrows():
                price = latest_price_map.get(row["Yahoo Ticker"])
                if price is not None and np.isfinite(price) and price > 0:
                    conn.execute(
                        """
                        UPDATE master_holdings
                        SET average_price = ?, updated_at = ?
                        WHERE owner = ? AND symbol = ?
                        """,
                        (float(price), now, owner, row["Symbol"]),
                    )
            conn.commit()
        df = normalize_frame(load_master_holdings(owner))

    df["Latest Price"] = df["Yahoo Ticker"].map(latest_price_map)

    invalid_mask = (
        df["Quantity"].isna() | (df["Quantity"] <= 0)
        | df["Average Price"].isna() | (df["Average Price"] <= 0)
        | df["Latest Price"].isna() | (df["Latest Price"] <= 0)
        | (df["Yahoo Ticker"].fillna("").astype(str).str.strip() == "")
        | (df["Currency"].fillna("").astype(str).str.strip() == "")
    )
    invalid_rows = df.loc[invalid_mask, "Symbol"].tolist()
    usable = df.loc[~invalid_mask].copy()
    if usable.empty:
        return pd.DataFrame(), invalid_rows

    fx_map = get_fx_to_inr_map(usable["Currency"].tolist())
    usable["FX to INR"] = usable["Currency"].map(fx_map)
    usable["Average Price INR"] = usable["Average Price"] * usable["FX to INR"]
    usable["Latest Price INR"] = usable["Latest Price"] * usable["FX to INR"]
    usable["Invested INR"] = usable["Quantity"] * usable["Average Price INR"]
    usable["Current Value INR"] = usable["Quantity"] * usable["Latest Price INR"]

    total_current = usable["Current Value INR"].sum()
    if total_current <= 0:
        raise ValueError("The master holdings table has no positive current INR portfolio value.")

    usable["Weight"] = usable["Current Value INR"] / total_current
    portfolio_df = usable[
        [
            "Symbol", "Stock Name", "Yahoo Ticker", "Exchange", "Currency",
            "Quantity", "Average Price", "Latest Price", "FX to INR",
            "Average Price INR", "Latest Price INR", "Invested INR",
            "Current Value INR", "Weight"
        ]
    ].sort_values("Weight", ascending=False).reset_index(drop=True)

    return portfolio_df, invalid_rows


def extend_allocation_with_universal_candidates(portfolio_df):
    """Add zero-quantity rows for Universal Portfolio symbols not already held.

    This lets the optimizer treat every shared Universal Portfolio symbol as a
    candidate for the optimal allocation (current weight 0) alongside the user's
    real holdings, so the rebalancing plan can recommend buying new stocks too.
    Returns (extended_df, added_symbols).
    """
    universal_df = load_master_holdings(UNIVERSAL_OWNER)
    if universal_df.empty:
        return portfolio_df, []

    held_symbols = set(portfolio_df["Symbol"]) if not portfolio_df.empty else set()
    candidates = universal_df[~universal_df["Symbol"].isin(held_symbols)].copy()
    if candidates.empty:
        return portfolio_df, []

    candidates["Symbol"] = candidates["Symbol"].map(normalize_portfolio_symbol)
    candidates["Yahoo Ticker"] = candidates["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    candidates["Currency"] = candidates["Currency"].fillna("").map(_normalize_currency_code)
    candidates = candidates[candidates["Yahoo Ticker"] != ""]
    if candidates.empty:
        return portfolio_df, []

    latest_price_map = get_latest_price_map(tuple(candidates["Yahoo Ticker"].tolist()))
    fx_map = get_fx_to_inr_map(candidates["Currency"].tolist())

    rows = []
    added_symbols = []
    for _, row in candidates.iterrows():
        price = latest_price_map.get(row["Yahoo Ticker"])
        fx = fx_map.get(row["Currency"])
        if price is None or not np.isfinite(price) or price <= 0 or fx is None:
            continue

        price_inr = float(price) * float(fx)
        rows.append({
            "Symbol": row["Symbol"],
            "Stock Name": row["Stock Name"],
            "Yahoo Ticker": row["Yahoo Ticker"],
            "Exchange": row["Exchange"],
            "Currency": row["Currency"],
            "Quantity": 0.0,
            "Average Price": float(price),
            "Latest Price": float(price),
            "FX to INR": float(fx),
            "Average Price INR": price_inr,
            "Latest Price INR": price_inr,
            "Invested INR": 0.0,
            "Current Value INR": 0.0,
            "Weight": 0.0,
        })
        added_symbols.append(row["Symbol"])

    if not rows:
        return portfolio_df, []

    extended_df = pd.concat([portfolio_df, pd.DataFrame(rows)], ignore_index=True)
    return extended_df, added_symbols

# =========================================================
# RETURNS / OPTIMIZATION
# =========================================================

@st.cache_data(show_spinner=False)
def download_close_history(
    symbols,
    start_date="2000-01-01",
    end_date=None,
    buffer_days=7,
):
    """Download closing-price history once and reuse it during the same analysis."""
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = pd.to_datetime(end_date)

    effective_end = (end_date - timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    prices, failures = _download_close_prices_resilient(
        list(symbols),
        start=start_date,
        end=effective_end,
        batch_size=12,
    )

    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError(
            _format_download_failure_message(
                [str(symbol).strip().upper() for symbol in symbols],
                failures,
                context="historical close history",
            )
        )

    return prices


@st.cache_data(show_spinner=False)
def find_drop_bottom_pct_nearest_target(
    symbols,
    target_trading_days=252,
    start_date="2000-01-01",
    end_date=None,
    buffer_days=7,
):
    """Return the 0.01 drop fraction producing trading days nearest to 252+.

    Values from 0.00 through 0.95 are tested, matching the Streamlit control.
    A result at or above the target is preferred. If the target cannot be
    reached, the value with the maximum available trading days is returned.
    """
    prices = download_close_history(
        tuple(symbols),
        start_date=start_date,
        end_date=end_date,
        buffer_days=buffer_days,
    ).copy()

    lengths = prices.count().sort_values(ascending=False)
    total_tickers = len(lengths)

    if total_tickers == 0:
        raise ValueError("No ticker history is available.")

    minimum_tickers_to_keep = 2 if total_tickers >= 2 else 1
    maximum_allowed_drop = total_tickers - minimum_tickers_to_keep
    candidates = []

    # Calculate each ticker boundary once. Candidate evaluation then only needs
    # date-index lookups instead of rebuilding up to 96 full return matrices.
    ordered_columns = list(lengths.index)
    first_valid_dates = prices[ordered_columns].apply(
        lambda column: column.first_valid_index()
    )
    last_valid_dates = prices[ordered_columns].apply(
        lambda column: column.last_valid_index()
    )
    date_index = prices.index

    for step_number in range(96):
        pct = round(step_number / 100, 2)
        num_to_drop = int(np.floor(pct * total_tickers))

        if num_to_drop > maximum_allowed_drop:
            continue

        kept_count = total_tickers - num_to_drop
        kept_columns = ordered_columns[:kept_count]

        valid_start = first_valid_dates.loc[kept_columns].max()
        valid_end = last_valid_dates.loc[kept_columns].min()

        if (
            valid_start is None
            or valid_end is None
            or pd.isna(valid_start)
            or pd.isna(valid_end)
            or valid_start >= valid_end
        ):
            continue

        first_position = int(date_index.searchsorted(valid_start, side="left"))
        end_position = int(date_index.searchsorted(valid_end, side="right"))

        # One price row is lost when prices are converted to daily returns.
        trading_days = max(end_position - first_position - 1, 0)

        if trading_days <= 0:
            continue

        candidates.append({
            "drop_bottom_pct": float(pct),
            "trading_days": int(trading_days),
            "num_to_drop": int(num_to_drop),
            "tickers_kept": int(kept_count),
        })

    if not candidates:
        raise ValueError("No usable overlapping history was found.")

    target_candidates = [
        candidate
        for candidate in candidates
        if candidate["trading_days"] >= target_trading_days
    ]

    if target_candidates:
        selected = min(
            target_candidates,
            key=lambda candidate: (
                candidate["trading_days"] - target_trading_days,
                candidate["drop_bottom_pct"],
            ),
        ).copy()
        selected["target_reached"] = True
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["trading_days"],
                -candidate["drop_bottom_pct"],
            ),
        ).copy()
        selected["target_reached"] = False

    selected["target_trading_days"] = int(target_trading_days)
    return selected


@st.cache_data(show_spinner=False)
def get_daily_log_returns(
    symbols,
    start_date=None,
    end_date=None,
    buffer_days=7,
    drop_bottom_pct=0.1,
    ticker_currency_pairs=(),
):
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = pd.to_datetime(end_date)

    if start_date is None:
        start_date = "2000-01-01"

    requested_tickers = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    df = download_close_history(
        tuple(symbols),
        start_date=start_date,
        end_date=end_date,
        buffer_days=buffer_days,
    ).copy()

    if df.empty:
        raise ValueError("No data available for the given tickers.")

    available_tickers = {str(col).strip().upper() for col in df.columns}
    missing_history_tickers = sorted(set(requested_tickers) - available_tickers)

    # Convert foreign-market price histories into INR before calculating returns.
    if ticker_currency_pairs:
        df = convert_price_history_to_inr(df, ticker_currency_pairs)

    lengths = df.count().sort_values(ascending=False)
    num_to_drop = int(np.floor(drop_bottom_pct * len(lengths)))

    dropped_df = pd.DataFrame()
    if num_to_drop > 0:
        dropped = lengths.tail(num_to_drop)
        kept = lengths.head(len(lengths) - num_to_drop)
        df = df[kept.index]
        dropped_df = pd.DataFrame({
            "Ticker": dropped.index,
            "Valid Days of Data": dropped.values,
        }).reset_index(drop=True)
    else:
        kept = lengths

    if len(kept) == 0:
        raise ValueError("History filtering removed every ticker.")

    valid_start = df[kept.index].apply(lambda x: x.first_valid_index()).max()
    valid_end = df[kept.index].apply(lambda x: x.last_valid_index()).min()

    if valid_start is None or valid_end is None or valid_start >= valid_end:
        raise ValueError("No overlapping date range found across tickers after filtering.")

    df_aligned = df.loc[valid_start:valid_end].ffill().dropna(axis=0, how="any")
    log_returns = np.log(df_aligned / df_aligned.shift(1)).dropna()

    lengths = df[kept.index].count()
    min_len_ticker = lengths.idxmin()

    try:
        start_price = df[min_len_ticker].dropna().iloc[0]
        end_price = df[min_len_ticker].dropna().iloc[-1]
        simulated_pnl = ((end_price - start_price) / start_price) * 100
    except Exception:
        simulated_pnl = np.nan

    min_len_df = pd.DataFrame({
        "Ticker": [min_len_ticker],
        "History Length (days)": [lengths[min_len_ticker]],
        "P&L (simulated, INR-adjusted)": [
            f"{simulated_pnl:.2f}%" if not np.isnan(simulated_pnl) else "N/A"
        ],
    })

    meta = {
        "valid_start": valid_start,
        "valid_end": valid_end,
        "dropped_df": dropped_df,
        "min_len_df": min_len_df,
        "missing_history_tickers": missing_history_tickers,
    }
    return log_returns, meta

def optimize_portfolio_max_return_given_daily_risk(log_returns, max_drawdown=0.1):
    from scipy.optimize import minimize
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        risk_free_rate_annual = 0.1171
        risk_free_rate_daily = risk_free_rate_annual / 250
        port_return = np.dot(weights, mean_returns)
        excess_return = port_return - risk_free_rate_daily
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_volatility == 0:
            return np.inf
        return -excess_return / port_volatility

    def portfolio_drawdown(weights):
        portfolio_returns = log_returns @ weights
        cumulative = portfolio_returns.cumsum()
        peak = cumulative.cummax()
        drawdown = (peak - cumulative).max()
        return drawdown

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: max_drawdown - portfolio_drawdown(x)}
    ]
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = np.ones(num_assets) / num_assets

    result = minimize(negative_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None


def optimize_max_sharpe_ratio(log_returns):
    from scipy.optimize import minimize
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        risk_free_rate = 0.112 / 250
        port_return = np.dot(weights, mean_returns)
        excess_return = port_return - risk_free_rate
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -excess_return / port_volatility if port_volatility != 0 else np.inf

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = np.ones(num_assets) / num_assets

    result = minimize(negative_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None


def optimize_portfolio_target_volatility(log_returns, target_volatility=0.1):
    from scipy.optimize import minimize
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(mean_returns)

    def objective(weights):
        return -np.dot(weights, mean_returns)

    def constraint_sum(weights):
        return np.sum(weights) - 1

    def constraint_volatility(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return target_volatility - portfolio_vol

    constraints = [
        {"type": "eq", "fun": constraint_sum},
        {"type": "ineq", "fun": constraint_volatility}
    ]
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial = np.ones(num_assets) / num_assets

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x if result.success else None


def portfolio_stats(weights, log_returns):
    from scipy.stats import kurtosis, norm, skew
    portfolio_returns = log_returns @ weights
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    annualized_return = mean * 250
    annualized_vol = std * np.sqrt(250)

    s = skew(portfolio_returns)
    k = kurtosis(portfolio_returns, fisher=True)
    alpha = 0.05
    z = norm.ppf(alpha)
    z_cf = z + (1/6)*(z**2 - 1)*s + (1/24)*(z**3 - 3*z)*k - (1/36)*(2*z**3 - 5*z)*s**2
    cvar_cf = -(mean + z_cf * std) * 250

    risk_free_rate_annual = 0.112
    excess_return = annualized_return - risk_free_rate_annual
    sharpe = excess_return / annualized_vol if annualized_vol != 0 else 0

    return {
        "Annual Return": annualized_return,
        "Annual Volatility": annualized_vol,
        "Cornish-Fisher CVaR": cvar_cf,
        "Sharpe Ratio": sharpe,
    }


def portfolio_stats_comparison(current_alloc, log_returns, optimal_weights):
    aligned = (
        current_alloc.set_index("Yahoo Ticker")
        .reindex(log_returns.columns)
        .dropna(subset=["Weight"])
    )
    if aligned.empty:
        raise ValueError("No holdings align with the return matrix.")

    current_weights = aligned["Weight"].values
    current_weights = current_weights / current_weights.sum()
    current_log_returns = log_returns[list(aligned.index)]

    current_stats = portfolio_stats(current_weights, current_log_returns)
    optimal_stats = portfolio_stats(optimal_weights, log_returns)
    return current_stats, optimal_stats

def run_portfolio_analysis_multi(
    symbols,
    current_alloc,
    max_dd=0.05,
    target_volatility=None,
    drop_bottom_pct=0.1,
):
    ticker_currency_pairs = tuple(
        sorted(
            (str(row["Yahoo Ticker"]), _normalize_currency_code(row["Currency"]))
            for _, row in current_alloc.iterrows()
            if str(row.get("Yahoo Ticker", "")).strip()
        )
    )
    log_returns, meta = get_daily_log_returns(
        tuple(symbols),
        drop_bottom_pct=drop_bottom_pct,
        ticker_currency_pairs=ticker_currency_pairs,
    )

    if target_volatility is not None:
        optimal_weights = optimize_portfolio_target_volatility(
            log_returns, target_volatility=target_volatility
        )
    else:
        optimal_weights = None

    if optimal_weights is None:
        optimal_weights = optimize_portfolio_max_return_given_daily_risk(
            log_returns, max_drawdown=max_dd
        )
    if optimal_weights is None:
        optimal_weights = optimize_max_sharpe_ratio(log_returns)
    if optimal_weights is None:
        return None, log_returns, None, None, meta

    current_stats, optimal_stats = portfolio_stats_comparison(
        current_alloc, log_returns, optimal_weights
    )
    return optimal_weights, log_returns, current_stats, optimal_stats, meta

# =========================================================
# REBALANCING
# =========================================================

def lumpsum_allocation_plan(current_alloc, optimal_weights, log_returns, prices, lumpsum_inr):
    """Allocate an INR lumpsum to optimized weights using whole-share quantities."""
    if lumpsum_inr <= 0:
        return pd.DataFrame(), float(lumpsum_inr), [], []

    alloc_df = current_alloc.set_index("Yahoo Ticker").copy()
    lr_tickers = list(log_returns.columns)
    common_tickers = [ticker for ticker in lr_tickers if ticker in alloc_df.index and ticker in prices]
    missing_prices = [ticker for ticker in lr_tickers if ticker not in prices]
    missing_alloc = [ticker for ticker in lr_tickers if ticker not in alloc_df.index]

    if not common_tickers:
        raise ValueError("No common tickers between returns, allocation, and latest prices.")

    positions = [lr_tickers.index(ticker) for ticker in common_tickers]
    aligned_weights = np.array(optimal_weights, dtype=float)[positions]
    aligned_weights = aligned_weights / aligned_weights.sum()
    price_inr = np.array(
        [prices[ticker] * alloc_df.loc[ticker, "FX to INR"] for ticker in common_tickers],
        dtype=float,
    )
    target_amounts = float(lumpsum_inr) * aligned_weights
    quantities = np.floor(target_amounts / price_inr).astype(int)
    invested_amounts = quantities * price_inr

    # Initial whole-share rounding leaves the target slices of higher-priced
    # assets unspent. Reinvest the residual one share at a time where it most
    # reduces the gap from the optimizer's target amount.
    remaining_cash = float(lumpsum_inr) - float(invested_amounts.sum())
    while True:
        affordable = np.flatnonzero(price_inr <= remaining_cash + 1e-9)
        if len(affordable) == 0:
            break

        current_amounts = quantities * price_inr
        gap_reduction = (
            np.abs(current_amounts[affordable] - target_amounts[affordable])
            - np.abs(
                current_amounts[affordable]
                + price_inr[affordable]
                - target_amounts[affordable]
            )
        )
        best_position = affordable[np.argmax(gap_reduction)]
        quantities[best_position] += 1
        invested_amounts[best_position] += price_inr[best_position]
        remaining_cash -= price_inr[best_position]

    plan = pd.DataFrame({
        "Symbol": alloc_df.loc[common_tickers, "Symbol"].values,
        "Yahoo Ticker": common_tickers,
        "Optimal Weight": aligned_weights,
        "Target Amount INR": target_amounts,
        "Latest Price INR": price_inr,
        "Suggested Quantity": quantities,
        "Estimated Investment INR": invested_amounts,
    }).sort_values("Optimal Weight", ascending=False).reset_index(drop=True)

    unallocated_cash = max(0.0, remaining_cash)
    return plan, unallocated_cash, missing_prices, missing_alloc


def lumpsum_execution_sheet_html(lumpsum_df, lumpsum_inr, unallocated_cash):
    """Build a compact shareable buy-order sheet without portfolio analysis details."""
    orders = lumpsum_df[lumpsum_df["Suggested Quantity"] > 0].copy()
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['Symbol']))}</td>"
        f"<td>{html.escape(str(row['Yahoo Ticker']))}</td>"
        f"<td>{int(row['Suggested Quantity']):,}</td>"
        f"<td>INR {float(row['Estimated Investment INR']):,.2f}</td>"
        "</tr>"
        for _, row in orders.iterrows()
    )
    if not rows:
        rows = '<tr><td colspan="4">No whole-share buy orders for this amount.</td></tr>'

    estimated_investment = float(orders["Estimated Investment INR"].sum())
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lumpsum Buy Orders</title>
<style>
body {{ font-family: Arial, sans-serif; color: #17212b; margin: 32px; max-width: 760px; }}
h1 {{ margin: 0 0 6px; font-size: 24px; }}
p {{ margin: 4px 0; }}
.summary {{ margin: 24px 0; padding: 14px; background: #f2f6f4; border-left: 4px solid #1f7a5b; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d7dce0; padding: 10px 8px; text-align: left; }}
th {{ background: #17212b; color: white; }}
.footnote {{ margin-top: 20px; color: #5d6872; font-size: 12px; }}
@media print {{ body {{ margin: 16px; }} }}
</style>
</head>
<body>
<h1>Lumpsum Buy Orders</h1>
<p>Generated {generated_at}</p>
<div class="summary">
<p><strong>Lumpsum:</strong> INR {float(lumpsum_inr):,.2f}</p>
<p><strong>Estimated order value:</strong> INR {estimated_investment:,.2f}</p>
<p><strong>Cash remaining:</strong> INR {float(unallocated_cash):,.2f}</p>
</div>
<table>
<thead><tr><th>Symbol</th><th>Yahoo Ticker</th><th>Buy Quantity</th><th>Estimated Value</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="footnote">Whole-share quantities only. Verify live price, exchange, and order limits with your broker before placing orders.</p>
</body>
</html>"""


def rebalance_plan_multi(current_alloc, optimal_weights, log_returns, prices, days_to_flip):
    alloc_df = current_alloc.set_index("Yahoo Ticker").copy()
    lr_tickers = list(log_returns.columns)
    common_tickers = [t for t in lr_tickers if t in alloc_df.index and t in prices]

    missing_prices = [t for t in lr_tickers if t not in prices]
    missing_alloc = [t for t in lr_tickers if t not in alloc_df.index]

    if not common_tickers:
        raise ValueError("No common tickers between returns, allocation, and latest prices.")

    pos = [lr_tickers.index(t) for t in common_tickers]
    aligned_optimal_weights = np.array(optimal_weights)[pos]
    aligned_optimal_weights = aligned_optimal_weights / aligned_optimal_weights.sum()

    current_values = np.array(
        [
            alloc_df.loc[t, "Quantity"]
            * prices[t]
            * alloc_df.loc[t, "FX to INR"]
            for t in common_tickers
        ],
        dtype=float,
    )
    portfolio_value = current_values.sum()
    current_weights = current_values / portfolio_value

    change_weights = aligned_optimal_weights - current_weights
    value_change_inr = portfolio_value * change_weights
    price_inr = np.array(
        [prices[t] * alloc_df.loc[t, "FX to INR"] for t in common_tickers],
        dtype=float,
    )
    qty_change = value_change_inr / price_inr

    action = np.where(change_weights > 0, "Buy", "Sell")
    abs_qty_change = np.abs(qty_change)
    exec_qty = np.where(
        action == "Sell",
        np.maximum(0, np.floor(abs_qty_change).astype(int) - 1),
        np.floor(abs_qty_change / days_to_flip).astype(int),
    )
    exec_val = exec_qty * price_inr

    # Rank the trades by expected annual return lift from moving the portfolio
    # toward the optimized weights. This is expressed in percentage-point return
    # terms, which is easier to interpret than an abstract trade score.
    annualized_mean = log_returns[common_tickers].mean().to_numpy() * 250.0
    annualized_vol = log_returns[common_tickers].std(ddof=1).to_numpy() * np.sqrt(250.0)
    target_gap = np.abs(change_weights)

    # Estimate annual return lift from the target-gap, and penalise high-volatility
    # assets so the ranking stays in a return-like, natural scale.
    return_lift_pct = target_gap * annualized_mean * 100.0
    volatility_penalty = target_gap * annualized_vol * 50.0
    trade_score = np.clip(return_lift_pct - volatility_penalty, 0.0, None)

    rebal_df = pd.DataFrame({
        "Symbol": [alloc_df.loc[t, "Symbol"] for t in common_tickers],
        "Yahoo Ticker": common_tickers,
        "Currency": [alloc_df.loc[t, "Currency"] for t in common_tickers],
        "Native Price": [prices[t] for t in common_tickers],
        "FX to INR": [alloc_df.loc[t, "FX to INR"] for t in common_tickers],
        "Current Weight": current_weights,
        "Optimal Weight": aligned_optimal_weights,
        "Action": action,
        "Change": np.abs(change_weights),
        "Expected Annual Return Lift (%)": trade_score,
        "Quantity": abs_qty_change,
        "Executable Quantity": exec_qty,
        "Executable Value": exec_val,
    })

    rebal_df = (
        rebal_df[rebal_df["Executable Quantity"] != 0]
        .sort_values(by="Expected Annual Return Lift (%)", ascending=False)
        .reset_index(drop=True)
    )
    return rebal_df, missing_prices, missing_alloc

@st.cache_data(show_spinner=False)
def get_latest_price_map(latest_prices):
    tickers = tuple(dict.fromkeys(str(t).strip().upper() for t in latest_prices if str(t).strip()))
    if not tickers:
        return {}

    price_history, _ = _download_close_prices_resilient(
        list(tickers),
        period="15d",
        batch_size=12,
    )

    if price_history.empty:
        return {}

    price_history = price_history.ffill()
    last_row = price_history.iloc[-1]
    return {
        str(col).upper(): float(last_row[col])
        for col in price_history.columns
        if pd.notna(last_row[col]) and np.isfinite(float(last_row[col]))
    }

def _sort_rebalance_df_for_priority(df):
    """Return a rebalancing table sorted by expected annual return lift first."""
    if df is None or df.empty:
        return df

    frame = df.copy()
    if "Expected Annual Return Lift (%)" in frame.columns:
        frame["_priority_sort"] = pd.to_numeric(
            frame["Expected Annual Return Lift (%)"],
            errors="coerce",
        )
        return (
            frame.sort_values(
                by="_priority_sort",
                ascending=False,
                na_position="last",
                kind="stable",
            )
            .drop(columns="_priority_sort")
            .reset_index(drop=True)
        )

    if "Optimal Weight" in frame.columns:
        weight_text = frame["Optimal Weight"].astype(str).str.strip()
        weight_is_percent = weight_text.str.endswith("%")
        weight_sort = pd.to_numeric(weight_text.str.rstrip("%"), errors="coerce")
        weight_sort = weight_sort.where(~weight_is_percent, weight_sort / 100.0)
        frame["_priority_sort"] = weight_sort
        return (
            frame.sort_values(
                by="_priority_sort",
                ascending=False,
                na_position="last",
                kind="stable",
            )
            .drop(columns="_priority_sort")
            .reset_index(drop=True)
        )

    return frame.sort_values(by=list(frame.columns[:1]), ascending=False, kind="stable").reset_index(drop=True)


def style_rebalance_df(df):
    def color_action_row(row):
        if row["Action"] == "Buy":
            return ["background-color: #d4edda; color: #155724;"] * len(row)
        return ["background-color: #f8d7da; color: #721c24;"] * len(row)

    formatters = {
        "Native Price": "{:,.2f}",
        "FX to INR": "{:,.4f}",
        "Current Weight": "{:.2%}",
        "Optimal Weight": "{:.2%}",
        "Change": "{:.2%}",
        "Expected Annual Return Lift (%)": "{:.2f}",
        "Quantity": "{:.2f}",
        "Executable Quantity": "{:.0f}",
        "Executable Value": "₹{:,.0f}",
    }

    sorted_df = _sort_rebalance_df_for_priority(df)
    return sorted_df.style.apply(color_action_row, axis=1).format(formatters, na_rep="N/A")

def metrics_df(stats_dict):
    return pd.DataFrame({
        "Metric": list(stats_dict.keys()),
        "Value": list(stats_dict.values())
    })


def calculate_drop_bottom_pct_recommendation(owner):
    """Calculate a display-only recommendation using INR-adjusted return history."""
    st.session_state.pop("drop_bottom_auto_result", None)
    st.session_state.pop("drop_bottom_auto_error", None)

    try:
        portfolio_df, _ = build_current_allocation_from_db(owner)
        if portfolio_df.empty:
            raise ValueError("No usable holdings are available.")

        yahoo_tickers = tuple(portfolio_df["Yahoo Ticker"].dropna().astype(str).tolist())
        if not yahoo_tickers:
            raise ValueError("No Yahoo Finance tickers could be resolved.")

        selected = find_drop_bottom_pct_by_return_gap(yahoo_tickers, portfolio_df.copy())
        st.session_state["drop_bottom_auto_result"] = selected
    except Exception as exc:
        st.session_state["drop_bottom_auto_error"] = str(exc)

def clear_drop_bottom_coverage_preview():
    """Discard a coverage preview when the manual percentage changes."""
    st.session_state.pop("drop_bottom_coverage_preview", None)
    st.session_state.pop("drop_bottom_coverage_error", None)


@st.cache_data(show_spinner=False)
def find_drop_bottom_pct_by_return_gap(
    symbols,
    current_alloc,
    max_dd=0.05,
    target_volatility=None,
):
    """Search drop_bottom_pct values from 0.10 to 0.99 for higher optimized returns."""
    if not symbols:
        raise ValueError("No tickers were provided for the recommendation search.")

    evaluated = {}

    def evaluate_pct(pct):
        pct = round(min(max(pct, 0.10), 0.99), 2)
        if pct in evaluated:
            return evaluated[pct]

        try:
            optimal_weights, log_returns, current_stats, optimal_stats, meta = (
                run_portfolio_analysis_multi(
                    symbols,
                    current_alloc,
                    max_dd=max_dd,
                    target_volatility=target_volatility,
                    drop_bottom_pct=pct,
                )
            )
        except Exception:
            evaluated[pct] = None
            return None

        if current_stats is None or optimal_stats is None:
            evaluated[pct] = None
            return None

        current_ret = current_stats.get("Annual Return", 0.0)
        optimal_ret = optimal_stats.get("Annual Return", 0.0)

        if optimal_ret <= current_ret:
            evaluated[pct] = None
            return None

        gap = max((optimal_ret - current_ret) / abs(optimal_ret), 0.0) if abs(optimal_ret) > 1e-12 else abs(current_ret - optimal_ret)

        candidate = {
            "drop_bottom_pct": pct,
            "current_annual_return": current_ret,
            "optimized_annual_return": optimal_ret,
            "return_gap_pct": gap,
            "trading_days": int(log_returns.shape[0]) if log_returns is not None else 0,
            "assets": int(log_returns.shape[1]) if log_returns is not None else 0,
            "target_reached": True,
        }
        evaluated[pct] = candidate
        return candidate

    low = 0.10
    high = 0.99
    current_pct = 0.18

    current_candidate = evaluate_pct(current_pct)
    if current_candidate is not None:
        return current_candidate

    queue = [(low, high)]

    while queue:
        low, high = queue.pop(0)
        if round(high - low, 3) < 0.01:
            continue

        mid = round((low + high) / 2, 2)
        if mid == low or mid == high:
            continue

        mid_candidate = evaluate_pct(mid)
        if mid_candidate is not None:
            return mid_candidate

        queue.append((low, mid))
        queue.append((mid, high))

    raise ValueError(
        "Could not find a usable history filter that supports portfolio optimization."
    )


def calculate_drop_bottom_coverage_preview(drop_bottom_pct, owner):
    """Estimate coverage using available Yahoo histories; do not run optimization."""
    portfolio_df, invalid_rows = build_current_allocation_from_db(owner)
    if portfolio_df.empty:
        raise ValueError("No usable holdings are available.")

    yahoo_tickers = tuple(portfolio_df["Yahoo Ticker"].dropna().astype(str).tolist())
    if not yahoo_tickers:
        raise ValueError("No Yahoo Finance tickers could be resolved.")

    total_tickers = len(yahoo_tickers)
    num_to_drop = int(np.floor(float(drop_bottom_pct) * total_tickers))
    tickers_kept = max(total_tickers - num_to_drop, 0)

    history, failures = _download_close_prices_resilient(
        list(yahoo_tickers),
        period="5y",
        batch_size=12,
    )
    history = history.dropna(axis=1, how="all")
    if history.empty:
        raise ValueError(
            _format_download_failure_message(
                list(yahoo_tickers),
                failures,
                context="coverage preview",
            )
        )

    lengths = history.count().sort_values(ascending=False)
    available_tickers = len(lengths)
    actual_kept = max(min(tickers_kept, available_tickers), 0)
    filtered_tickers = lengths.head(actual_kept).index.tolist()
    available_days = int(lengths.head(actual_kept).min()) if actual_kept > 0 else 0
    dropped_tickers = lengths.tail(num_to_drop).index.tolist() if num_to_drop > 0 else []

    return {
        "drop_bottom_pct": float(drop_bottom_pct),
        "trading_days": int(available_days),
        "assets": int(actual_kept),
        "total_tickers": int(total_tickers),
        "tickers_dropped": int(num_to_drop),
        "tickers_kept": int(actual_kept),
        "filtered_tickers": filtered_tickers,
        "dropped_tickers": dropped_tickers,
        "resolved_tickers": int(len(yahoo_tickers)),
        "invalid_holding_rows": int(len(invalid_rows)),
    }

# =========================================================
# UI
# =========================================================

try:
    recovered_db_path = init_holdings_db()
except sqlite3.Error as exc:
    st.error(
        "Could not initialize the holdings database. "
        f"SQLite reported: {exc}. Active database path: {DB_PATH}"
    )
    st.stop()

st.title("📊 Portfolio Rebalancer")
st.caption("Holdings are stored in a local SQLite master table with multi-market ticker and currency metadata.")
st.caption(f"Active SQLite file: `{DB_PATH}`")
st.caption(f"App build: `{APP_BUILD}`")

if recovered_db_path is not None:
    st.warning(
        "The previous database file was not a valid SQLite database and was preserved as "
        f"`{recovered_db_path.name}`. A new holdings database was created."
    )

st.divider()
st.subheader("👤 Your profile")
with st.container(border=True):
    st.caption(
        "Each nickname keeps its holdings and saved analysis completely separate from "
        "every other person using this app — nobody can see or overwrite your data."
    )
    entered_user = st.text_input(
        "Enter your name/nickname",
        value=st.session_state.get("current_user", ""),
        key="current_user_input",
        placeholder="e.g. Warren Buffett, Jane Doe, or just your initials",
    ).strip()

if entered_user != st.session_state.get("current_user", ""):
    st.session_state["current_user"] = entered_user
    clear_drop_bottom_coverage_preview()
    st.session_state.pop("drop_bottom_auto_result", None)
    st.session_state.pop("drop_bottom_auto_error", None)
    st.rerun()

CURRENT_USER = st.session_state.get("current_user", "").strip()

if not CURRENT_USER:
    st.info("Enter your name above to load your personal holdings and continue.")
    st.stop()

if CURRENT_USER.lower() == UNIVERSAL_OWNER.lower():
    st.error("That name is reserved for the shared universal portfolio. Please choose another nickname.")
    st.stop()

st.caption(f"Signed in as: **{CURRENT_USER}**")
st.divider()

# Filled after every add/remove operation using a fresh SQLite query.
live_count_banner_placeholder = st.empty()

# Latest saved/restored analysis is rendered directly below the live banner.
saved_analysis_placeholder = st.empty()

update_messages = []
update_warnings = []
update_errors = []

if "holdings_editor_version" not in st.session_state:
    st.session_state["holdings_editor_version"] = 0

for flash_key, target in (
    ("holdings_flash_success", update_messages),
    ("holdings_flash_warning", update_warnings),
    ("holdings_flash_error", update_errors),
):
    flash_message = st.session_state.pop(flash_key, None)
    if flash_message:
        target.append(flash_message)

# Self-heal market metadata created by older/NSE-only builds. In particular, the
# first multi-market build could leave VT stored as VT.NS/NSE/INR.
try:
    repaired_market_symbols = repair_master_holdings_metadata(CURRENT_USER)
except Exception:
    repaired_market_symbols = []

if repaired_market_symbols:
    st.session_state.pop("drop_bottom_coverage_preview", None)
    st.session_state.pop("drop_bottom_coverage_error", None)
    st.session_state.pop("drop_bottom_auto_result", None)
    st.session_state.pop("drop_bottom_auto_error", None)
    update_messages.append(
        "Repaired market/currency metadata: " + ", ".join(sorted(repaired_market_symbols))
        + ". Re-run optimization to refresh the saved analysis."
    )

with st.sidebar:
    st.header("Holdings database")
    sidebar_count_placeholder = st.empty()

    buy_input = st.text_area(
        "Buy / add symbols",
        placeholder="RELIANCE, TCS, VT, VOO, QQQ",
        help="Plain symbols found in the NSE equity list use .NS; otherwise the global Yahoo ticker is tried first (for example VT). Explicit Yahoo tickers such as RELIANCE.BO, 500325.BO or VUSA.L are preserved.",
    )
    sell_input = st.text_area(
        "Sell / remove symbols",
        placeholder="HDFCBANK, SBIN",
        help="Symbols entered here are removed completely from the master holdings table.",
    )

    update_holdings_btn = st.button("Update master holdings", width="stretch")

    st.divider()
    with st.expander("📦 Holdings backup and restore", expanded=False):
        holdings_csv = holdings_backup_bytes(CURRENT_USER)
        holdings_backup_date = datetime.now().strftime("%Y-%m-%d")
        st.download_button(
            "Download holdings backup CSV",
            data=holdings_csv,
            file_name=f"portfolio_holdings_backup_{holdings_backup_date}.csv",
            mime="text/csv",
            width="stretch",
            key="download_holdings_backup_sidebar",
            disabled=get_unique_holdings_count(CURRENT_USER) == 0,
            help="Download this before a Streamlit Cloud restart or redeployment.",
        )

        holdings_backup_upload = st.file_uploader(
            "Upload holdings backup CSV",
            type=["csv"],
            key="holdings_backup_upload",
            help="Restore Symbol, Yahoo Ticker, Exchange, Currency, Quantity, Average Price and timestamps. Old NSE-only backups are still accepted.",
        )
        holdings_restore_choice = st.radio(
            "Restore behaviour",
            options=["Merge/update current holdings", "Replace current holdings"],
            index=0,
            key="holdings_restore_choice",
            help=(
                "Merge updates matching symbols and keeps other rows. Replace deletes the "
                "current holdings first."
            ),
        )
        restore_holdings_btn = st.button(
            "Restore uploaded holdings",
            width="stretch",
            key="restore_holdings_btn",
            disabled=holdings_backup_upload is None,
        )

    st.divider()
    with st.expander("💾 Analysis results backup", expanded=False):
        existing_analysis_backup = latest_analysis_backup_bytes(CURRENT_USER)
        if existing_analysis_backup is not None:
            latest_saved_payload = load_latest_analysis(CURRENT_USER) or {}
            latest_saved_date = (
                str(latest_saved_payload.get("saved_at", ""))[:10] or "latest"
            )
            st.download_button(
                "Download latest analysis JSON",
                data=existing_analysis_backup,
                file_name=f"portfolio_analysis_backup_{latest_saved_date}.json",
                mime="application/json",
                width="stretch",
                key="download_saved_analysis_sidebar",
            )
        else:
            st.caption("Run analysis once before downloading a result backup.")

        analysis_backup_upload = st.file_uploader(
            "Upload analysis backup JSON",
            type=["json"],
            key="analysis_backup_upload",
            help="Restores a previously downloaded complete analysis result.",
        )
        restore_analysis_btn = st.button(
            "Restore uploaded analysis",
            width="stretch",
            key="restore_analysis_btn",
            disabled=analysis_backup_upload is None,
        )

    st.divider()
    st.header("Analysis inputs")
    days_to_flip = st.number_input("Expected days to flip", min_value=1, value=13, step=1)
    max_dd_pct = st.number_input(
        "Max drawdown input (%)",
        min_value=0.00,
        value=23.00,
        step=0.01,
        format="%.2f",
    )
    max_dd = (max_dd_pct / 100)
    st.caption(f"Internal max_dd used: {max_dd:.4f}")

    drop_bottom_pct = float(
        st.number_input(
            "Drop bottom fraction of tickers by history length",
            min_value=0.0,
            max_value=0.95,
            value=0.20,
            step=0.01,
            format="%.2f",
            key="manual_drop_bottom_pct_v8",
            on_change=clear_drop_bottom_coverage_preview,
            help=(
                "This is a fully manual analysis input. Use Preview history coverage "
                "to see the resulting trading days and assets before optimization."
            ),
        )
    )

    preview_coverage_btn = st.button(
        "Preview history coverage",
        width="stretch",
        help=(
            "Calculates history coverage for the current percentage without "
            "running optimization."
        ),
        key="preview_history_coverage_btn",
    )

    last_pct = st.session_state.get("drop_bottom_last_pct")
    should_recalculate = preview_coverage_btn or last_pct != drop_bottom_pct

    if should_recalculate:
        st.session_state["drop_bottom_last_pct"] = drop_bottom_pct
        st.session_state["drop_bottom_coverage_error"] = None
        try:
            with st.spinner("Calculating history coverage only..."):
                st.session_state["drop_bottom_coverage_preview"] = (
                    calculate_drop_bottom_coverage_preview(drop_bottom_pct, CURRENT_USER)
                )
        except Exception as exc:
            st.session_state["drop_bottom_coverage_preview"] = None
            st.session_state["drop_bottom_coverage_error"] = str(exc)

    coverage_preview = st.session_state.get("drop_bottom_coverage_preview")
    if coverage_preview:
        kept_tickers = int(coverage_preview.get("tickers_kept", coverage_preview.get("assets", 0)))
        total_tickers = int(coverage_preview.get("total_tickers", 0))
        st.info(
            f"**Preview:** using {coverage_preview['drop_bottom_pct']:.2f} will keep {kept_tickers} of {total_tickers} holdings.  |  "
            f"**Trading days analysed:** {coverage_preview['trading_days']:,}  |  "
            f"**Assets in return matrix:** {coverage_preview['assets']:,}"
        )
        if kept_tickers < 10:
            st.warning(
                f"Only {kept_tickers} holdings remain after filtering, which may make the optimization unstable."
            )
        st.caption(
            "Coverage preview only — portfolio optimization has not been run. "
            "Adjust the percentage and watch this preview update live."
        )

    coverage_error = st.session_state.get("drop_bottom_coverage_error")
    if coverage_error:
        st.error(f"Could not calculate history coverage: {coverage_error}")

    st.button(
        "Find drop_bottom_pct in 0.10–0.99 for higher optimized return",
        width="stretch",
        on_click=calculate_drop_bottom_pct_recommendation,
        args=(CURRENT_USER,),
        help=(
            "Searches drop_bottom_pct values by history length from 0.10 to 0.99 so the "
            "optimized portfolio's annual return is strictly higher than the current portfolio's."
        ),
    )

    auto_drop_result = st.session_state.get("drop_bottom_auto_result")
    if auto_drop_result:
        status = (
            "meets" if auto_drop_result.get("target_reached") else "does not meet"
        )
        st.info(
            f"Auto value: {auto_drop_result['drop_bottom_pct']:.2f} — "
            f"Optimized return is higher and current return is {auto_drop_result['return_gap_pct']:.2%} behind; "
            f"{status} the 5% gap target. "
            f"Trading days: {auto_drop_result['trading_days']:,}."
        )

    auto_drop_error = st.session_state.get("drop_bottom_auto_error")
    if auto_drop_error:
        st.error(f"Could not calculate the recommendation: {auto_drop_error}")

    use_target_vol = st.checkbox("Use target volatility")
    target_volatility = (
        st.number_input(
            "Target volatility",
            min_value=0.00696,
            value=0.01455,
            step=0.0001,
            format="%.5f",
        )
        if use_target_vol
        else None
    )

st.divider()
st.subheader("✅ Required steps for every user")
st.caption(
    "Upload your own broker holdings, then run optimization — it automatically "
    "considers both your holdings and the shared Universal Portfolio as candidates."
)

step_col1, step_col2, step_col3 = st.columns(3, gap="medium")

with step_col1:
    with st.container(border=True):
        st.markdown("**1️⃣ Upload your broker holdings**")
        broker_holdings_upload = st.file_uploader(
            "Upload broker holdings .xlsx",
            type=["xlsx", "xls"],
            key="broker_holdings_upload",
            help=(
                "Statement with Stock Name, ISIN, Quantity, Average Buy Price, Buy Value, "
                "Closing Price, Closing Value, Unrealised P&L. Header row is auto-detected "
                "(row 11 by default). Stocks are matched to symbols by ISIN."
            ),
        )
        broker_holdings_mode = st.radio(
            "Import behaviour",
            options=["Merge/update current holdings", "Replace current holdings"],
            index=0,
            key="broker_holdings_import_mode",
            help=(
                "Merge updates matching symbols (by ISIN) and keeps other rows. Replace "
                "deletes the current holdings first."
            ),
        )
        import_broker_holdings_btn = st.button(
            "Import broker holdings",
            width="stretch",
            key="import_broker_holdings_btn",
            disabled=broker_holdings_upload is None,
        )

with step_col2:
    with st.container(border=True):
        st.markdown("**2️⃣ Enter lumpsum (optional)**")
        lumpsum_inr = st.number_input(
            "Lumpsum to allocate (INR)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
            help="After optimization, creates a whole-share buy plan using the optimal portfolio weights.",
        )

with step_col3:
    with st.container(border=True):
        st.markdown("**3️⃣ Run optimization**")
        st.caption(
            "Uses the days-to-flip/drawdown/drop_bottom_pct/target-volatility settings "
            "from the sidebar, and includes Universal Portfolio symbols as buy candidates."
        )
        run_btn = st.button(
            "Run optimization",
            width="stretch",
            type="primary",
            key="run_optimization_btn_main",
        )

lumpsum_download_placeholder = st.empty()

st.divider()
with st.expander("🌐 Universal Portfolio", expanded=False):
    st.caption(
        "One shared list of symbols visible and editable by everyone (quantity is always "
        "0, so it never counts as anyone's real holding). Run optimization automatically "
        "considers these as additional buy candidates alongside your own holdings."
    )
    universal_df = load_master_holdings(UNIVERSAL_OWNER)

    with st.container(border=True):
        st.markdown("**Shared symbols**")
        if universal_df.empty:
            st.info("The universal portfolio is empty. Add symbols below.")
        else:
            st.dataframe(
                universal_df[["Symbol", "Stock Name", "Yahoo Ticker", "Exchange", "Currency"]],
                width="stretch",
                hide_index=True,
            )

    with st.container(border=True):
        st.markdown("**Add / remove symbols**")
        universal_buy_input = st.text_area(
            "Add symbols to universal portfolio",
            key="universal_buy_input",
            placeholder="RELIANCE, TCS, VOO",
        )
        universal_sell_input = st.text_area(
            "Remove symbols from universal portfolio",
            key="universal_sell_input",
            placeholder="HDFCBANK, SBIN",
        )
        update_universal_btn = st.button(
            "Update universal portfolio",
            width="stretch",
            key="update_universal_btn",
        )

    with st.container(border=True):
        st.markdown("**Backup and restore**")
        st.caption(
            "Download this before a redeploy/reboot that might reset the app's disk, "
            "and restore it afterwards to bring the shared universe back."
        )
        universal_backup_col1, universal_backup_col2 = st.columns(2, gap="medium")
        with universal_backup_col1:
            universal_csv = universal_df.to_csv(index=False).encode("utf-8-sig")
            universal_backup_date = datetime.now().strftime("%Y-%m-%d")
            st.download_button(
                "Download universal portfolio CSV",
                data=universal_csv,
                file_name=f"universal_portfolio_backup_{universal_backup_date}.csv",
                mime="text/csv",
                width="stretch",
                key="download_universal_backup",
                disabled=universal_df.empty,
            )
        with universal_backup_col2:
            universal_backup_upload = st.file_uploader(
                "Upload universal portfolio backup CSV",
                type=["csv"],
                key="universal_backup_upload",
            )
        universal_restore_choice = st.radio(
            "Restore behaviour",
            options=["Merge/update universal portfolio", "Replace universal portfolio"],
            index=0,
            key="universal_restore_choice",
        )
        restore_universal_backup_btn = st.button(
            "Restore uploaded universal portfolio",
            width="stretch",
            key="restore_universal_backup_btn",
            disabled=universal_backup_upload is None,
        )

st.divider()

if restore_holdings_btn:
    try:
        restore_mode = (
            "replace"
            if holdings_restore_choice == "Replace current holdings"
            else "merge"
        )
        restored_count = restore_holdings_backup(
            holdings_backup_upload,
            CURRENT_USER,
            mode=restore_mode,
        )
        st.session_state["holdings_editor_version"] += 1
        clear_drop_bottom_coverage_preview()
        st.session_state.pop("drop_bottom_auto_result", None)
        st.session_state.pop("drop_bottom_auto_error", None)
        st.session_state["holdings_flash_success"] = (
            f"Restored {restored_count} holdings from the CSV backup using "
            f"{restore_mode} mode."
        )
        st.rerun()
    except Exception as exc:
        update_errors.append(f"Could not restore holdings backup: {exc}")

if import_broker_holdings_btn:
    try:
        broker_import_mode = (
            "replace"
            if broker_holdings_mode == "Replace current holdings"
            else "merge"
        )
        imported_count, unresolved_isins = import_broker_holdings_excel(
            broker_holdings_upload,
            CURRENT_USER,
            mode=broker_import_mode,
        )
        st.session_state["holdings_editor_version"] += 1
        clear_drop_bottom_coverage_preview()
        st.session_state.pop("drop_bottom_auto_result", None)
        st.session_state.pop("drop_bottom_auto_error", None)
        flash_message = (
            f"Imported {imported_count} holdings from the broker statement using "
            f"{broker_import_mode} mode."
        )
        if unresolved_isins:
            flash_message += (
                " Unresolved ISINs (skipped): " + ", ".join(unresolved_isins[:10])
            )
        st.session_state["holdings_flash_success"] = flash_message
        st.rerun()
    except Exception as exc:
        update_errors.append(f"Could not import broker holdings: {exc}")

if restore_analysis_btn:
    try:
        restored_payload = restore_latest_analysis_backup(analysis_backup_upload, CURRENT_USER)
        update_messages.append(
            "Analysis backup restored successfully. Saved at: "
            + str(restored_payload.get("saved_at", "Unknown"))
        )
    except Exception as exc:
        update_errors.append(f"Could not restore analysis backup: {exc}")

if update_holdings_btn:
    buy_symbols = parse_symbol_input(buy_input)
    sell_symbols = parse_symbol_input(sell_input)
    buy_keys = {normalize_portfolio_symbol(symbol) for symbol in buy_symbols}
    sell_keys = {normalize_portfolio_symbol(symbol) for symbol in sell_symbols}
    overlap = sorted(buy_keys & sell_keys)

    if overlap:
        update_errors.append(
            "The same symbol cannot be present in both Buy and Sell: " + ", ".join(overlap)
        )
    elif not buy_symbols and not sell_symbols:
        update_warnings.append("Enter at least one symbol in Buy or Sell.")
    else:
        removed, not_held = remove_symbols_from_master(sell_symbols, CURRENT_USER)
        added, duplicates, invalid, missing_initial_price = add_symbols_to_master(buy_symbols, CURRENT_USER)

        if added:
            update_messages.append("Added: " + ", ".join(added))
        if removed:
            update_messages.append("Removed: " + ", ".join(removed))
        if duplicates:
            update_warnings.append("Already in master table: " + ", ".join(duplicates))
        if not_held:
            update_warnings.append("Not present in master table: " + ", ".join(not_held))
        if invalid:
            update_errors.append("Could not resolve on Yahoo Finance/NSE/BSE: " + ", ".join(invalid))
        if missing_initial_price:
            update_warnings.append(
                "Added without an initial price; enter Average Price in the editor before analysis: "
                + ", ".join(missing_initial_price)
            )

        if added or removed:
            # The holdings row set changed, so use a fresh editor widget key and
            # invalidate previews that were calculated for the previous holdings set.
            st.session_state["holdings_editor_version"] += 1
            clear_drop_bottom_coverage_preview()
            st.session_state.pop("drop_bottom_auto_result", None)
            st.session_state.pop("drop_bottom_auto_error", None)

if update_universal_btn:
    universal_buy_symbols = parse_symbol_input(universal_buy_input)
    universal_sell_symbols = parse_symbol_input(universal_sell_input)
    universal_buy_keys = {normalize_portfolio_symbol(s) for s in universal_buy_symbols}
    universal_sell_keys = {normalize_portfolio_symbol(s) for s in universal_sell_symbols}
    universal_overlap = sorted(universal_buy_keys & universal_sell_keys)

    if universal_overlap:
        update_errors.append(
            "The same symbol cannot be present in both Add and Remove: " + ", ".join(universal_overlap)
        )
    elif not universal_buy_symbols and not universal_sell_symbols:
        update_warnings.append("Enter at least one symbol to add or remove from the universal portfolio.")
    else:
        universal_removed, universal_not_held = remove_symbols_from_master(universal_sell_symbols, UNIVERSAL_OWNER)
        universal_added, universal_duplicates, universal_invalid = add_symbols_to_universal(universal_buy_symbols)

        if universal_added:
            update_messages.append("Added to universal portfolio: " + ", ".join(universal_added))
        if universal_removed:
            update_messages.append("Removed from universal portfolio: " + ", ".join(universal_removed))
        if universal_duplicates:
            update_warnings.append("Already in universal portfolio: " + ", ".join(universal_duplicates))
        if universal_not_held:
            update_warnings.append("Not present in universal portfolio: " + ", ".join(universal_not_held))
        if universal_invalid:
            update_errors.append("Could not resolve on Yahoo Finance/NSE/BSE: " + ", ".join(universal_invalid))

if restore_universal_backup_btn:
    try:
        universal_restore_mode = (
            "replace"
            if universal_restore_choice == "Replace universal portfolio"
            else "merge"
        )
        restored_universal_count = restore_holdings_backup(
            universal_backup_upload,
            UNIVERSAL_OWNER,
            mode=universal_restore_mode,
        )
        update_messages.append(
            f"Restored {restored_universal_count} symbols into the universal portfolio "
            f"using {universal_restore_mode} mode."
        )
    except Exception as exc:
        update_errors.append(f"Could not restore universal portfolio backup: {exc}")

for message in update_messages:
    st.success(message)
for message in update_warnings:
    st.warning(message)
for message in update_errors:
    st.error(message)

# Fresh, uncached database count on every Streamlit rerun.
live_unique_count = render_live_holdings_banner(live_count_banner_placeholder, CURRENT_USER)
sidebar_count_placeholder.metric("Current unique holdings", live_unique_count)
render_saved_analysis(saved_analysis_placeholder, CURRENT_USER)

st.subheader("Master Holdings")
master_df = load_master_holdings(CURRENT_USER)

if master_df.empty:
    st.info("The master holdings table is empty. Add symbols from the sidebar.")
else:
    st.dataframe(master_df, width="stretch", hide_index=True)

    with st.expander("Edit quantity and average price", expanded=False):
        st.caption(
            "A newly added symbol starts with quantity 1 and the latest available native-market price. "
            "After you save an edit, Quantity and Average Price are reloaded from SQLite."
        )
        editable_df = master_df[
            ["Symbol", "Stock Name", "Yahoo Ticker", "Exchange", "Currency", "Quantity", "Average Price"]
        ].copy()

        # Sort only the editable table by invested value (Quantity × Average Price).
        # The temporary sort column is removed before rendering, so no new column
        # is displayed to the user.
        editable_df["_sort_value"] = (
            pd.to_numeric(editable_df["Quantity"], errors="coerce").fillna(0)
            * pd.to_numeric(editable_df["Average Price"], errors="coerce").fillna(0)
        )
        editable_df = (
            editable_df.sort_values(
                "_sort_value",
                ascending=False,
                kind="stable",
            )
            .drop(columns="_sort_value")
            .reset_index(drop=True)
        )

        editor_version = st.session_state["holdings_editor_version"]

        with st.form(
            key=f"holdings_edit_form_{editor_version}",
            clear_on_submit=False,
        ):
            edited_df = st.data_editor(
                editable_df,
                width="stretch",
                hide_index=True,
                disabled=["Symbol", "Stock Name", "Yahoo Ticker", "Exchange", "Currency"],
                column_config={
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity",
                        min_value=0.000001,
                        format="%.6f",
                    ),
                    "Average Price": st.column_config.NumberColumn(
                        "Average Price",
                        min_value=0.01,
                        format="%.2f",
                    ),
                },
                key=f"holdings_editor_{editor_version}",
            )
            save_holdings_btn = st.form_submit_button(
                "Save quantity and price changes",
                width="stretch",
                type="primary",
            )

        if save_holdings_btn:
            try:
                updated_count = save_holding_values(edited_df, CURRENT_USER)
                # A new key forces Streamlit to discard the old editor snapshot.
                st.session_state["holdings_editor_version"] += 1
                clear_drop_bottom_coverage_preview()
                st.session_state.pop("drop_bottom_auto_result", None)
                st.session_state.pop("drop_bottom_auto_error", None)
                st.session_state["holdings_flash_success"] = (
                    f"Saved Quantity and Average Price for {updated_count} holdings."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save holdings: {exc}")


if run_btn:
    try:
        if master_df.empty:
            st.error("Add at least one symbol before running the analysis.")
            st.stop()

        with st.spinner("Loading holdings from the database..."):
            portfolio_df, invalid_holding_rows = build_current_allocation_from_db(CURRENT_USER)

        if portfolio_df.empty:
            st.error(
                "No usable holdings were found. Add valid Average Price and Quantity values in the master table."
            )
            st.stop()

        if invalid_holding_rows:
            st.warning(
                "Skipped holdings with missing or invalid quantity/price: "
                + ", ".join(invalid_holding_rows)
            )

        with st.spinner("Adding Universal Portfolio symbols as buy candidates..."):
            portfolio_df, universal_candidate_symbols = extend_allocation_with_universal_candidates(portfolio_df)
        if universal_candidate_symbols:
            st.info(
                "Universal Portfolio candidates included in this optimization (quantity 0): "
                + ", ".join(universal_candidate_symbols)
            )

        col1, col2 = st.columns([2, 1], gap="medium")

        with col1:
            st.subheader("Current Allocation")
            st.dataframe(
                portfolio_df.style.format({
                    "Quantity": "{:,.4f}",
                    "Average Price": "{:,.2f}",
                    "Latest Price": "{:,.2f}",
                    "FX to INR": "{:,.4f}",
                    "Average Price INR": "₹{:,.2f}",
                    "Latest Price INR": "₹{:,.2f}",
                    "Invested INR": "₹{:,.2f}",
                    "Current Value INR": "₹{:,.2f}",
                    "Weight": "{:.2%}",
                }),
                width="stretch",
            )

        with col2:
            total_invested = float(portfolio_df["Invested INR"].sum())
            total_current_value = float(portfolio_df["Current Value INR"].sum())
            st.metric("Holdings in database", int((portfolio_df["Quantity"] > 0).sum()))
            st.metric("Cost basis (INR @ current FX)", f"₹{total_invested:,.2f}")
            st.metric("Current portfolio value", f"₹{total_current_value:,.2f}")

        yahoo_tickers = portfolio_df["Yahoo Ticker"].dropna().astype(str).tolist()
        unresolved = []
        if not yahoo_tickers:
            st.error("No valid Yahoo tickers resolved.")
            st.stop()

        with st.spinner("Running optimization..."):
            optimal_weights, log_returns, current_stats, optimal_stats, meta = run_portfolio_analysis_multi(
                yahoo_tickers,
                portfolio_df.copy(),
                max_dd=max_dd,
                target_volatility=target_volatility,
                drop_bottom_pct=drop_bottom_pct,
            )

        if optimal_weights is None:
            st.error("Portfolio optimization did not return a usable allocation.")
            st.stop()

        if not meta["dropped_df"].empty:
            st.subheader("Dropped Tickers")
            st.dataframe(meta["dropped_df"], width="stretch")

        st.subheader("History Coverage")
        st.dataframe(meta["min_len_df"], width="stretch")
        st.info(
            f"**drop_bottom_pct used:** `{drop_bottom_pct:.2f}`  |  "
            f"**Trading days analysed:** {int(log_returns.shape[0]):,}  |  "
            f"**Assets in return matrix:** {int(log_returns.shape[1]):,}"
        )

        missing_history_tickers = meta.get("missing_history_tickers") or []
        if missing_history_tickers:
            st.warning(
                "Skipped symbols without usable history from Yahoo Finance: "
                + ", ".join(missing_history_tickers)
            )

        st.caption(f"Overlapping date range: {meta['valid_start'].date()} to {meta['valid_end'].date()}")
        st.caption(f"Log return shape: {log_returns.shape} — foreign assets are FX-adjusted to INR before return calculation.")

        if current_stats and optimal_stats:
            st.subheader("Portfolio Stats Comparison")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Current Portfolio**")
                current_df = metrics_df(current_stats)
                current_df["Value"] = current_df.apply(
                    lambda r: f"{r['Value']:.2%}" if r["Metric"] != "Sharpe Ratio" else f"{r['Value']:.2f}",
                    axis=1,
                )
                st.dataframe(current_df, width="stretch")

            with c2:
                st.markdown("**Optimized Portfolio**")
                optimal_df = metrics_df(optimal_stats)
                optimal_df["Value"] = optimal_df.apply(
                    lambda r: f"{r['Value']:.2%}" if r["Metric"] != "Sharpe Ratio" else f"{r['Value']:.2f}",
                    axis=1,
                )
                st.dataframe(optimal_df, width="stretch")

        st.subheader("Top Correlated Pairs")

        top_corrs = pd.DataFrame(
            columns=["Ticker 1", "Ticker 2", "Abs Correlation"]
        )
        corr_matrix = log_returns.corr()

        if corr_matrix.shape[1] < 2:
            st.info("Need at least 2 stocks to show correlated pairs.")
        else:
            upper_mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
            corr_only = corr_matrix.where(upper_mask)
            corr_unstacked = corr_only.stack()

            if corr_unstacked.empty:
                st.info("No valid correlated pairs found.")
            else:
                top_corrs = pd.DataFrame(
                    [(idx[0], idx[1], abs(val)) for idx, val in corr_unstacked.items()],
                    columns=["Ticker 1", "Ticker 2", "Abs Correlation"],
                ).sort_values("Abs Correlation", ascending=False)

                st.dataframe(top_corrs.head(5), width="stretch")

        with st.spinner("Fetching latest prices for the rebalancing plan..."):
            latest_prices = log_returns.columns.tolist()
            price_map = get_latest_price_map(latest_prices)

        rebal_df, missing_prices, missing_alloc = rebalance_plan_multi(
            portfolio_df.copy(),
            optimal_weights,
            log_returns,
            price_map,
            days_to_flip,
        )

        if missing_prices:
            st.warning(f"Skipped symbols with missing latest price: {', '.join(missing_prices)}")
        if missing_alloc:
            st.warning(f"Skipped symbols missing in allocation: {', '.join(missing_alloc)}")

        if lumpsum_inr > 0:
            lumpsum_df, unallocated_cash, lumpsum_missing_prices, lumpsum_missing_alloc = (
                lumpsum_allocation_plan(
                    portfolio_df.copy(),
                    optimal_weights,
                    log_returns,
                    price_map,
                    lumpsum_inr,
                )
            )
            st.subheader("Lumpsum Optimal Allocation")
            st.caption(
                f"Lumpsum: ₹{lumpsum_inr:,.2f} | "
                f"Estimated investment: ₹{lumpsum_df['Estimated Investment INR'].sum():,.2f} | "
                f"Unallocated cash: ₹{unallocated_cash:,.2f}"
            )
            if lumpsum_missing_prices:
                st.warning(
                    "Lumpsum plan skipped symbols with missing latest price: "
                    + ", ".join(lumpsum_missing_prices)
                )
            if lumpsum_missing_alloc:
                st.warning(
                    "Lumpsum plan skipped symbols missing in allocation: "
                    + ", ".join(lumpsum_missing_alloc)
                )
            with lumpsum_download_placeholder.container():
                st.divider()
                st.subheader("Lumpsum order download")
                st.download_button(
                    "Download compact execution sheet (HTML)",
                    data=lumpsum_execution_sheet_html(
                        lumpsum_df,
                        lumpsum_inr,
                        unallocated_cash,
                    ).encode("utf-8"),
                    file_name="lumpsum_buy_orders.html",
                    mime="text/html",
                    width="stretch",
                    key="download_lumpsum_execution_sheet_sidebar",
                )
            st.dataframe(
                lumpsum_df.style.format({
                    "Optimal Weight": "{:.2%}",
                    "Target Amount INR": "₹{:,.2f}",
                    "Latest Price INR": "₹{:,.2f}",
                    "Suggested Quantity": "{:,.0f}",
                    "Estimated Investment INR": "₹{:,.2f}",
                }),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download lumpsum allocation CSV",
                data=lumpsum_df.to_csv(index=False).encode("utf-8"),
                file_name="lumpsum_optimal_allocation.csv",
                mime="text/csv",
                width="stretch",
            )
            st.download_button(
                "Download compact execution sheet (HTML)",
                data=lumpsum_execution_sheet_html(
                    lumpsum_df,
                    lumpsum_inr,
                    unallocated_cash,
                ).encode("utf-8"),
                file_name="lumpsum_buy_orders.html",
                mime="text/html",
                width="stretch",
            )

        st.subheader("Rebalancing Plan")

        if rebal_df.empty:
            st.success("No trades to execute after filtering Executable Quantity = 0")
        else:
            ordered_rebal_df = _sort_rebalance_df_for_priority(rebal_df)
            st.dataframe(style_rebalance_df(ordered_rebal_df), width="stretch")

            csv = ordered_rebal_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download rebalancing plan CSV",
                data=csv,
                file_name="rebalancing_plan.csv",
                mime="text/csv",
                width="stretch",
            )

        analysis_payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "holdings_analyzed": int((portfolio_df["Quantity"] > 0).sum()),
            "total_invested": float(total_invested),
            "executable_trade_count": int(len(rebal_df)),
            "settings": {
                "days_to_flip": int(days_to_flip),
                "max_drawdown_input_pct": float(max_dd_pct),
                "internal_max_dd": float(max_dd),
                "drop_bottom_fraction": float(drop_bottom_pct),
                "use_target_volatility": bool(use_target_vol),
                "target_volatility": (
                    float(target_volatility)
                    if target_volatility is not None
                    else None
                ),
            },
            "history": {
                "valid_start": str(meta["valid_start"]),
                "valid_end": str(meta["valid_end"]),
                "log_return_rows": int(log_returns.shape[0]),
                "log_return_columns": int(log_returns.shape[1]),
                "minimum_history": (
                    meta["min_len_df"].to_dict(orient="records")
                    if not meta["min_len_df"].empty
                    else []
                ),
                "dropped_tickers": (
                    meta["dropped_df"].to_dict(orient="records")
                    if not meta["dropped_df"].empty
                    else []
                ),
            },
            "current_stats": current_stats or {},
            "optimal_stats": optimal_stats or {},
            "top_correlations": (
                top_corrs.head(5).to_dict(orient="records")
                if not top_corrs.empty
                else []
            ),
            "current_allocation": portfolio_df.to_dict(orient="records"),
            "rebalancing_plan": rebal_df.to_dict(orient="records"),
            "warnings": {
                "invalid_holding_rows": invalid_holding_rows,
                "unresolved_yahoo_tickers": unresolved,
                "missing_history_tickers": meta.get("missing_history_tickers", []),
                "missing_latest_prices": missing_prices,
                "missing_allocation": missing_alloc,
            },
        }

        save_latest_analysis(analysis_payload, CURRENT_USER)
        render_saved_analysis(saved_analysis_placeholder, CURRENT_USER)
        st.success(
            "Analysis completed and saved. You can now download the complete "
            "analysis backup as JSON."
        )

    except Exception as e:
        st.error(f"Error: {e}")
