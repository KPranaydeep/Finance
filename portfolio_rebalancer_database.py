import io
import json
import os
import re
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Portfolio Rebalancer", layout="wide")

APP_BUILD = "2026-06-29-auto-252-control-v1"

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
def resolve_yahoo_tickers(symbols_base):
    resolved = {}
    for sym in symbols_base:
        for suffix in [".NS", ".BO"]:
            try:
                test = yf.download(sym + suffix, period="5d", progress=False, auto_adjust=True)
                if test is not None and not test.empty:
                    resolved[sym] = sym + suffix
                    break
            except Exception:
                continue
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

MASTER_HOLDINGS_DDL = """
CREATE TABLE IF NOT EXISTS master_holdings (
    symbol TEXT PRIMARY KEY,
    stock_name TEXT NOT NULL,
    quantity REAL NOT NULL DEFAULT 1,
    average_price REAL,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

LATEST_ANALYSIS_DDL = """
CREATE TABLE IF NOT EXISTS latest_analysis (
    id INTEGER PRIMARY KEY CHECK (id = 1),
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

    latest_columns = {
        str(row["name"]).lower()
        for row in conn.execute("PRAGMA table_info(latest_analysis)").fetchall()
    }
    required_latest_columns = {"id", "saved_at", "payload_json"}
    if not required_latest_columns.issubset(latest_columns):
        legacy_latest_name = (
            "latest_analysis_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        conn.execute(
            f'ALTER TABLE latest_analysis RENAME TO "{legacy_latest_name}"'
        )
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
                legacy_row = conn.execute(
                    f'SELECT "{time_column}", payload_json '
                    f'FROM "{legacy_latest_name}" LIMIT 1'
                ).fetchone()
                if legacy_row is not None:
                    conn.execute(
                        """
                        INSERT INTO latest_analysis (id, saved_at, payload_json)
                        VALUES (1, ?, ?)
                        """,
                        (str(legacy_row[0]), legacy_row[1]),
                    )

    table_info = conn.execute("PRAGMA table_info(master_holdings)").fetchall()
    columns = {str(row["name"]).lower() for row in table_info}

    # A table with this name but no symbol column belongs to an incompatible old
    # schema. Preserve it rather than deleting user data, then create a clean table.
    if "symbol" not in columns:
        legacy_name = "master_holdings_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        conn.execute(f'ALTER TABLE master_holdings RENAME TO "{legacy_name}"')
        conn.execute(MASTER_HOLDINGS_DDL)
        columns = {
            str(row["name"]).lower()
            for row in conn.execute("PRAGMA table_info(master_holdings)").fetchall()
        }

    # IF NOT EXISTS does not add columns to a table created by an older app version.
    migrations = {
        "stock_name": "ALTER TABLE master_holdings ADD COLUMN stock_name TEXT",
        "quantity": "ALTER TABLE master_holdings ADD COLUMN quantity REAL DEFAULT 1",
        "average_price": "ALTER TABLE master_holdings ADD COLUMN average_price REAL",
        "added_at": "ALTER TABLE master_holdings ADD COLUMN added_at TEXT",
        "updated_at": "ALTER TABLE master_holdings ADD COLUMN updated_at TEXT",
    }
    for column, sql in migrations.items():
        if column not in columns:
            conn.execute(sql)

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        """
        UPDATE master_holdings
        SET stock_name = COALESCE(NULLIF(TRIM(stock_name), ''), symbol),
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
    symbol = str(value or "").strip().upper()
    if symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol


def resolve_nse_symbol(symbol, available_symbols, allow_be_fallback=False):
    """Resolve broker-style NSE symbols against canonical NSE symbols.

    The broker may append the trading-series suffix ``-BE`` to the NSE symbol.
    We first try the exact value. When it ends in ``-BE``, we retry with the
    suffix removed. For additions, ``allow_be_fallback=True`` permits the
    trimmed base symbol even when the external equity mapping is stale or
    incomplete; Yahoo price lookup is then used by the normal add workflow.
    """
    normalized = normalize_nse_symbol(symbol)
    if normalized in available_symbols:
        return normalized

    if normalized.endswith("-BE"):
        base_symbol = normalized[:-3].rstrip("-").strip()
        if base_symbol and (base_symbol in available_symbols or allow_be_fallback):
            return base_symbol

    return None


def parse_symbol_input(raw_text):
    parts = re.split(r"[,;\n]+", raw_text or "")
    symbols = []
    seen = set()

    for part in parts:
        symbol = normalize_nse_symbol(part)
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)

    return symbols


def get_nse_company_lookup():
    equity_map = load_equity_mapping().copy()
    equity_map["Symbol"] = equity_map["Symbol"].astype(str).str.strip().str.upper()
    equity_map["Company Name"] = equity_map["Company Name"].astype(str).str.strip()
    return dict(zip(equity_map["Symbol"], equity_map["Company Name"]))


def load_master_holdings():
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                symbol AS Symbol,
                stock_name AS "Stock Name",
                quantity AS Quantity,
                average_price AS "Average Price",
                added_at AS "Added At",
                updated_at AS "Updated At"
            FROM master_holdings
            ORDER BY symbol
            """,
            conn,
        )
    return df



def holdings_backup_bytes():
    """Export the complete holdings master table as a UTF-8 CSV backup."""
    holdings = load_master_holdings()
    return holdings.to_csv(index=False).encode("utf-8-sig")


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

    # Allow either the exported display headings or direct SQLite column names.
    aliases = {
        "symbol": "Symbol",
        "stock name": "Stock Name",
        "stock_name": "Stock Name",
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

    if "Stock Name" not in backup_df.columns:
        backup_df["Stock Name"] = backup_df["Symbol"]
    if "Added At" not in backup_df.columns:
        backup_df["Added At"] = None
    if "Updated At" not in backup_df.columns:
        backup_df["Updated At"] = None

    cleaned = backup_df[
        ["Symbol", "Stock Name", "Quantity", "Average Price", "Added At", "Updated At"]
    ].copy()
    cleaned["Symbol"] = cleaned["Symbol"].map(normalize_nse_symbol)
    cleaned["Stock Name"] = cleaned["Stock Name"].fillna("").astype(str).str.strip()
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

    cleaned["Stock Name"] = cleaned.apply(
        lambda row: row["Stock Name"] or row["Symbol"], axis=1
    )
    return cleaned


def restore_holdings_backup(uploaded_file, mode="merge"):
    """Restore holdings from CSV using replace or merge/update semantics."""
    cleaned = _read_holdings_backup(uploaded_file)
    normalized_mode = str(mode or "merge").strip().lower()
    if normalized_mode not in {"replace", "merge"}:
        raise ValueError("Restore mode must be either 'replace' or 'merge'.")

    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for _, row in cleaned.iterrows():
        added_at = str(row["Added At"]).strip() if pd.notna(row["Added At"]) else now
        updated_at = (
            str(row["Updated At"]).strip() if pd.notna(row["Updated At"]) else now
        )
        records.append(
            (
                row["Symbol"],
                row["Stock Name"],
                float(row["Quantity"]),
                (
                    float(row["Average Price"])
                    if pd.notna(row["Average Price"])
                    else None
                ),
                added_at or now,
                updated_at or now,
            )
        )

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            if normalized_mode == "replace":
                conn.execute("DELETE FROM master_holdings")

            conn.executemany(
                """
                INSERT INTO master_holdings
                    (symbol, stock_name, quantity, average_price, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    stock_name = excluded.stock_name,
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


def get_unique_holdings_count():
    """Return the current distinct NSE-symbol count directly from SQLite."""
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT UPPER(TRIM(symbol))) AS unique_count
            FROM master_holdings
            WHERE symbol IS NOT NULL
              AND TRIM(symbol) <> ''
            """
        ).fetchone()
    return int(row["unique_count"] or 0)


def render_live_holdings_banner(placeholder):
    """Render the live count without taking down the whole app on a DB error."""
    try:
        unique_count = get_unique_holdings_count()
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


def save_latest_analysis(payload):
    """Persist the latest successful analysis as one singleton SQLite record."""
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
            INSERT INTO latest_analysis (id, saved_at, payload_json)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                saved_at = excluded.saved_at,
                payload_json = excluded.payload_json
            """,
            (saved_at, payload_json),
        )
        conn.commit()

    return safe_payload


def load_latest_analysis():
    """Load the latest saved analysis without preventing the app from opening."""
    try:
        with get_db_connection() as conn:
            conn.execute(LATEST_ANALYSIS_DDL)
            row = conn.execute(
                "SELECT payload_json FROM latest_analysis WHERE id = 1"
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


def latest_analysis_backup_bytes():
    """Return the latest saved analysis as UTF-8 JSON bytes, when available."""
    payload = load_latest_analysis()
    if payload is None:
        return None

    return json.dumps(
        make_json_safe(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def restore_latest_analysis_backup(uploaded_file):
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

    return save_latest_analysis(payload)


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


def render_saved_analysis(placeholder):
    """Display the latest saved analysis immediately below the live banner."""
    placeholder.empty()

    with placeholder.container():
        payload = load_latest_analysis()

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
            use_container_width=True,
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
                            use_container_width=True,
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
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption("No optimized-portfolio statistics were saved.")

        with st.expander("Saved rebalancing plan", expanded=True):
            if rebalancing_plan:
                saved_rebal_df = pd.DataFrame(rebalancing_plan)

                # Sort only the saved rebalancing-plan display by Optimal Weight
                # in descending order. The temporary sort key is removed before
                # rendering, so no additional column is shown to the user.
                if "Optimal Weight" in saved_rebal_df.columns:
                    optimal_weight_text = (
                        saved_rebal_df["Optimal Weight"]
                        .astype(str)
                        .str.strip()
                    )
                    optimal_weight_is_percent = optimal_weight_text.str.endswith("%")
                    optimal_weight_sort = pd.to_numeric(
                        optimal_weight_text.str.rstrip("%"),
                        errors="coerce",
                    )
                    optimal_weight_sort = optimal_weight_sort.where(
                        ~optimal_weight_is_percent,
                        optimal_weight_sort / 100.0,
                    )
                    saved_rebal_df = (
                        saved_rebal_df.assign(
                            _optimal_weight_sort=optimal_weight_sort
                        )
                        .sort_values(
                            "_optimal_weight_sort",
                            ascending=False,
                            na_position="last",
                            kind="stable",
                        )
                        .drop(columns="_optimal_weight_sort")
                        .reset_index(drop=True)
                    )

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
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        saved_rebal_df,
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.success("The saved analysis had no executable trades.")


def add_symbols_to_master(symbols):
    if not symbols:
        return [], [], [], []

    company_lookup = get_nse_company_lookup()
    available_symbols = set(company_lookup)

    valid_symbols = []
    invalid_symbols = []
    seen_resolved = set()

    for entered_symbol in symbols:
        resolved_symbol = resolve_nse_symbol(
            entered_symbol, available_symbols, allow_be_fallback=True
        )
        if resolved_symbol is None:
            invalid_symbols.append(entered_symbol)
        elif resolved_symbol not in seen_resolved:
            valid_symbols.append(resolved_symbol)
            seen_resolved.add(resolved_symbol)

    with get_db_connection() as conn:
        existing = {
            row["symbol"]
            for row in conn.execute(
                "SELECT symbol FROM master_holdings WHERE symbol IN ({})".format(
                    ",".join("?" for _ in valid_symbols)
                ),
                valid_symbols,
            ).fetchall()
        } if valid_symbols else set()

    duplicates = [s for s in valid_symbols if s in existing]
    new_symbols = [s for s in valid_symbols if s not in existing]

    price_map = {}
    if new_symbols:
        try:
            price_map = get_latest_price_map([f"{s}.NS" for s in new_symbols])
        except Exception:
            price_map = {}

    now = datetime.now().isoformat(timespec="seconds")
    added = []
    missing_initial_price = []

    with get_db_connection() as conn:
        for symbol in new_symbols:
            initial_price = price_map.get(symbol)
            if initial_price is None or not np.isfinite(initial_price) or initial_price <= 0:
                initial_price = None
                missing_initial_price.append(symbol)

            conn.execute(
                """
                INSERT INTO master_holdings
                    (symbol, stock_name, quantity, average_price, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    company_lookup.get(symbol, symbol),
                    1.0,
                    initial_price,
                    now,
                    now,
                ),
            )
            added.append(symbol)
        conn.commit()

    return added, duplicates, invalid_symbols, missing_initial_price


def remove_symbols_from_master(symbols):
    if not symbols:
        return [], []

    with get_db_connection() as conn:
        all_existing = {
            row["symbol"]
            for row in conn.execute("SELECT symbol FROM master_holdings").fetchall()
        }

        resolved_to_remove = []
        missing = []
        seen_resolved = set()

        for entered_symbol in symbols:
            resolved_symbol = resolve_nse_symbol(entered_symbol, all_existing)
            if resolved_symbol is None:
                missing.append(entered_symbol)
            elif resolved_symbol not in seen_resolved:
                resolved_to_remove.append(resolved_symbol)
                seen_resolved.add(resolved_symbol)

        if resolved_to_remove:
            conn.execute(
                "DELETE FROM master_holdings WHERE symbol IN ({})".format(
                    ",".join("?" for _ in resolved_to_remove)
                ),
                resolved_to_remove,
            )
            conn.commit()

    return resolved_to_remove, missing


def save_holding_values(edited_df):
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
                    f"SELECT symbol FROM master_holdings WHERE symbol IN ({placeholders})",
                    symbols,
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
                    WHERE symbol = ?
                    """,
                    (
                        float(row["Quantity"]),
                        float(row["Average Price"]),
                        now,
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


def build_current_allocation_from_db():
    df = load_master_holdings()
    if df.empty:
        return pd.DataFrame(), []

    df["Symbol"] = df["Symbol"].map(normalize_nse_symbol)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Price"] = pd.to_numeric(df["Average Price"], errors="coerce")

    needs_price = df[
        df["Average Price"].isna()
        | ~np.isfinite(df["Average Price"])
        | (df["Average Price"] <= 0)
    ]["Symbol"].tolist()

    price_map = {}
    if needs_price:
        try:
            price_map = get_latest_price_map([f"{s}.NS" for s in needs_price])
        except Exception:
            price_map = {}

        now = datetime.now().isoformat(timespec="seconds")
        with get_db_connection() as conn:
            for symbol in needs_price:
                price = price_map.get(symbol)
                if price is not None and np.isfinite(price) and price > 0:
                    conn.execute(
                        """
                        UPDATE master_holdings
                        SET average_price = ?, updated_at = ?
                        WHERE symbol = ?
                        """,
                        (float(price), now, symbol),
                    )
            conn.commit()

        df = load_master_holdings()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Average Price"] = pd.to_numeric(df["Average Price"], errors="coerce")

    invalid_rows = df[
        df["Quantity"].isna()
        | (df["Quantity"] <= 0)
        | df["Average Price"].isna()
        | (df["Average Price"] <= 0)
    ]["Symbol"].tolist()

    usable = df[~df["Symbol"].isin(invalid_rows)].copy()
    if usable.empty:
        return pd.DataFrame(), invalid_rows

    usable["Invested"] = usable["Quantity"] * usable["Average Price"]
    total = usable["Invested"].sum()
    if total <= 0:
        raise ValueError("The master holdings table has no positive portfolio value.")

    usable["Weight"] = usable["Invested"] / total
    portfolio_df = usable[
        ["Symbol", "Stock Name", "Quantity", "Average Price", "Weight"]
    ].sort_values("Weight", ascending=False).reset_index(drop=True)

    return portfolio_df, invalid_rows

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

    prices = yf.download(
        list(symbols),
        start=start_date,
        end=effective_end,
        progress=False,
        auto_adjust=True,
    )["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError("No data available for the given tickers.")

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
def get_daily_log_returns(symbols, start_date=None, end_date=None, buffer_days=7, drop_bottom_pct=0.1):
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = pd.to_datetime(end_date)

    if start_date is None:
        start_date = "2000-01-01"

    df = download_close_history(
        tuple(symbols),
        start_date=start_date,
        end_date=end_date,
        buffer_days=buffer_days,
    ).copy()

    if df.empty:
        raise ValueError("No data available for the given tickers.")

    lengths = df.count().sort_values(ascending=False)
    num_to_drop = int(np.floor(drop_bottom_pct * len(lengths)))

    dropped_df = pd.DataFrame()
    if num_to_drop > 0:
        dropped = lengths.tail(num_to_drop)
        kept = lengths.head(len(lengths) - num_to_drop)
        df = df[kept.index]

        dropped_df = pd.DataFrame({
            "Ticker": dropped.index,
            "Valid Days of Data": dropped.values
        }).reset_index(drop=True)
    else:
        kept = lengths

    valid_start = df[kept.index].apply(lambda x: x.first_valid_index()).max()
    valid_end = df[kept.index].apply(lambda x: x.last_valid_index()).min()

    if valid_start is None or valid_end is None or valid_start >= valid_end:
        raise ValueError("No overlapping date range found across tickers after filtering.")

    df_aligned = df.loc[valid_start:valid_end].dropna(axis=1, how="any")
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
        "P&L (simulated)": [f"{simulated_pnl:.2f}%" if not np.isnan(simulated_pnl) else "N/A"]
    })

    meta = {
        "valid_start": valid_start,
        "valid_end": valid_end,
        "dropped_df": dropped_df,
        "min_len_df": min_len_df
    }
    return log_returns, meta


def optimize_portfolio_max_return_given_daily_risk(log_returns, max_drawdown=0.1):
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
    lr_symbols = [col.replace(".NS", "").replace(".BO", "") for col in log_returns.columns]

    aligned = current_alloc.set_index("Symbol").reindex(lr_symbols).dropna(subset=["Weight"])
    current_weights = aligned["Weight"].values
    current_weights = current_weights / current_weights.sum()

    current_log_returns = log_returns[
        [c for c in log_returns.columns if c.replace(".NS", "").replace(".BO", "") in aligned.index]
    ]
    current_stats = portfolio_stats(current_weights, current_log_returns)
    optimal_stats = portfolio_stats(optimal_weights, log_returns)

    return current_stats, optimal_stats


def run_portfolio_analysis_multi(symbols, current_alloc, max_dd=0.05, target_volatility=None, drop_bottom_pct=0.1):
    log_returns, meta = get_daily_log_returns(symbols, drop_bottom_pct=drop_bottom_pct)

    if target_volatility is not None:
        optimal_weights = optimize_portfolio_target_volatility(log_returns, target_volatility=target_volatility)
    else:
        optimal_weights = None

    if optimal_weights is None:
        optimal_weights = optimize_portfolio_max_return_given_daily_risk(log_returns, max_drawdown=max_dd)

    if optimal_weights is None:
        optimal_weights = optimize_max_sharpe_ratio(log_returns)

    if optimal_weights is None:
        return None, log_returns, None, None, meta

    current_stats, optimal_stats = portfolio_stats_comparison(current_alloc, log_returns, optimal_weights)
    return optimal_weights, log_returns, current_stats, optimal_stats, meta


# =========================================================
# REBALANCING
# =========================================================

def rebalance_plan_multi(current_alloc, optimal_weights, log_returns, prices, days_to_flip):
    lr_symbols = [col.replace(".NS", "").replace(".BO", "") for col in log_returns.columns]
    alloc_df = current_alloc.set_index("Symbol").copy()

    common_symbols = [s for s in lr_symbols if s in alloc_df.index and s in prices]

    missing_prices = [s for s in lr_symbols if s not in prices]
    missing_alloc = [s for s in lr_symbols if s not in alloc_df.index]

    if not common_symbols:
        raise ValueError("No common symbols between returns, allocation, and prices.")

    pos = [lr_symbols.index(s) for s in common_symbols]
    aligned_optimal_weights = np.array(optimal_weights)[pos]
    aligned_optimal_weights = aligned_optimal_weights / aligned_optimal_weights.sum()

    current_values = np.array(
        [alloc_df.loc[s, "Quantity"] * prices[s] for s in common_symbols],
        dtype=float
    )
    portfolio_value = current_values.sum()
    current_weights = current_values / portfolio_value

    change_weights = aligned_optimal_weights - current_weights
    value_change = portfolio_value * change_weights
    qty_change = np.array([vc / prices[s] for vc, s in zip(value_change, common_symbols)])

    action = np.where(change_weights > 0, "Buy", "Sell")
    abs_qty_change = np.abs(qty_change)

    exec_qty = np.where(
        action == "Sell",
        np.maximum(0, np.floor(abs_qty_change).astype(int) - 1),
        np.floor(abs_qty_change / days_to_flip).astype(int)
    )

    exec_val = np.array([eq * prices[s] for eq, s in zip(exec_qty, common_symbols)], dtype=float)

    rebal_df = pd.DataFrame({
        "Symbol": common_symbols,
        "Current Weight": current_weights,
        "Optimal Weight": aligned_optimal_weights,
        "Action": action,
        "Change": np.abs(change_weights),
        "Quantity": abs_qty_change,
        "Executable Quantity": exec_qty,
        "Executable Value": exec_val,
    })

    rebal_df = (
        rebal_df[rebal_df["Executable Quantity"] != 0]
        .sort_values(by="Executable Value", ascending=False)
        .reset_index(drop=True)
    )

    return rebal_df, missing_prices, missing_alloc


@st.cache_data(show_spinner=False)
def get_latest_price_map(latest_prices):
    price_history = yf.download(latest_prices, period="15d", progress=False, auto_adjust=True)["Close"]

    if isinstance(price_history, pd.Series):
        price_history = price_history.to_frame()

    price_history = price_history.ffill()
    last_row = price_history.iloc[-1]

    price_map = {
        col.replace(".NS", "").replace(".BO", ""): float(last_row[col])
        for col in price_history.columns
        if pd.notna(last_row[col])
    }
    return price_map


def style_rebalance_df(df):
    def color_action_row(row):
        if row["Action"] == "Buy":
            return ["background-color: #d4edda; color: #155724;"] * len(row)
        return ["background-color: #f8d7da; color: #721c24;"] * len(row)

    return (
        df.style
        .apply(color_action_row, axis=1)
        .format({
            "Current Weight": "{:.2%}",
            "Optimal Weight": "{:.2%}",
            "Change": "{:.2%}",
            "Quantity": "{:.0f}",
            "Executable Quantity": "{:.0f}",
            "Executable Value": "₹{:,.0f}",
        })
    )


def metrics_df(stats_dict):
    return pd.DataFrame({
        "Metric": list(stats_dict.keys()),
        "Value": list(stats_dict.values())
    })


def auto_set_drop_bottom_pct_for_analysis():
    """Set the visible control to the value nearest to 252+ trading days."""
    st.session_state.pop("drop_bottom_auto_result", None)
    st.session_state.pop("drop_bottom_auto_error", None)

    try:
        portfolio_df, _ = build_current_allocation_from_db()

        if portfolio_df.empty:
            raise ValueError("No usable holdings are available.")

        symbols = (
            portfolio_df["Symbol"]
            .dropna()
            .astype(str)
            .str.upper()
            .tolist()
        )

        resolved_map = resolve_yahoo_tickers(symbols)
        yahoo_tickers = tuple(resolved_map.values())

        if not yahoo_tickers:
            raise ValueError("No Yahoo Finance tickers could be resolved.")

        selected = find_drop_bottom_pct_nearest_target(
            yahoo_tickers,
            target_trading_days=252,
        )

        st.session_state["drop_bottom_pct"] = float(
            selected["drop_bottom_pct"]
        )
        st.session_state["drop_bottom_auto_result"] = selected

    except Exception as exc:
        st.session_state["drop_bottom_auto_error"] = str(exc)


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
st.caption("Holdings are stored in a local SQLite master table instead of being uploaded from Excel.")
st.caption(f"Active SQLite file: `{DB_PATH}`")
st.caption(f"App build: `{APP_BUILD}`")

if recovered_db_path is not None:
    st.warning(
        "The previous database file was not a valid SQLite database and was preserved as "
        f"`{recovered_db_path.name}`. A new holdings database was created."
    )

# Filled after every add/remove operation using a fresh SQLite query.
live_count_banner_placeholder = st.empty()

# Latest saved/restored analysis is rendered directly below the live banner.
saved_analysis_placeholder = st.empty()

update_messages = []
update_warnings = []
update_errors = []

if "holdings_editor_version" not in st.session_state:
    st.session_state["holdings_editor_version"] = 0

if "drop_bottom_pct" not in st.session_state:
    st.session_state["drop_bottom_pct"] = 0.20

for flash_key, target in (
    ("holdings_flash_success", update_messages),
    ("holdings_flash_warning", update_warnings),
    ("holdings_flash_error", update_errors),
):
    flash_message = st.session_state.pop(flash_key, None)
    if flash_message:
        target.append(flash_message)

with st.sidebar:
    st.header("Holdings database")
    sidebar_count_placeholder = st.empty()

    buy_input = st.text_area(
        "Buy / add NSE symbols",
        placeholder="RELIANCE, TCS, INFY",
        help="Enter one or more NSE symbols separated by commas, semicolons, or new lines.",
    )
    sell_input = st.text_area(
        "Sell / remove NSE symbols",
        placeholder="HDFCBANK, SBIN",
        help="Symbols entered here are removed completely from the master holdings table.",
    )

    update_holdings_btn = st.button("Update master holdings", use_container_width=True)

    st.divider()
    st.header("Holdings backup and restore")

    holdings_csv = holdings_backup_bytes()
    holdings_backup_date = datetime.now().strftime("%Y-%m-%d")
    st.download_button(
        "Download holdings backup CSV",
        data=holdings_csv,
        file_name=f"portfolio_holdings_backup_{holdings_backup_date}.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_holdings_backup_sidebar",
        disabled=get_unique_holdings_count() == 0,
        help="Download this before a Streamlit Cloud restart or redeployment.",
    )

    holdings_backup_upload = st.file_uploader(
        "Upload holdings backup CSV",
        type=["csv"],
        key="holdings_backup_upload",
        help="Restore Symbol, Stock Name, Quantity, Average Price and timestamps.",
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
        use_container_width=True,
        key="restore_holdings_btn",
        disabled=holdings_backup_upload is None,
    )

    st.divider()
    st.header("Analysis results backup")

    existing_analysis_backup = latest_analysis_backup_bytes()
    if existing_analysis_backup is not None:
        latest_saved_payload = load_latest_analysis() or {}
        latest_saved_date = (
            str(latest_saved_payload.get("saved_at", ""))[:10] or "latest"
        )
        st.download_button(
            "Download latest analysis JSON",
            data=existing_analysis_backup,
            file_name=f"portfolio_analysis_backup_{latest_saved_date}.json",
            mime="application/json",
            use_container_width=True,
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
        use_container_width=True,
        key="restore_analysis_btn",
        disabled=analysis_backup_upload is None,
    )

    st.divider()
    st.header("Analysis inputs")
    days_to_flip = st.number_input("Expected days to flip", min_value=1, value=13, step=1)
    max_dd_pct = st.number_input(
        "Max drawdown input (%)",
        min_value=0.00,
        value=7.60,
        step=0.01,
        format="%.2f",
    )
    max_dd = (max_dd_pct / 100) * 4
    st.caption(f"Internal max_dd used: {max_dd:.4f}")
    st.number_input(
        "Drop bottom fraction of tickers by history length",
        min_value=0.0,
        max_value=0.95,
        step=0.01,
        format="%.2f",
        key="drop_bottom_pct",
        help=(
            "When Run analysis is clicked, this control is automatically set "
            "to the 0.01 value producing trading days nearest to 252 from "
            "the 252+ side. You can still edit the displayed value manually."
        ),
    )
    drop_bottom_pct = float(st.session_state["drop_bottom_pct"])
    use_target_vol = st.checkbox("Use target volatility")
    target_volatility = (
        st.number_input(
            "Target volatility",
            min_value=0.001,
            value=0.10,
            step=0.01,
            format="%.3f",
        )
        if use_target_vol
        else None
    )
    run_btn = st.button(
        "Run analysis",
        use_container_width=True,
        type="primary",
        on_click=auto_set_drop_bottom_pct_for_analysis,
    )

    auto_drop_result = st.session_state.get("drop_bottom_auto_result")
    if auto_drop_result:
        if auto_drop_result["target_reached"]:
            st.success(
                f"Auto value: {auto_drop_result['drop_bottom_pct']:.2f} — "
                f"{auto_drop_result['trading_days']:,} trading days."
            )
        else:
            st.warning(
                f"252 trading days could not be reached. Using "
                f"{auto_drop_result['drop_bottom_pct']:.2f}, which provides "
                f"the maximum available {auto_drop_result['trading_days']:,} days."
            )

    auto_drop_error = st.session_state.get("drop_bottom_auto_error")
    if auto_drop_error:
        st.error(
            "Could not automatically calculate drop_bottom_pct: "
            f"{auto_drop_error}"
        )

if restore_holdings_btn:
    try:
        restore_mode = (
            "replace"
            if holdings_restore_choice == "Replace current holdings"
            else "merge"
        )
        restored_count = restore_holdings_backup(
            holdings_backup_upload,
            mode=restore_mode,
        )
        st.session_state["holdings_editor_version"] += 1
        st.session_state["holdings_flash_success"] = (
            f"Restored {restored_count} holdings from the CSV backup using "
            f"{restore_mode} mode."
        )
        st.rerun()
    except Exception as exc:
        update_errors.append(f"Could not restore holdings backup: {exc}")

if restore_analysis_btn:
    try:
        restored_payload = restore_latest_analysis_backup(analysis_backup_upload)
        update_messages.append(
            "Analysis backup restored successfully. Saved at: "
            + str(restored_payload.get("saved_at", "Unknown"))
        )
    except Exception as exc:
        update_errors.append(f"Could not restore analysis backup: {exc}")

if update_holdings_btn:
    buy_symbols = parse_symbol_input(buy_input)
    sell_symbols = parse_symbol_input(sell_input)
    overlap = sorted(set(buy_symbols) & set(sell_symbols))

    if overlap:
        update_errors.append(
            "The same symbol cannot be present in both Buy and Sell: " + ", ".join(overlap)
        )
    elif not buy_symbols and not sell_symbols:
        update_warnings.append("Enter at least one NSE symbol in Buy or Sell.")
    else:
        removed, not_held = remove_symbols_from_master(sell_symbols)
        added, duplicates, invalid, missing_initial_price = add_symbols_to_master(buy_symbols)

        if added:
            update_messages.append("Added: " + ", ".join(added))
        if removed:
            update_messages.append("Removed: " + ", ".join(removed))
        if duplicates:
            update_warnings.append("Already in master table: " + ", ".join(duplicates))
        if not_held:
            update_warnings.append("Not present in master table: " + ", ".join(not_held))
        if invalid:
            update_errors.append("Not found in the NSE equity symbol list: " + ", ".join(invalid))
        if missing_initial_price:
            update_warnings.append(
                "Added without an initial price; enter Average Price in the editor before analysis: "
                + ", ".join(missing_initial_price)
            )

        if added or removed:
            # The holdings row set changed, so use a fresh editor widget key.
            st.session_state["holdings_editor_version"] += 1

for message in update_messages:
    st.success(message)
for message in update_warnings:
    st.warning(message)
for message in update_errors:
    st.error(message)

# Fresh, uncached database count on every Streamlit rerun.
live_unique_count = render_live_holdings_banner(live_count_banner_placeholder)
sidebar_count_placeholder.metric("Current unique holdings", live_unique_count)
render_saved_analysis(saved_analysis_placeholder)

st.subheader("Master Holdings")
master_df = load_master_holdings()

if master_df.empty:
    st.info("The master holdings table is empty. Add NSE symbols from the sidebar.")
else:
    st.dataframe(master_df, use_container_width=True, hide_index=True)

    with st.expander("Edit quantity and average price", expanded=False):
        st.caption(
            "A newly added symbol starts with quantity 1 and the latest available NSE price. "
            "After you save an edit, Quantity and Average Price are reloaded from SQLite."
        )
        editable_df = master_df[
            ["Symbol", "Stock Name", "Quantity", "Average Price"]
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
                use_container_width=True,
                hide_index=True,
                disabled=["Symbol", "Stock Name"],
                column_config={
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity",
                        min_value=0.000001,
                        format="%.6f",
                    ),
                    "Average Price": st.column_config.NumberColumn(
                        "Average Price",
                        min_value=0.01,
                        format="₹%.2f",
                    ),
                },
                key=f"holdings_editor_{editor_version}",
            )
            save_holdings_btn = st.form_submit_button(
                "Save quantity and price changes",
                use_container_width=True,
                type="primary",
            )

        if save_holdings_btn:
            try:
                updated_count = save_holding_values(edited_df)
                # A new key forces Streamlit to discard the old editor snapshot.
                st.session_state["holdings_editor_version"] += 1
                st.session_state["holdings_flash_success"] = (
                    f"Saved Quantity and Average Price for {updated_count} holdings."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save holdings: {exc}")


if run_btn:
    try:
        auto_drop_error = st.session_state.get("drop_bottom_auto_error")
        if auto_drop_error:
            st.error(
                "Analysis was not started because drop_bottom_pct could not be "
                f"calculated automatically: {auto_drop_error}"
            )
            st.stop()

        if master_df.empty:
            st.error("Add at least one NSE symbol before running the analysis.")
            st.stop()

        with st.spinner("Loading holdings from the database..."):
            portfolio_df, invalid_holding_rows = build_current_allocation_from_db()

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

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Current Allocation")
            st.dataframe(
                portfolio_df.style.format({
                    "Quantity": "{:,.4f}",
                    "Average Price": "₹{:,.2f}",
                    "Weight": "{:.2%}",
                }),
                use_container_width=True,
            )

        with col2:
            total_invested = float((portfolio_df["Quantity"] * portfolio_df["Average Price"]).sum())
            st.metric("Stocks in database", len(portfolio_df))
            st.metric("Total invested", f"₹{total_invested:,.2f}")

        symbols = portfolio_df["Symbol"].dropna().astype(str).str.upper().tolist()

        with st.spinner("Resolving Yahoo tickers..."):
            resolved_map = resolve_yahoo_tickers(symbols)

        unresolved = sorted(set(symbols) - set(resolved_map.keys()))
        if unresolved:
            st.warning(f"Unresolved Yahoo tickers: {', '.join(unresolved)}")

        yahoo_tickers = list(resolved_map.values())
        if not yahoo_tickers:
            st.error("No valid Yahoo tickers resolved.")
            st.stop()

        with st.spinner("Running optimization..."):
            optimal_weights, log_returns, current_stats, optimal_stats, meta = run_portfolio_analysis_multi(
                yahoo_tickers,
                portfolio_df[portfolio_df["Symbol"].isin(resolved_map.keys())].copy(),
                max_dd=max_dd,
                target_volatility=target_volatility,
                drop_bottom_pct=drop_bottom_pct,
            )

        if optimal_weights is None:
            st.error("Portfolio optimization did not return a usable allocation.")
            st.stop()

        if not meta["dropped_df"].empty:
            st.subheader("Dropped Tickers")
            st.dataframe(meta["dropped_df"], use_container_width=True)

        st.subheader("History Coverage")
        st.dataframe(meta["min_len_df"], use_container_width=True)
        st.info(
            f"**drop_bottom_pct used:** `{drop_bottom_pct:.2f}`  |  "
            f"**Trading days analysed:** {int(log_returns.shape[0]):,}  |  "
            f"**Assets in return matrix:** {int(log_returns.shape[1]):,}"
        )
        st.caption(f"Overlapping date range: {meta['valid_start'].date()} to {meta['valid_end'].date()}")
        st.caption(f"Log return shape: {log_returns.shape}")

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
                st.dataframe(current_df, use_container_width=True)

            with c2:
                st.markdown("**Optimized Portfolio**")
                optimal_df = metrics_df(optimal_stats)
                optimal_df["Value"] = optimal_df.apply(
                    lambda r: f"{r['Value']:.2%}" if r["Metric"] != "Sharpe Ratio" else f"{r['Value']:.2f}",
                    axis=1,
                )
                st.dataframe(optimal_df, use_container_width=True)

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

                st.dataframe(top_corrs.head(5), use_container_width=True)

        with st.spinner("Fetching latest prices..."):
            latest_prices = log_returns.columns.tolist()
            price_map = get_latest_price_map(latest_prices)

        rebal_df, missing_prices, missing_alloc = rebalance_plan_multi(
            portfolio_df[portfolio_df["Symbol"].isin(price_map.keys())].copy(),
            optimal_weights,
            log_returns,
            price_map,
            days_to_flip,
        )

        if missing_prices:
            st.warning(f"Skipped symbols with missing latest price: {', '.join(missing_prices)}")
        if missing_alloc:
            st.warning(f"Skipped symbols missing in allocation: {', '.join(missing_alloc)}")

        st.subheader("Rebalancing Plan")

        if rebal_df.empty:
            st.success("No trades to execute after filtering Executable Quantity = 0")
        else:
            st.dataframe(style_rebalance_df(rebal_df), use_container_width=True)

            csv = rebal_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download rebalancing plan CSV",
                data=csv,
                file_name="rebalancing_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )

        analysis_payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "holdings_analyzed": int(len(portfolio_df)),
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
                "missing_latest_prices": missing_prices,
                "missing_allocation": missing_alloc,
            },
        }

        save_latest_analysis(analysis_payload)
        render_saved_analysis(saved_analysis_placeholder)
        st.success(
            "Analysis completed and saved. You can now download the complete "
            "analysis backup as JSON."
        )

    except Exception as e:
        st.error(f"Error: {e}")
