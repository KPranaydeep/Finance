import io
import json
import os
import re
import sqlite3
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from scipy.optimize import minimize
from scipy.stats import kurtosis, norm, skew
from urllib3.util.retry import Retry

try:
    from mftool import Mftool
except ImportError:  # The app shows a clearer message when historical data is requested.
    Mftool = None

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Indian Mutual Fund Portfolio Rebalancer",
    page_icon="📈",
    layout="wide",
)

APP_BUILD = "2026-07-01-amfi-mf-v1"
AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
AMFI_NAV_FALLBACK_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_HISTORY_URL = "https://api.mfapi.in/mf/{scheme_code}"
ANNUALIZATION_DAYS = 252


# =========================================================
# HTTP / AMFI / MFTOOL DATA SOURCES
# =========================================================

@st.cache_resource
def get_http_session() -> requests.Session:
    """Create a retrying HTTP session for AMFI and the historical-NAV fallback."""
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
            )
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def parse_amfi_nav_text(text: str) -> pd.DataFrame:
    """Parse AMFI's semicolon-delimited NAVAll.txt report.

    The report contains category and AMC headings in addition to data rows. Only
    rows with at least six fields and a scheme code are treated as schemes.
    """
    rows: List[dict] = []
    current_category = ""
    current_amc = ""

    for raw_line in text.replace("\ufeff", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if lower_line.startswith("scheme code;"):
            continue

        parts = [part.strip() for part in line.split(";")]
        is_scheme_row = len(parts) >= 6 and bool(parts[0])

        if is_scheme_row:
            scheme_code = parts[0]
            # AMFI mutual-fund scheme codes are normally numeric. Retaining the
            # text representation prevents accidental conversion or truncation.
            if not re.fullmatch(r"[A-Za-z0-9_-]+", scheme_code):
                continue

            rows.append(
                {
                    "Scheme Code": scheme_code,
                    "ISIN Growth/Payout": parts[1] if len(parts) > 1 else "",
                    "ISIN Reinvestment": parts[2] if len(parts) > 2 else "",
                    "Scheme Name": parts[3] if len(parts) > 3 else "",
                    "Latest NAV": parts[4] if len(parts) > 4 else None,
                    "NAV Date": parts[5] if len(parts) > 5 else None,
                    "Scheme Category": current_category,
                    "AMC": current_amc,
                }
            )
            continue

        if ";" not in line:
            if lower_line.startswith(
                ("open ended schemes", "close ended schemes", "interval fund schemes")
            ):
                current_category = line
                current_amc = ""
            else:
                current_amc = line

    if not rows:
        raise ValueError("AMFI NAV report did not contain any parseable scheme rows.")

    df = pd.DataFrame(rows)
    df["Scheme Code"] = df["Scheme Code"].astype(str).str.strip()
    df["Scheme Name"] = df["Scheme Name"].astype(str).str.strip()
    df["Latest NAV"] = pd.to_numeric(df["Latest NAV"], errors="coerce")
    df["NAV Date"] = pd.to_datetime(df["NAV Date"], dayfirst=True, errors="coerce")
    df = (
        df[df["Scheme Code"].ne("")]
        .drop_duplicates(subset=["Scheme Code"], keep="last")
        .sort_values(["AMC", "Scheme Name", "Scheme Code"], kind="stable")
        .reset_index(drop=True)
    )
    return df


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_amfi_nav_data() -> pd.DataFrame:
    """Load the official AMFI scheme master and latest NAV report."""
    session = get_http_session()
    errors = []

    for url in (AMFI_NAV_URL, AMFI_NAV_FALLBACK_URL):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            # AMFI content is mostly ASCII/UTF-8, but replacement keeps the app
            # usable if a fund name contains an unexpected byte sequence.
            text = response.content.decode("utf-8-sig", errors="replace")
            return parse_amfi_nav_text(text)
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise ConnectionError(
        "Could not download the AMFI NAV report. " + " | ".join(errors)
    )


@st.cache_resource
def get_mftool_client():
    """Create one shared mftool client instead of reloading AMFI per scheme."""
    if Mftool is None:
        raise RuntimeError(
            "The 'mftool' package is not installed. Run: pip install mftool"
        )
    return Mftool()


_MFTOOL_LOCK = threading.Lock()


def _history_payload_to_series(payload: dict, scheme_code: str) -> pd.Series:
    if not isinstance(payload, dict):
        raise ValueError(f"Historical NAV response for {scheme_code} is not a dictionary.")

    records = payload.get("data")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Historical NAV is unavailable for scheme {scheme_code}.")

    history = pd.DataFrame(records)
    if "date" not in history.columns or "nav" not in history.columns:
        raise ValueError(f"Historical NAV response for {scheme_code} is incomplete.")

    history["date"] = pd.to_datetime(history["date"], dayfirst=True, errors="coerce")
    history["nav"] = pd.to_numeric(history["nav"], errors="coerce")
    history = (
        history.dropna(subset=["date", "nav"])
        .loc[lambda frame: frame["nav"] > 0]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
    )

    if history.empty:
        raise ValueError(f"Historical NAV has no valid observations for {scheme_code}.")

    series = history.set_index("date")["nav"].astype(float)
    series.name = str(scheme_code)
    return series


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_scheme_history(scheme_code: str) -> pd.Series:
    """Fetch a scheme's full NAV history with mftool and an HTTP fallback.

    mftool is the primary MF-specific library. Its historical endpoint is also
    exposed by mfapi.in, which is used only if the package call fails.
    """
    code = normalize_scheme_code(scheme_code)
    primary_error: Optional[Exception] = None

    try:
        # mftool's internal session/cache is shared; serializing calls avoids any
        # race in Streamlit deployments with simultaneous sessions.
        with _MFTOOL_LOCK:
            payload = get_mftool_client().get_scheme_historical_nav(
                code,
                as_json=False,
                as_Dataframe=False,
            )
        return _history_payload_to_series(payload, code)
    except Exception as exc:
        primary_error = exc

    try:
        response = get_http_session().get(
            MFAPI_HISTORY_URL.format(scheme_code=code),
            timeout=45,
        )
        response.raise_for_status()
        return _history_payload_to_series(response.json(), code)
    except Exception as fallback_error:
        raise RuntimeError(
            f"Could not fetch historical NAV for {code}. "
            f"mftool: {primary_error}; fallback: {fallback_error}"
        ) from fallback_error


def download_nav_history(
    scheme_codes: Sequence[str],
    start_date: Optional[str] = "2000-01-01",
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Download and combine historical NAVs for multiple AMFI scheme codes."""
    series_list: List[pd.Series] = []
    errors: Dict[str, str] = {}

    start_ts = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end_ts = pd.to_datetime(end_date, errors="coerce") if end_date else None

    for raw_code in scheme_codes:
        code = normalize_scheme_code(raw_code)
        try:
            series = fetch_scheme_history(code)
            if start_ts is not None and pd.notna(start_ts):
                series = series.loc[series.index >= start_ts]
            if end_ts is not None and pd.notna(end_ts):
                series = series.loc[series.index <= end_ts]
            if series.empty:
                raise ValueError("No observations remain in the selected date range.")
            series_list.append(series.rename(code))
        except Exception as exc:
            errors[code] = str(exc)

    if not series_list:
        error_text = " | ".join(f"{code}: {message}" for code, message in errors.items())
        raise ValueError("No historical mutual-fund NAV data could be loaded. " + error_text)

    nav_history = pd.concat(series_list, axis=1).sort_index()
    nav_history = nav_history[~nav_history.index.duplicated(keep="last")]
    return nav_history, errors


# =========================================================
# DATABASE / HOLDINGS
# =========================================================

SCRIPT_DB_PATH = Path(__file__).resolve().with_name("mf_portfolio_holdings.db")
USER_DB_PATH = Path.home() / ".mf_portfolio_rebalancer" / "mf_portfolio_holdings.db"
DEFAULT_DB_PATH = SCRIPT_DB_PATH if SCRIPT_DB_PATH.exists() else USER_DB_PATH
DB_PATH = Path(os.getenv("MF_PORTFOLIO_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()

MASTER_HOLDINGS_DDL = """
CREATE TABLE IF NOT EXISTS master_holdings (
    scheme_code TEXT PRIMARY KEY,
    scheme_name TEXT NOT NULL,
    units REAL NOT NULL DEFAULT 1,
    average_nav REAL,
    latest_nav REAL,
    nav_date TEXT,
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


def _connect_sqlite() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(MASTER_HOLDINGS_DDL)
    conn.execute(LATEST_ANALYSIS_DDL)

    columns = {
        str(row["name"]).lower()
        for row in conn.execute("PRAGMA table_info(master_holdings)").fetchall()
    }

    if "scheme_code" not in columns:
        legacy_name = "master_holdings_legacy_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        conn.execute(f'ALTER TABLE master_holdings RENAME TO "{legacy_name}"')
        conn.execute(MASTER_HOLDINGS_DDL)
        columns = {
            str(row["name"]).lower()
            for row in conn.execute("PRAGMA table_info(master_holdings)").fetchall()
        }

    migrations = {
        "scheme_name": "ALTER TABLE master_holdings ADD COLUMN scheme_name TEXT",
        "units": "ALTER TABLE master_holdings ADD COLUMN units REAL DEFAULT 1",
        "average_nav": "ALTER TABLE master_holdings ADD COLUMN average_nav REAL",
        "latest_nav": "ALTER TABLE master_holdings ADD COLUMN latest_nav REAL",
        "nav_date": "ALTER TABLE master_holdings ADD COLUMN nav_date TEXT",
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
        SET scheme_name = COALESCE(NULLIF(TRIM(scheme_name), ''), scheme_code),
            units = CASE WHEN units IS NULL OR units <= 0 THEN 1 ELSE units END,
            added_at = COALESCE(NULLIF(TRIM(added_at), ''), ?),
            updated_at = COALESCE(NULLIF(TRIM(updated_at), ''), ?)
        """,
        (now, now),
    )
    conn.commit()


def get_db_connection() -> sqlite3.Connection:
    conn = _connect_sqlite()
    try:
        _ensure_schema(conn)
        return conn
    except Exception:
        conn.close()
        raise


def init_holdings_db() -> Optional[Path]:
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


def normalize_scheme_code(value) -> str:
    return str(value or "").strip().upper()


def parse_scheme_code_input(raw_text: str) -> List[str]:
    codes: List[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", raw_text or ""):
        code = normalize_scheme_code(part)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    return codes


def amfi_lookup(nav_master: pd.DataFrame) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for _, row in nav_master.iterrows():
        code = normalize_scheme_code(row["Scheme Code"])
        lookup[code] = {
            "scheme_name": str(row.get("Scheme Name") or code).strip(),
            "latest_nav": (
                float(row["Latest NAV"])
                if pd.notna(row.get("Latest NAV")) and float(row["Latest NAV"]) > 0
                else None
            ),
            "nav_date": (
                pd.Timestamp(row["NAV Date"]).date().isoformat()
                if pd.notna(row.get("NAV Date"))
                else None
            ),
            "amc": str(row.get("AMC") or "").strip(),
            "category": str(row.get("Scheme Category") or "").strip(),
        }
    return lookup


def load_master_holdings() -> pd.DataFrame:
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                scheme_code AS "Scheme Code",
                scheme_name AS "Scheme Name",
                units AS "Units",
                average_nav AS "Average NAV",
                latest_nav AS "Latest NAV",
                nav_date AS "NAV Date",
                added_at AS "Added At",
                updated_at AS "Updated At"
            FROM master_holdings
            ORDER BY scheme_name, scheme_code
            """,
            conn,
        )
    return df


def get_unique_holdings_count() -> int:
    with get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT TRIM(scheme_code)) AS unique_count
            FROM master_holdings
            WHERE scheme_code IS NOT NULL AND TRIM(scheme_code) <> ''
            """
        ).fetchone()
    return int(row["unique_count"] or 0)


def add_scheme_codes_to_master(
    scheme_codes: Sequence[str],
    nav_master: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if not scheme_codes:
        return [], [], [], []

    lookup = amfi_lookup(nav_master)
    valid = []
    invalid = []
    seen = set()

    for raw_code in scheme_codes:
        code = normalize_scheme_code(raw_code)
        if code not in lookup:
            invalid.append(code)
        elif code not in seen:
            valid.append(code)
            seen.add(code)

    with get_db_connection() as conn:
        existing = (
            {
                row["scheme_code"]
                for row in conn.execute(
                    "SELECT scheme_code FROM master_holdings WHERE scheme_code IN ({})".format(
                        ",".join("?" for _ in valid)
                    ),
                    valid,
                ).fetchall()
            }
            if valid
            else set()
        )

    duplicates = [code for code in valid if code in existing]
    new_codes = [code for code in valid if code not in existing]
    missing_nav = []
    added = []
    now = datetime.now().isoformat(timespec="seconds")

    with get_db_connection() as conn:
        for code in new_codes:
            info = lookup[code]
            latest_nav = info["latest_nav"]
            if latest_nav is None:
                missing_nav.append(code)

            conn.execute(
                """
                INSERT INTO master_holdings
                    (scheme_code, scheme_name, units, average_nav, latest_nav,
                     nav_date, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    code,
                    info["scheme_name"],
                    1.0,
                    latest_nav,
                    latest_nav,
                    info["nav_date"],
                    now,
                    now,
                ),
            )
            added.append(code)
        conn.commit()

    return added, duplicates, invalid, missing_nav


def remove_scheme_codes_from_master(scheme_codes: Sequence[str]) -> Tuple[List[str], List[str]]:
    if not scheme_codes:
        return [], []

    normalized = list(dict.fromkeys(normalize_scheme_code(code) for code in scheme_codes))
    with get_db_connection() as conn:
        existing = {
            row["scheme_code"]
            for row in conn.execute("SELECT scheme_code FROM master_holdings").fetchall()
        }
        removed = [code for code in normalized if code in existing]
        missing = [code for code in normalized if code not in existing]

        if removed:
            conn.execute(
                "DELETE FROM master_holdings WHERE scheme_code IN ({})".format(
                    ",".join("?" for _ in removed)
                ),
                removed,
            )
            conn.commit()

    return removed, missing


def refresh_holdings_latest_nav(nav_master: pd.DataFrame) -> List[str]:
    lookup = amfi_lookup(nav_master)
    now = datetime.now().isoformat(timespec="seconds")
    missing_codes = []

    with get_db_connection() as conn:
        codes = [
            row["scheme_code"]
            for row in conn.execute("SELECT scheme_code FROM master_holdings").fetchall()
        ]

        for code in codes:
            info = lookup.get(code)
            if info is None or info["latest_nav"] is None:
                missing_codes.append(code)
                continue
            conn.execute(
                """
                UPDATE master_holdings
                SET scheme_name = ?, latest_nav = ?, nav_date = ?, updated_at = ?
                WHERE scheme_code = ?
                """,
                (
                    info["scheme_name"],
                    float(info["latest_nav"]),
                    info["nav_date"],
                    now,
                    code,
                ),
            )
        conn.commit()

    return missing_codes


def save_holding_values(edited_df: pd.DataFrame) -> int:
    required = ["Scheme Code", "Units", "Average NAV"]
    missing_columns = [column for column in required if column not in edited_df.columns]
    if missing_columns:
        raise ValueError(f"Missing editable holding columns: {missing_columns}")

    cleaned = edited_df.copy()
    cleaned["Scheme Code"] = cleaned["Scheme Code"].map(normalize_scheme_code)
    cleaned["Units"] = pd.to_numeric(cleaned["Units"], errors="coerce")
    cleaned["Average NAV"] = pd.to_numeric(cleaned["Average NAV"], errors="coerce")

    if cleaned.empty:
        raise ValueError("There are no mutual-fund holdings to save.")
    if cleaned["Scheme Code"].eq("").any():
        raise ValueError("Every holding must contain a valid AMFI scheme code.")
    if cleaned["Scheme Code"].duplicated().any():
        raise ValueError("Duplicate scheme codes are not allowed.")
    if cleaned["Units"].isna().any() or (cleaned["Units"] <= 0).any():
        raise ValueError("Units must be greater than zero for every holding.")
    if cleaned["Average NAV"].isna().any() or (cleaned["Average NAV"] <= 0).any():
        raise ValueError("Average NAV must be greater than zero for every holding.")

    codes = cleaned["Scheme Code"].tolist()
    placeholders = ",".join("?" for _ in codes)
    now = datetime.now().isoformat(timespec="seconds")

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = {
                row["scheme_code"]
                for row in conn.execute(
                    f"SELECT scheme_code FROM master_holdings WHERE scheme_code IN ({placeholders})",
                    codes,
                ).fetchall()
            }
            absent = sorted(set(codes) - existing)
            if absent:
                raise ValueError("These schemes no longer exist in SQLite: " + ", ".join(absent))

            updated_count = 0
            for _, row in cleaned.iterrows():
                cursor = conn.execute(
                    """
                    UPDATE master_holdings
                    SET units = ?, average_nav = ?, updated_at = ?
                    WHERE scheme_code = ?
                    """,
                    (
                        float(row["Units"]),
                        float(row["Average NAV"]),
                        now,
                        row["Scheme Code"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"SQLite did not update scheme {row['Scheme Code']} exactly once."
                    )
                updated_count += cursor.rowcount

            conn.commit()
            return updated_count
        except Exception:
            conn.rollback()
            raise


def build_current_allocation_from_db(nav_master: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    refresh_holdings_latest_nav(nav_master)
    df = load_master_holdings()
    if df.empty:
        return pd.DataFrame(), []

    for column in ("Units", "Average NAV", "Latest NAV"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_mask = (
        df["Units"].isna()
        | (df["Units"] <= 0)
        | df["Average NAV"].isna()
        | (df["Average NAV"] <= 0)
        | df["Latest NAV"].isna()
        | (df["Latest NAV"] <= 0)
    )
    invalid_codes = df.loc[invalid_mask, "Scheme Code"].astype(str).tolist()
    usable = df.loc[~invalid_mask].copy()

    if usable.empty:
        return pd.DataFrame(), invalid_codes

    usable["Invested Amount"] = usable["Units"] * usable["Average NAV"]
    usable["Current Value"] = usable["Units"] * usable["Latest NAV"]
    usable["Unrealized P&L"] = usable["Current Value"] - usable["Invested Amount"]
    usable["Return %"] = np.where(
        usable["Invested Amount"] > 0,
        usable["Unrealized P&L"] / usable["Invested Amount"],
        np.nan,
    )
    total_value = float(usable["Current Value"].sum())
    if total_value <= 0:
        raise ValueError("The mutual-fund portfolio has no positive current value.")
    usable["Weight"] = usable["Current Value"] / total_value

    columns = [
        "Scheme Code",
        "Scheme Name",
        "Units",
        "Average NAV",
        "Latest NAV",
        "NAV Date",
        "Invested Amount",
        "Current Value",
        "Unrealized P&L",
        "Return %",
        "Weight",
    ]
    return (
        usable[columns]
        .sort_values("Weight", ascending=False, kind="stable")
        .reset_index(drop=True),
        invalid_codes,
    )


# =========================================================
# BACKUP / RESTORE / SAVED ANALYSIS
# =========================================================


def holdings_backup_bytes() -> bytes:
    return load_master_holdings().to_csv(index=False).encode("utf-8-sig")


def _read_holdings_backup(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Choose a mutual-fund holdings backup CSV first.")
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected holdings backup is empty.")

    try:
        backup_df = pd.read_csv(io.BytesIO(raw), dtype={"Scheme Code": str})
    except Exception as exc:
        raise ValueError(f"Could not read the holdings CSV backup: {exc}") from exc

    aliases = {
        "scheme code": "Scheme Code",
        "scheme_code": "Scheme Code",
        "scheme name": "Scheme Name",
        "scheme_name": "Scheme Name",
        "units": "Units",
        "average nav": "Average NAV",
        "average_nav": "Average NAV",
        "latest nav": "Latest NAV",
        "latest_nav": "Latest NAV",
        "nav date": "NAV Date",
        "nav_date": "NAV Date",
        "added at": "Added At",
        "added_at": "Added At",
        "updated at": "Updated At",
        "updated_at": "Updated At",
    }
    backup_df = backup_df.rename(
        columns={
            column: aliases[str(column).strip().lower()]
            for column in backup_df.columns
            if str(column).strip().lower() in aliases
        }
    )

    required = ["Scheme Code", "Units", "Average NAV"]
    missing = [column for column in required if column not in backup_df.columns]
    if missing:
        raise ValueError("The backup is missing required columns: " + ", ".join(missing))

    defaults = {
        "Scheme Name": backup_df["Scheme Code"],
        "Latest NAV": np.nan,
        "NAV Date": None,
        "Added At": None,
        "Updated At": None,
    }
    for column, value in defaults.items():
        if column not in backup_df.columns:
            backup_df[column] = value

    cleaned = backup_df[
        [
            "Scheme Code",
            "Scheme Name",
            "Units",
            "Average NAV",
            "Latest NAV",
            "NAV Date",
            "Added At",
            "Updated At",
        ]
    ].copy()
    cleaned["Scheme Code"] = cleaned["Scheme Code"].map(normalize_scheme_code)
    cleaned["Scheme Name"] = cleaned["Scheme Name"].fillna("").astype(str).str.strip()
    for column in ("Units", "Average NAV", "Latest NAV"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if cleaned.empty:
        raise ValueError("The backup does not contain any rows.")
    if cleaned["Scheme Code"].eq("").any():
        raise ValueError("Every backup row must contain a valid Scheme Code.")
    if cleaned["Scheme Code"].duplicated().any():
        duplicates = sorted(
            cleaned.loc[
                cleaned["Scheme Code"].duplicated(keep=False), "Scheme Code"
            ].unique()
        )
        raise ValueError("The backup contains duplicate scheme codes: " + ", ".join(duplicates))
    if cleaned["Units"].isna().any() or (cleaned["Units"] <= 0).any():
        raise ValueError("Units must be greater than zero for every restored holding.")
    if cleaned["Average NAV"].isna().any() or (cleaned["Average NAV"] <= 0).any():
        raise ValueError("Average NAV must be greater than zero for every restored holding.")

    cleaned["Scheme Name"] = cleaned.apply(
        lambda row: row["Scheme Name"] or row["Scheme Code"], axis=1
    )
    return cleaned


def restore_holdings_backup(uploaded_file, mode: str = "merge") -> int:
    cleaned = _read_holdings_backup(uploaded_file)
    normalized_mode = str(mode or "merge").strip().lower()
    if normalized_mode not in {"replace", "merge"}:
        raise ValueError("Restore mode must be either 'replace' or 'merge'.")

    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for _, row in cleaned.iterrows():
        latest_nav = (
            float(row["Latest NAV"])
            if pd.notna(row["Latest NAV"]) and float(row["Latest NAV"]) > 0
            else None
        )
        records.append(
            (
                row["Scheme Code"],
                row["Scheme Name"],
                float(row["Units"]),
                float(row["Average NAV"]),
                latest_nav,
                str(row["NAV Date"]).strip() if pd.notna(row["NAV Date"]) else None,
                str(row["Added At"]).strip() if pd.notna(row["Added At"]) else now,
                str(row["Updated At"]).strip() if pd.notna(row["Updated At"]) else now,
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
                    (scheme_code, scheme_name, units, average_nav, latest_nav,
                     nav_date, added_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scheme_code) DO UPDATE SET
                    scheme_name = excluded.scheme_name,
                    units = excluded.units,
                    average_nav = excluded.average_nav,
                    latest_nav = excluded.latest_nav,
                    nav_date = excluded.nav_date,
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


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def save_latest_analysis(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Analysis payload must be a dictionary.")
    safe_payload = make_json_safe(payload)
    saved_at = str(safe_payload.get("saved_at") or datetime.now().isoformat(timespec="seconds"))
    safe_payload["saved_at"] = saved_at
    payload_json = json.dumps(safe_payload, ensure_ascii=False, indent=2, allow_nan=False)

    with get_db_connection() as conn:
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


def load_latest_analysis() -> Optional[dict]:
    try:
        with get_db_connection() as conn:
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


def latest_analysis_backup_bytes() -> Optional[bytes]:
    payload = load_latest_analysis()
    if payload is None:
        return None
    return json.dumps(
        make_json_safe(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def restore_latest_analysis_backup(uploaded_file) -> dict:
    if uploaded_file is None:
        raise ValueError("Choose an analysis backup JSON file first.")
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected analysis backup is empty.")
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Could not read the JSON backup: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("The analysis backup must contain one JSON object.")

    required = {
        "saved_at",
        "holdings_analyzed",
        "total_current_value",
        "current_stats",
        "optimal_stats",
        "rebalancing_plan",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("This is not a complete MF analysis backup. Missing: " + ", ".join(missing))
    return save_latest_analysis(payload)


# =========================================================
# NAV ALIGNMENT / RETURNS / OPTIMIZATION
# =========================================================


def _align_nav_history(
    nav_history: pd.DataFrame,
    kept_codes: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    selected = nav_history[list(kept_codes)].copy()
    first_dates = selected.apply(lambda series: series.first_valid_index())
    last_dates = selected.apply(lambda series: series.last_valid_index())

    valid_start = first_dates.max()
    valid_end = last_dates.min()
    if valid_start is None or valid_end is None or pd.isna(valid_start) or pd.isna(valid_end):
        raise ValueError("No overlapping NAV date range was found across the selected schemes.")
    if valid_start >= valid_end:
        raise ValueError("The overlapping NAV date range is too short.")

    aligned = selected.loc[valid_start:valid_end].dropna(axis=0, how="any")
    if len(aligned) < 3:
        raise ValueError("Fewer than three common NAV observations are available.")
    return aligned, pd.Timestamp(valid_start), pd.Timestamp(valid_end)


def get_daily_log_returns(
    scheme_codes: Sequence[str],
    start_date: Optional[str] = "2000-01-01",
    end_date: Optional[str] = None,
    drop_bottom_pct: float = 0.10,
):
    nav_history, history_errors = download_nav_history(
        scheme_codes,
        start_date=start_date,
        end_date=end_date,
    )

    lengths = nav_history.count().sort_values(ascending=False)
    if lengths.empty:
        raise ValueError("No scheme history is available.")

    total_schemes = len(lengths)
    minimum_to_keep = 2 if total_schemes >= 2 else 1
    num_to_drop = min(
        int(np.floor(float(drop_bottom_pct) * total_schemes)),
        total_schemes - minimum_to_keep,
    )

    if num_to_drop > 0:
        dropped = lengths.tail(num_to_drop)
        kept = lengths.head(total_schemes - num_to_drop)
    else:
        dropped = pd.Series(dtype=float)
        kept = lengths

    aligned_navs, valid_start, valid_end = _align_nav_history(nav_history, kept.index.tolist())
    log_returns = np.log(aligned_navs / aligned_navs.shift(1)).dropna(how="any")
    if log_returns.empty:
        raise ValueError("NAV history did not produce usable daily returns.")

    min_code = kept.idxmin()
    min_series = nav_history[min_code].dropna()
    if len(min_series) >= 2:
        simulated_return = (float(min_series.iloc[-1]) / float(min_series.iloc[0])) - 1.0
    else:
        simulated_return = np.nan

    dropped_df = pd.DataFrame(
        {
            "Scheme Code": dropped.index.astype(str),
            "Valid NAV Observations": dropped.values.astype(int),
        }
    )
    min_len_df = pd.DataFrame(
        {
            "Scheme Code": [str(min_code)],
            "History Length (NAV observations)": [int(kept[min_code])],
            "Since-first-observation Return": [simulated_return],
        }
    )

    meta = {
        "valid_start": valid_start,
        "valid_end": valid_end,
        "dropped_df": dropped_df,
        "min_len_df": min_len_df,
        "history_errors": history_errors,
        "aligned_navs": aligned_navs,
        "raw_nav_history": nav_history,
    }
    return log_returns, meta


def find_drop_bottom_pct_nearest_target(
    scheme_codes: Sequence[str],
    target_observations: int = 252,
    start_date: Optional[str] = "2000-01-01",
    end_date: Optional[str] = None,
) -> dict:
    nav_history, history_errors = download_nav_history(
        scheme_codes,
        start_date=start_date,
        end_date=end_date,
    )
    lengths = nav_history.count().sort_values(ascending=False)
    total = len(lengths)
    if total == 0:
        raise ValueError("No mutual-fund NAV history is available.")

    minimum_to_keep = 2 if total >= 2 else 1
    maximum_drop = total - minimum_to_keep
    candidates = []

    for step in range(96):
        pct = round(step / 100, 2)
        num_to_drop = int(np.floor(pct * total))
        if num_to_drop > maximum_drop:
            continue
        kept_codes = lengths.head(total - num_to_drop).index.tolist()
        try:
            aligned, valid_start, valid_end = _align_nav_history(nav_history, kept_codes)
        except ValueError:
            continue
        observations = max(len(aligned) - 1, 0)
        if observations <= 0:
            continue
        candidates.append(
            {
                "drop_bottom_pct": pct,
                "common_return_observations": observations,
                "num_to_drop": num_to_drop,
                "schemes_kept": len(kept_codes),
                "valid_start": str(valid_start),
                "valid_end": str(valid_end),
            }
        )

    if not candidates:
        raise ValueError("No usable overlapping NAV history was found.")

    above_target = [
        candidate
        for candidate in candidates
        if candidate["common_return_observations"] >= target_observations
    ]
    if above_target:
        selected = min(
            above_target,
            key=lambda candidate: (
                candidate["common_return_observations"] - target_observations,
                candidate["drop_bottom_pct"],
            ),
        ).copy()
        selected["target_reached"] = True
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["common_return_observations"],
                -candidate["drop_bottom_pct"],
            ),
        ).copy()
        selected["target_reached"] = False

    selected["target_observations"] = int(target_observations)
    selected["history_errors"] = history_errors
    return selected


def portfolio_max_drawdown(weights: np.ndarray, log_returns: pd.DataFrame) -> float:
    portfolio_returns = log_returns @ weights
    growth = np.exp(portfolio_returns.cumsum())
    running_peak = growth.cummax()
    drawdown = 1.0 - (growth / running_peak)
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _daily_risk_free_rate(annual_rate: float) -> float:
    annual_rate = max(float(annual_rate), -0.99)
    return float(np.log1p(annual_rate) / ANNUALIZATION_DAYS)


def optimize_max_sharpe_ratio(
    log_returns: pd.DataFrame,
    risk_free_rate_annual: float,
) -> Optional[np.ndarray]:
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    num_assets = len(mean_returns)
    risk_free_daily = _daily_risk_free_rate(risk_free_rate_annual)

    def negative_sharpe(weights):
        portfolio_return = float(np.dot(weights, mean_returns))
        variance = float(weights.T @ cov_matrix @ weights)
        volatility = np.sqrt(max(variance, 0.0))
        if volatility <= 1e-12:
            return 1e9
        return -((portfolio_return - risk_free_daily) / volatility)

    result = minimize(
        negative_sharpe,
        np.ones(num_assets) / num_assets,
        method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return result.x if result.success else None


def optimize_max_sharpe_with_drawdown(
    log_returns: pd.DataFrame,
    max_drawdown: float,
    risk_free_rate_annual: float,
) -> Optional[np.ndarray]:
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    num_assets = len(mean_returns)
    risk_free_daily = _daily_risk_free_rate(risk_free_rate_annual)

    def negative_sharpe(weights):
        portfolio_return = float(np.dot(weights, mean_returns))
        variance = float(weights.T @ cov_matrix @ weights)
        volatility = np.sqrt(max(variance, 0.0))
        if volatility <= 1e-12:
            return 1e9
        return -((portfolio_return - risk_free_daily) / volatility)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: float(max_drawdown)
            - portfolio_max_drawdown(weights, log_returns),
        },
    ]
    result = minimize(
        negative_sharpe,
        np.ones(num_assets) / num_assets,
        method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return result.x if result.success else None


def optimize_target_daily_volatility(
    log_returns: pd.DataFrame,
    target_volatility: float,
) -> Optional[np.ndarray]:
    mean_returns = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    num_assets = len(mean_returns)

    def objective(weights):
        return -float(np.dot(weights, mean_returns))

    def volatility_constraint(weights):
        variance = float(weights.T @ cov_matrix @ weights)
        return float(target_volatility) - np.sqrt(max(variance, 0.0))

    result = minimize(
        objective,
        np.ones(num_assets) / num_assets,
        method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(num_assets)),
        constraints=[
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            {"type": "ineq", "fun": volatility_constraint},
        ],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    return result.x if result.success else None


def portfolio_stats(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    risk_free_rate_annual: float,
) -> dict:
    portfolio_returns = log_returns @ weights
    mean_daily = float(portfolio_returns.mean())
    std_daily = float(portfolio_returns.std(ddof=1))
    annualized_return = float(np.expm1(mean_daily * ANNUALIZATION_DAYS))
    annualized_volatility = std_daily * np.sqrt(ANNUALIZATION_DAYS)

    series_skew = float(skew(portfolio_returns, bias=False)) if len(portfolio_returns) > 2 else 0.0
    series_kurtosis = (
        float(kurtosis(portfolio_returns, fisher=True, bias=False))
        if len(portfolio_returns) > 3
        else 0.0
    )
    if not np.isfinite(series_skew):
        series_skew = 0.0
    if not np.isfinite(series_kurtosis):
        series_kurtosis = 0.0

    alpha = 0.05
    z = norm.ppf(alpha)
    z_cf = (
        z
        + (1 / 6) * (z**2 - 1) * series_skew
        + (1 / 24) * (z**3 - 3 * z) * series_kurtosis
        - (1 / 36) * (2 * z**3 - 5 * z) * series_skew**2
    )
    daily_cf_loss = -(mean_daily + z_cf * std_daily)
    annualized_cf_cvar = daily_cf_loss * np.sqrt(ANNUALIZATION_DAYS)

    sharpe = (
        (annualized_return - float(risk_free_rate_annual)) / annualized_volatility
        if annualized_volatility > 1e-12
        else 0.0
    )

    return {
        "Annual Return": annualized_return,
        "Annual Volatility": annualized_volatility,
        "Cornish-Fisher CVaR": annualized_cf_cvar,
        "Maximum Drawdown": portfolio_max_drawdown(weights, log_returns),
        "Sharpe Ratio": sharpe,
    }


def portfolio_stats_comparison(
    current_allocation: pd.DataFrame,
    log_returns: pd.DataFrame,
    optimal_weights: np.ndarray,
    risk_free_rate_annual: float,
) -> Tuple[dict, dict]:
    codes = [str(column) for column in log_returns.columns]
    aligned = (
        current_allocation.set_index("Scheme Code")
        .reindex(codes)
        .dropna(subset=["Weight"])
    )
    if aligned.empty:
        raise ValueError("Current mutual-fund weights could not be aligned to NAV history.")

    current_weights = aligned["Weight"].to_numpy(dtype=float)
    current_weights = current_weights / current_weights.sum()
    current_returns = log_returns[aligned.index.tolist()]

    return (
        portfolio_stats(current_weights, current_returns, risk_free_rate_annual),
        portfolio_stats(optimal_weights, log_returns, risk_free_rate_annual),
    )


def run_portfolio_analysis(
    scheme_codes: Sequence[str],
    current_allocation: pd.DataFrame,
    max_drawdown: float,
    risk_free_rate_annual: float,
    target_daily_volatility: Optional[float],
    drop_bottom_pct: float,
):
    log_returns, meta = get_daily_log_returns(
        scheme_codes,
        drop_bottom_pct=drop_bottom_pct,
    )

    optimal_weights = None
    optimization_method = ""
    if target_daily_volatility is not None:
        optimal_weights = optimize_target_daily_volatility(
            log_returns,
            target_daily_volatility,
        )
        if optimal_weights is not None:
            optimization_method = "Maximum return under target daily volatility"

    if optimal_weights is None:
        optimal_weights = optimize_max_sharpe_with_drawdown(
            log_returns,
            max_drawdown=max_drawdown,
            risk_free_rate_annual=risk_free_rate_annual,
        )
        if optimal_weights is not None:
            optimization_method = "Maximum Sharpe under drawdown constraint"

    if optimal_weights is None:
        optimal_weights = optimize_max_sharpe_ratio(
            log_returns,
            risk_free_rate_annual=risk_free_rate_annual,
        )
        if optimal_weights is not None:
            optimization_method = "Maximum Sharpe fallback"

    if optimal_weights is None:
        return None, log_returns, None, None, meta, "Optimization failed"

    current_stats, optimal_stats = portfolio_stats_comparison(
        current_allocation,
        log_returns,
        optimal_weights,
        risk_free_rate_annual,
    )
    return (
        optimal_weights,
        log_returns,
        current_stats,
        optimal_stats,
        meta,
        optimization_method,
    )


# =========================================================
# REBALANCING / INDICATORS / DISPLAY
# =========================================================


def calculate_ema252_change_map(aligned_navs: pd.DataFrame) -> Dict[str, float]:
    changes: Dict[str, float] = {}
    for code in aligned_navs.columns:
        series = pd.to_numeric(aligned_navs[code], errors="coerce").dropna()
        if len(series) < 252:
            continue
        ema252 = series.ewm(span=252, adjust=False, min_periods=252).mean().iloc[-1]
        latest_nav = series.iloc[-1]
        if pd.notna(ema252) and ema252 > 0 and latest_nav > 0:
            changes[str(code)] = float(latest_nav / ema252 - 1.0)
    return changes


def build_rebalancing_plan(
    current_allocation: pd.DataFrame,
    optimal_weights: np.ndarray,
    log_returns: pd.DataFrame,
    tranches: int,
    minimum_change_amount: float,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    codes = [str(column) for column in log_returns.columns]
    allocation = current_allocation.set_index("Scheme Code").copy()
    latest_nav_map = allocation["Latest NAV"].to_dict()

    common_codes = [
        code
        for code in codes
        if code in allocation.index
        and code in latest_nav_map
        and pd.notna(latest_nav_map[code])
        and float(latest_nav_map[code]) > 0
    ]
    missing_navs = [code for code in codes if code not in latest_nav_map or pd.isna(latest_nav_map.get(code))]
    missing_allocation = [code for code in codes if code not in allocation.index]

    if not common_codes:
        raise ValueError("No common schemes exist between allocation, NAV history, and latest NAVs.")

    positions = [codes.index(code) for code in common_codes]
    aligned_optimal = np.asarray(optimal_weights, dtype=float)[positions]
    aligned_optimal = aligned_optimal / aligned_optimal.sum()

    current_values = allocation.loc[common_codes, "Current Value"].to_numpy(dtype=float)
    portfolio_value = float(current_values.sum())
    current_weights = current_values / portfolio_value
    target_values = portfolio_value * aligned_optimal
    signed_change = target_values - current_values
    absolute_change = np.abs(signed_change)
    latest_navs = np.array([float(latest_nav_map[code]) for code in common_codes])
    total_units = absolute_change / latest_navs
    tranches = max(int(tranches), 1)
    tranche_amount = absolute_change / tranches
    tranche_units = total_units / tranches

    plan = pd.DataFrame(
        {
            "Scheme Code": common_codes,
            "Scheme Name": allocation.loc[common_codes, "Scheme Name"].values,
            "Latest NAV": latest_navs,
            "Current Weight": current_weights,
            "Optimal Weight": aligned_optimal,
            "Action": np.where(signed_change >= 0, "Invest", "Redeem"),
            "Weight Change": np.abs(aligned_optimal - current_weights),
            "Total Change Amount": absolute_change,
            "Total Units": total_units,
            "Per-Tranche Amount": tranche_amount,
            "Per-Tranche Units": tranche_units,
            "Tranches": tranches,
        }
    )
    plan = (
        plan[plan["Total Change Amount"] >= float(minimum_change_amount)]
        .sort_values("Total Change Amount", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return plan, missing_navs, missing_allocation


def style_rebalance_df(df: pd.DataFrame):
    def color_action_row(row):
        if row.get("Action") == "Invest":
            return ["background-color: #d4edda; color: #155724;"] * len(row)
        return ["background-color: #f8d7da; color: #721c24;"] * len(row)

    formatters = {
        "Latest NAV": "₹{:,.4f}",
        "Current Weight": "{:.2%}",
        "Optimal Weight": "{:.2%}",
        "Weight Change": "{:.2%}",
        "Change from 252-day EMA": "{:.2%}",
        "Total Change Amount": "₹{:,.2f}",
        "Total Units": "{:,.6f}",
        "Per-Tranche Amount": "₹{:,.2f}",
        "Per-Tranche Units": "{:,.6f}",
    }
    applicable = {key: value for key, value in formatters.items() if key in df.columns}
    return df.style.apply(color_action_row, axis=1).format(applicable, na_rep="N/A")


def metrics_df(stats: dict) -> pd.DataFrame:
    return pd.DataFrame({"Metric": list(stats.keys()), "Value": list(stats.values())})


def format_saved_metric(metric_name: str, value) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if metric_name == "Sharpe Ratio":
        return f"{numeric:.2f}"
    return f"{numeric:.2%}"


def render_live_holdings_banner(placeholder) -> int:
    try:
        unique_count = get_unique_holdings_count()
    except sqlite3.Error as exc:
        placeholder.error(
            f"Mutual-fund database is unavailable: {exc}. Active path: {DB_PATH}"
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
                ● LIVE MUTUAL-FUND DATABASE STATUS
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; margin-top: 4px;">
                Current unique schemes: {unique_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return unique_count


def render_saved_analysis(placeholder) -> None:
    placeholder.empty()
    with placeholder.container():
        payload = load_latest_analysis()
        if payload is None:
            st.info("No saved MF analysis yet. Run optimization or restore an analysis backup.")
            return

        saved_at = str(payload.get("saved_at") or "Unknown")
        holdings_analyzed = int(payload.get("holdings_analyzed") or 0)
        total_current_value = float(payload.get("total_current_value") or 0.0)
        rebalancing_plan = payload.get("rebalancing_plan") or []
        trade_count = int(payload.get("executable_trade_count") or len(rebalancing_plan))

        st.subheader("💾 Last Saved MF Analysis")
        st.caption(f"Saved at: {saved_at}")

        history = payload.get("history") or {}
        valid_start = pd.to_datetime(history.get("valid_start"), errors="coerce")
        valid_end = pd.to_datetime(history.get("valid_end"), errors="coerce")
        return_rows = history.get("log_return_rows")
        schemes = history.get("log_return_columns")

        parts = []
        if pd.notna(valid_start) and pd.notna(valid_end):
            parts.append(
                f"**NAV period:** {valid_start.strftime('%d %b %Y')} → {valid_end.strftime('%d %b %Y')}"
            )
        if return_rows is not None:
            parts.append(f"**Common return observations:** {int(return_rows):,}")
        if schemes is not None:
            parts.append(f"**Schemes optimized:** {int(schemes):,}")
        if parts:
            st.info("  |  ".join(parts))

        col1, col2, col3 = st.columns(3)
        col1.metric("Schemes analysed", holdings_analyzed)
        col2.metric("Current portfolio value", f"₹{total_current_value:,.2f}")
        col3.metric("Rebalance actions", trade_count)

        backup = json.dumps(
            make_json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        backup_date = saved_at[:10] if saved_at and saved_at != "Unknown" else "latest"
        st.download_button(
            "Download complete MF analysis backup JSON",
            data=backup,
            file_name=f"mf_portfolio_analysis_{backup_date}.json",
            mime="application/json",
            use_container_width=True,
            key="download_saved_mf_analysis_" + saved_at,
        )

        current_stats = payload.get("current_stats") or {}
        optimal_stats = payload.get("optimal_stats") or {}
        if current_stats or optimal_stats:
            with st.expander("Saved portfolio statistics", expanded=False):
                left, right = st.columns(2)
                with left:
                    st.markdown("**Current Portfolio**")
                    if current_stats:
                        frame = pd.DataFrame(
                            {
                                "Metric": list(current_stats.keys()),
                                "Value": [
                                    format_saved_metric(metric, current_stats[metric])
                                    for metric in current_stats
                                ],
                            }
                        )
                        st.dataframe(frame, use_container_width=True, hide_index=True)
                with right:
                    st.markdown("**Optimized Portfolio**")
                    if optimal_stats:
                        frame = pd.DataFrame(
                            {
                                "Metric": list(optimal_stats.keys()),
                                "Value": [
                                    format_saved_metric(metric, optimal_stats[metric])
                                    for metric in optimal_stats
                                ],
                            }
                        )
                        st.dataframe(frame, use_container_width=True, hide_index=True)

        with st.expander("Saved rebalancing plan", expanded=True):
            if rebalancing_plan:
                frame = pd.DataFrame(rebalancing_plan)
                if "Optimal Weight" in frame.columns:
                    frame = frame.sort_values(
                        "Optimal Weight",
                        ascending=False,
                        na_position="last",
                        kind="stable",
                    ).reset_index(drop=True)
                if "Action" in frame.columns:
                    st.dataframe(style_rebalance_df(frame), use_container_width=True, hide_index=True)
                else:
                    st.dataframe(frame, use_container_width=True, hide_index=True)
            else:
                st.success("The saved analysis had no rebalance actions above the threshold.")


# =========================================================
# COVERAGE PREVIEW CALLBACKS
# =========================================================


def calculate_drop_bottom_pct_recommendation(nav_master: pd.DataFrame) -> None:
    st.session_state.pop("drop_bottom_auto_result", None)
    st.session_state.pop("drop_bottom_auto_error", None)
    try:
        portfolio_df, _ = build_current_allocation_from_db(nav_master)
        if portfolio_df.empty:
            raise ValueError("No usable mutual-fund holdings are available.")
        selected = find_drop_bottom_pct_nearest_target(
            tuple(portfolio_df["Scheme Code"].astype(str)),
            target_observations=252,
        )
        st.session_state["drop_bottom_auto_result"] = selected
    except Exception as exc:
        st.session_state["drop_bottom_auto_error"] = str(exc)


def clear_drop_bottom_coverage_preview() -> None:
    st.session_state.pop("drop_bottom_coverage_preview", None)
    st.session_state.pop("drop_bottom_coverage_error", None)


def calculate_drop_bottom_coverage_preview(
    nav_master: pd.DataFrame,
    drop_bottom_pct: float,
) -> dict:
    portfolio_df, invalid_codes = build_current_allocation_from_db(nav_master)
    if portfolio_df.empty:
        raise ValueError("No usable mutual-fund holdings are available.")

    log_returns, meta = get_daily_log_returns(
        tuple(portfolio_df["Scheme Code"].astype(str)),
        drop_bottom_pct=float(drop_bottom_pct),
    )
    return {
        "drop_bottom_pct": float(drop_bottom_pct),
        "common_return_observations": int(log_returns.shape[0]),
        "schemes": int(log_returns.shape[1]),
        "valid_start": str(meta["valid_start"]),
        "valid_end": str(meta["valid_end"]),
        "invalid_holding_rows": len(invalid_codes),
        "history_errors": meta["history_errors"],
    }


# =========================================================
# STREAMLIT UI
# =========================================================

try:
    recovered_db_path = init_holdings_db()
except sqlite3.Error as exc:
    st.error(f"Could not initialize the MF holdings database: {exc}. Path: {DB_PATH}")
    st.stop()

st.title("📈 Indian Mutual Fund Portfolio Rebalancer")
st.caption(
    "Latest scheme master and NAV: official AMFI NAVAll.txt. "
    "Historical NAVs: mftool, with a direct historical-API fallback."
)
st.caption("Mutual funds are identified by AMFI Scheme Code; no Yahoo Finance ticker is used.")
st.caption(f"Active SQLite file: `{DB_PATH}`")
st.caption(f"App build: `{APP_BUILD}`")
st.caption("Analysis support only; review scheme rules, taxes, exit loads, and suitability before transacting.")

if recovered_db_path is not None:
    st.warning(
        "A corrupt database was preserved as "
        f"`{recovered_db_path.name}` and a new MF database was created."
    )

try:
    with st.spinner("Loading AMFI scheme master and latest NAVs..."):
        amfi_nav_master = load_amfi_nav_data()
except Exception as exc:
    st.error(f"Could not load the AMFI NAV database: {exc}")
    st.stop()

live_count_banner_placeholder = st.empty()
saved_analysis_placeholder = st.empty()
update_messages: List[str] = []
update_warnings: List[str] = []
update_errors: List[str] = []

if "holdings_editor_version" not in st.session_state:
    st.session_state["holdings_editor_version"] = 0

for flash_key, target in (
    ("holdings_flash_success", update_messages),
    ("holdings_flash_warning", update_warnings),
    ("holdings_flash_error", update_errors),
):
    flash = st.session_state.pop(flash_key, None)
    if flash:
        target.append(flash)

with st.sidebar:
    st.header("MF holdings database")
    sidebar_count_placeholder = st.empty()

    search_text = st.text_input(
        "Search AMFI scheme name",
        placeholder="Example: HDFC Mid Cap Direct Growth",
    )
    search_results = pd.DataFrame()
    if search_text.strip():
        mask = (
            amfi_nav_master["Scheme Name"].str.contains(
                re.escape(search_text.strip()),
                case=False,
                na=False,
            )
            | amfi_nav_master["AMC"].str.contains(
                re.escape(search_text.strip()),
                case=False,
                na=False,
            )
        )
        search_results = amfi_nav_master.loc[
            mask,
            ["Scheme Code", "Scheme Name", "Latest NAV", "NAV Date", "AMC"],
        ].head(50)

        if search_results.empty:
            st.caption("No AMFI schemes matched the search.")
        else:
            st.dataframe(
                search_results,
                use_container_width=True,
                hide_index=True,
                height=220,
                column_config={
                    "Latest NAV": st.column_config.NumberColumn(format="₹%.4f"),
                    "NAV Date": st.column_config.DateColumn(format="DD-MMM-YYYY"),
                },
            )

    quick_add_code = ""
    if not search_results.empty:
        options = [""] + search_results.apply(
            lambda row: f"{row['Scheme Code']} | {row['Scheme Name']}", axis=1
        ).tolist()
        quick_add_selection = st.selectbox(
            "Quick-add one search result",
            options=options,
            index=0,
        )
        if quick_add_selection:
            quick_add_code = quick_add_selection.split("|", 1)[0].strip()

    add_input = st.text_area(
        "Add AMFI scheme codes",
        placeholder="119597, 120505",
        help="Enter scheme codes separated by commas, semicolons, or new lines.",
    )
    remove_input = st.text_area(
        "Remove AMFI scheme codes",
        placeholder="119597",
    )
    update_holdings_btn = st.button("Update MF holdings", use_container_width=True)
    refresh_nav_btn = st.button("Refresh latest AMFI NAVs", use_container_width=True)

    st.divider()
    st.header("Holdings backup and restore")
    backup_date = datetime.now().strftime("%Y-%m-%d")
    st.download_button(
        "Download MF holdings backup CSV",
        data=holdings_backup_bytes(),
        file_name=f"mf_holdings_backup_{backup_date}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=get_unique_holdings_count() == 0,
    )
    holdings_backup_upload = st.file_uploader(
        "Upload MF holdings backup CSV",
        type=["csv"],
        key="mf_holdings_backup_upload",
    )
    holdings_restore_choice = st.radio(
        "Restore behaviour",
        options=["Merge/update current holdings", "Replace current holdings"],
        index=0,
    )
    restore_holdings_btn = st.button(
        "Restore uploaded MF holdings",
        use_container_width=True,
        disabled=holdings_backup_upload is None,
    )

    st.divider()
    st.header("Analysis backup")
    existing_analysis_backup = latest_analysis_backup_bytes()
    if existing_analysis_backup is not None:
        latest_payload = load_latest_analysis() or {}
        saved_date = str(latest_payload.get("saved_at", ""))[:10] or "latest"
        st.download_button(
            "Download latest MF analysis JSON",
            data=existing_analysis_backup,
            file_name=f"mf_analysis_backup_{saved_date}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Run optimization once before downloading an analysis backup.")

    analysis_backup_upload = st.file_uploader(
        "Upload MF analysis backup JSON",
        type=["json"],
        key="mf_analysis_backup_upload",
    )
    restore_analysis_btn = st.button(
        "Restore uploaded MF analysis",
        use_container_width=True,
        disabled=analysis_backup_upload is None,
    )

    st.divider()
    st.header("Analysis inputs")
    tranches = st.number_input(
        "Rebalance over (days/tranches)",
        min_value=1,
        value=13,
        step=1,
    )
    minimum_change_amount = st.number_input(
        "Minimum total rebalance amount (₹)",
        min_value=0.0,
        value=100.0,
        step=100.0,
    )
    max_drawdown_pct = st.number_input(
        "Maximum drawdown constraint (%)",
        min_value=0.0,
        max_value=100.0,
        value=23.0,
        step=0.1,
    )
    max_drawdown = float(max_drawdown_pct) / 100.0
    risk_free_rate_pct = st.number_input(
        "Annual risk-free rate input (%)",
        min_value=0.0,
        max_value=30.0,
        value=6.5,
        step=0.1,
        help="A modelling input used for the Sharpe ratio; adjust it for your analysis.",
    )
    risk_free_rate_annual = float(risk_free_rate_pct) / 100.0

    drop_bottom_pct = float(
        st.number_input(
            "Drop bottom fraction of schemes by history length",
            min_value=0.0,
            max_value=0.95,
            value=0.0,
            step=0.01,
            format="%.2f",
            key="mf_manual_drop_bottom_pct",
            on_change=clear_drop_bottom_coverage_preview,
        )
    )

    preview_coverage_btn = st.button(
        "Preview common NAV coverage",
        use_container_width=True,
    )
    if preview_coverage_btn:
        st.session_state.pop("drop_bottom_coverage_preview", None)
        st.session_state.pop("drop_bottom_coverage_error", None)
        try:
            with st.spinner("Loading mutual-fund NAV histories..."):
                st.session_state["drop_bottom_coverage_preview"] = (
                    calculate_drop_bottom_coverage_preview(
                        amfi_nav_master,
                        drop_bottom_pct,
                    )
                )
        except Exception as exc:
            st.session_state["drop_bottom_coverage_error"] = str(exc)

    coverage_preview = st.session_state.get("drop_bottom_coverage_preview")
    if coverage_preview:
        st.info(
            f"**Drop fraction:** `{coverage_preview['drop_bottom_pct']:.2f}`  |  "
            f"**Common return observations:** {coverage_preview['common_return_observations']:,}  |  "
            f"**Schemes:** {coverage_preview['schemes']:,}"
        )
        st.caption("Coverage preview only; optimization has not been run.")
    coverage_error = st.session_state.get("drop_bottom_coverage_error")
    if coverage_error:
        st.error(f"Could not calculate common NAV coverage: {coverage_error}")

    if st.button("Calculate 252-observation recommendation", use_container_width=True):
        calculate_drop_bottom_pct_recommendation(amfi_nav_master)
    auto_result = st.session_state.get("drop_bottom_auto_result")
    if auto_result:
        st.info(
            f"Suggested drop fraction: {auto_result['drop_bottom_pct']:.2f} — "
            f"{auto_result['common_return_observations']:,} common return observations."
        )
    auto_error = st.session_state.get("drop_bottom_auto_error")
    if auto_error:
        st.error(f"Could not calculate the recommendation: {auto_error}")

    use_target_volatility = st.checkbox("Use target daily volatility")
    target_daily_volatility = (
        st.number_input(
            "Target daily volatility",
            min_value=0.0001,
            value=0.0145,
            step=0.0001,
            format="%.5f",
        )
        if use_target_volatility
        else None
    )

    run_btn = st.button("Run MF optimization", use_container_width=True, type="primary")


if restore_holdings_btn:
    try:
        mode = "replace" if holdings_restore_choice == "Replace current holdings" else "merge"
        restored = restore_holdings_backup(holdings_backup_upload, mode=mode)
        refresh_holdings_latest_nav(amfi_nav_master)
        st.session_state["holdings_editor_version"] += 1
        st.session_state["holdings_flash_success"] = (
            f"Restored {restored} mutual-fund holdings using {mode} mode."
        )
        st.rerun()
    except Exception as exc:
        update_errors.append(f"Could not restore MF holdings: {exc}")

if restore_analysis_btn:
    try:
        restored_payload = restore_latest_analysis_backup(analysis_backup_upload)
        update_messages.append(
            "MF analysis backup restored. Saved at: "
            + str(restored_payload.get("saved_at", "Unknown"))
        )
    except Exception as exc:
        update_errors.append(f"Could not restore MF analysis: {exc}")

if refresh_nav_btn:
    try:
        load_amfi_nav_data.clear()
        amfi_nav_master = load_amfi_nav_data()
        missing = refresh_holdings_latest_nav(amfi_nav_master)
        if missing:
            update_warnings.append(
                "Latest AMFI NAV was unavailable for: " + ", ".join(missing)
            )
        update_messages.append("Latest AMFI NAVs refreshed in the local database.")
    except Exception as exc:
        update_errors.append(f"Could not refresh AMFI NAVs: {exc}")

if update_holdings_btn:
    add_codes = parse_scheme_code_input(add_input)
    if quick_add_code:
        add_codes = list(dict.fromkeys(add_codes + [quick_add_code]))
    remove_codes = parse_scheme_code_input(remove_input)
    overlap = sorted(set(add_codes) & set(remove_codes))

    if overlap:
        update_errors.append(
            "The same scheme code cannot be in both Add and Remove: " + ", ".join(overlap)
        )
    elif not add_codes and not remove_codes:
        update_warnings.append("Enter at least one AMFI scheme code to add or remove.")
    else:
        removed, not_held = remove_scheme_codes_from_master(remove_codes)
        added, duplicates, invalid, missing_nav = add_scheme_codes_to_master(
            add_codes,
            amfi_nav_master,
        )

        if added:
            update_messages.append("Added AMFI scheme codes: " + ", ".join(added))
        if removed:
            update_messages.append("Removed AMFI scheme codes: " + ", ".join(removed))
        if duplicates:
            update_warnings.append("Already in holdings: " + ", ".join(duplicates))
        if not_held:
            update_warnings.append("Not present in holdings: " + ", ".join(not_held))
        if invalid:
            update_errors.append("Not found in AMFI NAVAll.txt: " + ", ".join(invalid))
        if missing_nav:
            update_warnings.append(
                "Added without a usable latest NAV; edit after AMFI publishes a NAV: "
                + ", ".join(missing_nav)
            )
        if added or removed:
            st.session_state["holdings_editor_version"] += 1

for message in update_messages:
    st.success(message)
for message in update_warnings:
    st.warning(message)
for message in update_errors:
    st.error(message)

live_count = render_live_holdings_banner(live_count_banner_placeholder)
sidebar_count_placeholder.metric("Current unique schemes", live_count)
render_saved_analysis(saved_analysis_placeholder)

st.subheader("Master Mutual-Fund Holdings")
master_df = load_master_holdings()

if master_df.empty:
    st.info("The MF holdings table is empty. Search AMFI and add scheme codes from the sidebar.")
else:
    display_master = master_df.copy()
    display_master["Current Value"] = (
        pd.to_numeric(display_master["Units"], errors="coerce")
        * pd.to_numeric(display_master["Latest NAV"], errors="coerce")
    )
    st.dataframe(
        display_master,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Units": st.column_config.NumberColumn(format="%.6f"),
            "Average NAV": st.column_config.NumberColumn(format="₹%.4f"),
            "Latest NAV": st.column_config.NumberColumn(format="₹%.4f"),
            "Current Value": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )

    with st.expander("Edit units and average purchase NAV", expanded=False):
        st.caption(
            "Latest NAV is refreshed from AMFI. Units and Average NAV are your editable holding inputs."
        )
        editable_df = master_df[
            ["Scheme Code", "Scheme Name", "Units", "Average NAV", "Latest NAV", "NAV Date"]
        ].copy()
        editable_df["_sort_value"] = (
            pd.to_numeric(editable_df["Units"], errors="coerce").fillna(0)
            * pd.to_numeric(editable_df["Latest NAV"], errors="coerce").fillna(0)
        )
        editable_df = (
            editable_df.sort_values("_sort_value", ascending=False, kind="stable")
            .drop(columns="_sort_value")
            .reset_index(drop=True)
        )
        editor_version = st.session_state["holdings_editor_version"]

        with st.form(f"mf_holdings_form_{editor_version}", clear_on_submit=False):
            edited_df = st.data_editor(
                editable_df,
                use_container_width=True,
                hide_index=True,
                disabled=["Scheme Code", "Scheme Name", "Latest NAV", "NAV Date"],
                column_config={
                    "Units": st.column_config.NumberColumn(
                        min_value=0.000001,
                        format="%.6f",
                    ),
                    "Average NAV": st.column_config.NumberColumn(
                        min_value=0.0001,
                        format="₹%.4f",
                    ),
                    "Latest NAV": st.column_config.NumberColumn(format="₹%.4f"),
                },
                key=f"mf_holdings_editor_{editor_version}",
            )
            save_holdings_btn = st.form_submit_button(
                "Save units and average NAV",
                use_container_width=True,
                type="primary",
            )

        if save_holdings_btn:
            try:
                updated = save_holding_values(edited_df)
                st.session_state["holdings_editor_version"] += 1
                st.session_state["holdings_flash_success"] = (
                    f"Saved Units and Average NAV for {updated} mutual-fund holdings."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save MF holdings: {exc}")


if run_btn:
    try:
        if master_df.empty:
            st.error("Add at least one AMFI scheme code before running optimization.")
            st.stop()

        with st.spinner("Refreshing latest AMFI NAVs and building current allocation..."):
            portfolio_df, invalid_holding_codes = build_current_allocation_from_db(
                amfi_nav_master
            )

        if portfolio_df.empty:
            st.error(
                "No usable MF holdings were found. Check Units, Average NAV, and latest AMFI NAV."
            )
            st.stop()

        if invalid_holding_codes:
            st.warning(
                "Skipped holdings with invalid/missing Units or NAV: "
                + ", ".join(invalid_holding_codes)
            )

        left, right = st.columns([3, 1])
        with left:
            st.subheader("Current MF Allocation")
            st.dataframe(
                portfolio_df.style.format(
                    {
                        "Units": "{:,.6f}",
                        "Average NAV": "₹{:,.4f}",
                        "Latest NAV": "₹{:,.4f}",
                        "Invested Amount": "₹{:,.2f}",
                        "Current Value": "₹{:,.2f}",
                        "Unrealized P&L": "₹{:,.2f}",
                        "Return %": "{:.2%}",
                        "Weight": "{:.2%}",
                    },
                    na_rep="N/A",
                ),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            total_invested = float(portfolio_df["Invested Amount"].sum())
            total_current_value = float(portfolio_df["Current Value"].sum())
            total_pnl = total_current_value - total_invested
            st.metric("Schemes", len(portfolio_df))
            st.metric("Invested amount", f"₹{total_invested:,.2f}")
            st.metric("Current value", f"₹{total_current_value:,.2f}")
            st.metric("Unrealized P&L", f"₹{total_pnl:,.2f}")

        scheme_codes = portfolio_df["Scheme Code"].astype(str).tolist()
        with st.spinner("Loading historical NAVs and running MF optimization..."):
            (
                optimal_weights,
                log_returns,
                current_stats,
                optimal_stats,
                meta,
                optimization_method,
            ) = run_portfolio_analysis(
                scheme_codes=scheme_codes,
                current_allocation=portfolio_df,
                max_drawdown=max_drawdown,
                risk_free_rate_annual=risk_free_rate_annual,
                target_daily_volatility=target_daily_volatility,
                drop_bottom_pct=drop_bottom_pct,
            )

        if optimal_weights is None:
            st.error("Mutual-fund portfolio optimization did not return a usable allocation.")
            st.stop()

        name_lookup = portfolio_df.set_index("Scheme Code")["Scheme Name"].to_dict()

        if meta["history_errors"]:
            st.warning(
                "Historical NAV could not be loaded for: "
                + ", ".join(sorted(meta["history_errors"]))
            )

        if not meta["dropped_df"].empty:
            dropped_display = meta["dropped_df"].copy()
            dropped_display.insert(
                1,
                "Scheme Name",
                dropped_display["Scheme Code"].map(name_lookup),
            )
            st.subheader("Schemes Dropped by History-Length Filter")
            st.dataframe(dropped_display, use_container_width=True, hide_index=True)

        st.subheader("Historical NAV Coverage")
        min_history_display = meta["min_len_df"].copy()
        min_history_display.insert(
            1,
            "Scheme Name",
            min_history_display["Scheme Code"].map(name_lookup),
        )
        st.dataframe(
            min_history_display.style.format(
                {"Since-first-observation Return": "{:.2%}"},
                na_rep="N/A",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            f"**Drop fraction used:** `{drop_bottom_pct:.2f}`  |  "
            f"**Common return observations:** {int(log_returns.shape[0]):,}  |  "
            f"**Schemes in return matrix:** {int(log_returns.shape[1]):,}"
        )
        st.caption(
            f"Overlapping NAV dates: {meta['valid_start'].date()} to {meta['valid_end'].date()}"
        )
        st.caption(f"Optimization method used: {optimization_method}")

        st.subheader("Portfolio Statistics Comparison")
        current_col, optimal_col = st.columns(2)
        with current_col:
            st.markdown("**Current Portfolio**")
            current_frame = metrics_df(current_stats)
            current_frame["Value"] = current_frame.apply(
                lambda row: (
                    f"{row['Value']:.2f}"
                    if row["Metric"] == "Sharpe Ratio"
                    else f"{row['Value']:.2%}"
                ),
                axis=1,
            )
            st.dataframe(current_frame, use_container_width=True, hide_index=True)
        with optimal_col:
            st.markdown("**Optimized Portfolio**")
            optimal_frame = metrics_df(optimal_stats)
            optimal_frame["Value"] = optimal_frame.apply(
                lambda row: (
                    f"{row['Value']:.2f}"
                    if row["Metric"] == "Sharpe Ratio"
                    else f"{row['Value']:.2%}"
                ),
                axis=1,
            )
            st.dataframe(optimal_frame, use_container_width=True, hide_index=True)

        st.subheader("Top Correlated Mutual-Fund Pairs")
        top_correlations = pd.DataFrame(
            columns=[
                "Scheme Code 1",
                "Scheme Name 1",
                "Scheme Code 2",
                "Scheme Name 2",
                "Absolute Correlation",
            ]
        )
        correlation_matrix = log_returns.corr()
        if correlation_matrix.shape[1] < 2:
            st.info("At least two schemes are required to show correlated pairs.")
        else:
            upper_mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)
            stacked = correlation_matrix.where(upper_mask).stack()
            if stacked.empty:
                st.info("No valid mutual-fund correlation pairs were found.")
            else:
                top_correlations = pd.DataFrame(
                    [
                        {
                            "Scheme Code 1": str(index[0]),
                            "Scheme Name 1": name_lookup.get(str(index[0]), ""),
                            "Scheme Code 2": str(index[1]),
                            "Scheme Name 2": name_lookup.get(str(index[1]), ""),
                            "Absolute Correlation": abs(float(value)),
                        }
                        for index, value in stacked.items()
                    ]
                ).sort_values("Absolute Correlation", ascending=False, kind="stable")
                st.dataframe(
                    top_correlations.head(5).style.format(
                        {"Absolute Correlation": "{:.3f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        rebalancing_df, missing_latest_navs, missing_allocation = build_rebalancing_plan(
            current_allocation=portfolio_df,
            optimal_weights=optimal_weights,
            log_returns=log_returns,
            tranches=int(tranches),
            minimum_change_amount=float(minimum_change_amount),
        )
        ema252_map = calculate_ema252_change_map(meta["aligned_navs"])
        if not rebalancing_df.empty:
            rebalancing_df.insert(
                7,
                "Change from 252-day EMA",
                rebalancing_df["Scheme Code"].map(ema252_map),
            )

        if missing_latest_navs:
            st.warning("Skipped schemes with missing latest NAV: " + ", ".join(missing_latest_navs))
        if missing_allocation:
            st.warning("Skipped schemes missing from current allocation: " + ", ".join(missing_allocation))

        st.subheader("Mutual-Fund Rebalancing Plan")
        st.caption(
            "Invest/Redeem amounts are split into the selected number of tranches. "
            "Fractional mutual-fund units are retained instead of stock-style integer rounding."
        )
        if rebalancing_df.empty:
            st.success("No rebalance action exceeds the minimum amount threshold.")
        else:
            st.dataframe(
                style_rebalance_df(rebalancing_df),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download MF rebalancing plan CSV",
                data=rebalancing_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="mf_rebalancing_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )

        analysis_payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "data_sources": {
                "latest_nav": AMFI_NAV_URL,
                "historical_nav": "mftool with mfapi.in fallback",
            },
            "holdings_analyzed": int(len(portfolio_df)),
            "total_invested": total_invested,
            "total_current_value": total_current_value,
            "total_unrealized_pnl": total_pnl,
            "executable_trade_count": int(len(rebalancing_df)),
            "optimization_method": optimization_method,
            "settings": {
                "tranches": int(tranches),
                "minimum_change_amount": float(minimum_change_amount),
                "max_drawdown_input_pct": float(max_drawdown_pct),
                "max_drawdown": float(max_drawdown),
                "risk_free_rate_input_pct": float(risk_free_rate_pct),
                "risk_free_rate_annual": float(risk_free_rate_annual),
                "drop_bottom_fraction": float(drop_bottom_pct),
                "use_target_daily_volatility": bool(use_target_volatility),
                "target_daily_volatility": (
                    float(target_daily_volatility)
                    if target_daily_volatility is not None
                    else None
                ),
            },
            "history": {
                "valid_start": str(meta["valid_start"]),
                "valid_end": str(meta["valid_end"]),
                "log_return_rows": int(log_returns.shape[0]),
                "log_return_columns": int(log_returns.shape[1]),
                "minimum_history": meta["min_len_df"].to_dict(orient="records"),
                "dropped_schemes": meta["dropped_df"].to_dict(orient="records"),
            },
            "current_stats": current_stats,
            "optimal_stats": optimal_stats,
            "top_correlations": top_correlations.head(5).to_dict(orient="records"),
            "current_allocation": portfolio_df.to_dict(orient="records"),
            "rebalancing_plan": rebalancing_df.to_dict(orient="records"),
            "warnings": {
                "invalid_holding_codes": invalid_holding_codes,
                "history_errors": meta["history_errors"],
                "missing_latest_navs": missing_latest_navs,
                "missing_allocation": missing_allocation,
            },
        }
        save_latest_analysis(analysis_payload)
        render_saved_analysis(saved_analysis_placeholder)
        st.success(
            "MF analysis completed and saved. The complete result is available as JSON."
        )

    except Exception as exc:
        st.error(f"Error: {exc}")
