import io
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

DEFAULT_DB_PATH = Path(__file__).resolve().with_name("portfolio_holdings.db")
DB_PATH = Path(os.getenv("PORTFOLIO_DB_PATH", str(DEFAULT_DB_PATH)))


def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_holdings_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS master_holdings (
                symbol TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                quantity REAL NOT NULL DEFAULT 1,
                average_price REAL,
                added_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


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
    # Derived display/export column. This is calculated from the saved
    # Quantity and Average Price; it is not independently stored in SQLite.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Price"] = pd.to_numeric(df["Average Price"], errors="coerce")
    df["Current (Invested)"] = df["Quantity"] * df["Average Price"]

    ordered_columns = [
        "Symbol",
        "Stock Name",
        "Quantity",
        "Average Price",
        "Current (Invested)",
        "Added At",
        "Updated At",
    ]
    return df[ordered_columns]


def export_master_holdings_csv():
    """Export the complete master holdings table as a portable CSV backup."""
    df = load_master_holdings()
    return df.to_csv(index=False).encode("utf-8-sig")


def import_master_holdings_csv(uploaded_file, replace_existing=True):
    """Restore holdings from a CSV backup.

    When ``replace_existing`` is True, the current master table is replaced.
    Otherwise, uploaded rows are merged and existing symbols are updated.
    """
    if uploaded_file is None:
        raise ValueError("Choose a holdings backup CSV file first.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded backup file is empty.")

    restored = pd.read_csv(io.BytesIO(file_bytes))
    restored.columns = [str(c).strip() for c in restored.columns]

    required = ["Symbol", "Stock Name", "Quantity", "Average Price"]
    missing = [c for c in required if c not in restored.columns]
    if missing:
        raise ValueError(f"Backup is missing required columns: {missing}")

    restored = restored.copy()
    restored["Symbol"] = restored["Symbol"].map(normalize_nse_symbol)
    restored["Stock Name"] = restored["Stock Name"].fillna("").astype(str).str.strip()
    restored["Quantity"] = pd.to_numeric(restored["Quantity"], errors="coerce")
    restored["Average Price"] = pd.to_numeric(restored["Average Price"], errors="coerce")

    restored = restored[restored["Symbol"] != ""].copy()
    restored = restored.drop_duplicates(subset=["Symbol"], keep="last")

    if restored.empty:
        raise ValueError("The backup does not contain any valid symbols.")
    if restored["Quantity"].isna().any() or (restored["Quantity"] <= 0).any():
        raise ValueError("Every restored holding must have Quantity greater than zero.")

    invalid_prices = restored["Average Price"].notna() & (restored["Average Price"] <= 0)
    if invalid_prices.any():
        raise ValueError("Average Price must be greater than zero when provided.")

    now = datetime.now().isoformat(timespec="seconds")
    if "Added At" not in restored.columns:
        restored["Added At"] = now
    if "Updated At" not in restored.columns:
        restored["Updated At"] = now

    restored["Added At"] = restored["Added At"].fillna(now).astype(str)
    restored["Updated At"] = restored["Updated At"].fillna(now).astype(str)
    restored["Stock Name"] = restored.apply(
        lambda row: row["Stock Name"] if row["Stock Name"] else row["Symbol"], axis=1
    )

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN")
            if replace_existing:
                conn.execute("DELETE FROM master_holdings")

            for _, row in restored.iterrows():
                average_price = (
                    None if pd.isna(row["Average Price"]) else float(row["Average Price"])
                )
                conn.execute(
                    """
                    INSERT INTO master_holdings
                        (symbol, stock_name, quantity, average_price, added_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        stock_name = excluded.stock_name,
                        quantity = excluded.quantity,
                        average_price = excluded.average_price,
                        updated_at = excluded.updated_at
                    """,
                    (
                        row["Symbol"],
                        row["Stock Name"],
                        float(row["Quantity"]),
                        average_price,
                        row["Added At"],
                        row["Updated At"],
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return len(restored)


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
    """Render an always-visible live holdings-count banner."""
    unique_count = get_unique_holdings_count()
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
    """Persist edited quantity and average price values without resetting them."""
    required = ["Symbol", "Quantity", "Average Price"]
    missing_columns = [c for c in required if c not in edited_df.columns]
    if missing_columns:
        raise ValueError(f"Missing editable holding columns: {missing_columns}")

    cleaned = edited_df.copy()
    cleaned["Symbol"] = cleaned["Symbol"].map(normalize_nse_symbol)
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["Average Price"] = pd.to_numeric(cleaned["Average Price"], errors="coerce")

    if cleaned["Symbol"].duplicated().any():
        raise ValueError("Duplicate symbols are not allowed in the master holdings table.")
    if cleaned["Quantity"].isna().any() or (cleaned["Quantity"] <= 0).any():
        raise ValueError("Quantity must be greater than zero for every holding.")
    if cleaned["Average Price"].isna().any() or (cleaned["Average Price"] <= 0).any():
        raise ValueError("Average Price must be greater than zero for every holding.")

    now = datetime.now().isoformat(timespec="seconds")
    updated_rows = 0

    with get_db_connection() as conn:
        try:
            conn.execute("BEGIN")
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
                updated_rows += cursor.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    if updated_rows != len(cleaned):
        raise ValueError(
            f"Saved {updated_rows} of {len(cleaned)} holdings. Reload and try again."
        )

    return updated_rows


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

    usable["Current (Invested)"] = usable["Quantity"] * usable["Average Price"]
    total = usable["Current (Invested)"].sum()
    if total <= 0:
        raise ValueError("The master holdings table has no positive portfolio value.")

    usable["Weight"] = usable["Invested"] / total
    portfolio_df = usable[
        [
            "Symbol",
            "Stock Name",
            "Quantity",
            "Average Price",
            "Current (Invested)",
            "Weight",
        ]
    ].sort_values("Weight", ascending=False).reset_index(drop=True)

    return portfolio_df, invalid_rows

# =========================================================
# RETURNS / OPTIMIZATION
# =========================================================

@st.cache_data(show_spinner=False)
def get_daily_log_returns(symbols, start_date=None, end_date=None, buffer_days=7, drop_bottom_pct=0.1):
    if end_date is None:
        end_date = datetime.today()
    else:
        end_date = pd.to_datetime(end_date)

    if start_date is None:
        start_date = "2000-01-01"

    effective_end = (end_date - timedelta(days=buffer_days)).strftime("%Y-%m-%d")

    df = yf.download(symbols, start=start_date, end=effective_end, progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna(axis=1, how="all")

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


# =========================================================
# UI
# =========================================================

init_holdings_db()

st.title("📊 Portfolio Rebalancer")
st.caption("Holdings are stored in a local SQLite master table instead of being uploaded from Excel.")

if "holdings_editor_version" not in st.session_state:
    st.session_state["holdings_editor_version"] = 0

for flash_type, flash_message in st.session_state.pop("portfolio_flash_messages", []):
    if flash_type == "success":
        st.success(flash_message)
    elif flash_type == "warning":
        st.warning(flash_message)
    else:
        st.error(flash_message)

# Filled after every add/remove operation using a fresh SQLite query.
live_count_banner_placeholder = st.empty()

update_messages = []
update_warnings = []
update_errors = []

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
    st.subheader("Backup and restore")
    backup_bytes = export_master_holdings_csv()
    st.download_button(
        "Download holdings backup CSV",
        data=backup_bytes,
        file_name=f"master_holdings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=get_unique_holdings_count() == 0,
    )

    uploaded_backup = st.file_uploader(
        "Upload holdings backup CSV",
        type=["csv"],
        key="holdings_backup_uploader",
        help="Restore Quantity and Average Price exactly as saved in the backup.",
    )
    restore_mode = st.radio(
        "Restore mode",
        ["Replace current holdings", "Merge/update current holdings"],
        index=0,
    )
    restore_backup_btn = st.button(
        "Restore uploaded backup",
        use_container_width=True,
        disabled=uploaded_backup is None,
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
    drop_bottom_pct = st.slider(
        "Drop bottom fraction of tickers by history length",
        min_value=0.0,
        max_value=0.95,
        value=0.0,
        step=0.05,
    )
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
    run_btn = st.button("Run analysis", use_container_width=True, type="primary")

if restore_backup_btn:
    try:
        restored_count = import_master_holdings_csv(
            uploaded_backup,
            replace_existing=restore_mode == "Replace current holdings",
        )
        st.session_state["holdings_editor_version"] += 1
        st.session_state["portfolio_flash_messages"] = [
            ("success", f"Restored {restored_count} holdings from the uploaded backup.")
        ]
        st.rerun()
    except Exception as exc:
        update_errors.append(f"Could not restore backup: {exc}")

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

for message in update_messages:
    st.success(message)
for message in update_warnings:
    st.warning(message)
for message in update_errors:
    st.error(message)

# Fresh, uncached database count on every Streamlit rerun.
live_unique_count = render_live_holdings_banner(live_count_banner_placeholder)
sidebar_count_placeholder.metric("Current unique holdings", live_unique_count)

st.subheader("Master Holdings")
master_df = load_master_holdings()

if master_df.empty:
    st.info("The master holdings table is empty. Add NSE symbols from the sidebar.")
else:
    st.dataframe(
        master_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Quantity": st.column_config.NumberColumn("Quantity", format="%.6f"),
            "Average Price": st.column_config.NumberColumn("Average Price", format="₹%.2f"),
            "Current (Invested)": st.column_config.NumberColumn(
                "Current (Invested)",
                help="Quantity × Average Price",
                format="₹%.2f",
            ),
        },
    )

    with st.expander("Edit quantity and average price", expanded=False):
        st.caption(
            "A newly added symbol starts with quantity 1 and the latest available NSE price. "
            "After saving, your edited values are reloaded directly from SQLite and remain unchanged."
        )
        editable_df = master_df[
            ["Symbol", "Stock Name", "Quantity", "Average Price", "Current (Invested)"]
        ].copy()

        # Show the largest portfolio positions first in the edit table.
        total_invested_for_editor = pd.to_numeric(
            editable_df["Current (Invested)"], errors="coerce"
        ).fillna(0).sum()
        editable_df["Current Weight"] = (
            pd.to_numeric(editable_df["Current (Invested)"], errors="coerce").fillna(0)
            / total_invested_for_editor
            * 100
            if total_invested_for_editor > 0
            else 0.0
        )
        editable_df = editable_df.sort_values(
            by=["Current Weight", "Current (Invested)", "Symbol"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        st.caption("Edit table is sorted by Current Weight, highest to lowest.")

        with st.form("holdings_editor_form", clear_on_submit=False):
            edited_df = st.data_editor(
                editable_df,
                use_container_width=True,
                hide_index=True,
                disabled=["Symbol", "Stock Name", "Current (Invested)", "Current Weight"],
                column_config={
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity", min_value=0.000001, format="%.6f"
                    ),
                    "Average Price": st.column_config.NumberColumn(
                        "Average Price", min_value=0.01, format="₹%.2f"
                    ),
                    "Current (Invested)": st.column_config.NumberColumn(
                        "Current (Invested)",
                        help="Read-only: Quantity × Average Price. It refreshes after saving.",
                        format="₹%.2f",
                    ),
                    "Current Weight": st.column_config.NumberColumn(
                        "Current Weight",
                        help="Read-only portfolio weight based on Current (Invested).",
                        format="%.2f%%",
                    ),
                },
                key=f"holdings_editor_{st.session_state['holdings_editor_version']}",
            )
            save_holdings_btn = st.form_submit_button(
                "Save quantity and price changes",
                use_container_width=True,
                type="primary",
            )

        if save_holdings_btn:
            try:
                updated_rows = save_holding_values(edited_df)
                st.session_state["holdings_editor_version"] += 1
                st.session_state["portfolio_flash_messages"] = [
                    ("success", f"Saved Quantity and Average Price for {updated_rows} holdings.")
                ]
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save holdings: {exc}")

if run_btn:
    try:
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
                    "Current (Invested)": "₹{:,.2f}",
                    "Weight": "{:.2%}",
                }),
                use_container_width=True,
            )

        with col2:
            total_invested = float(portfolio_df["Current (Invested)"].sum())
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

        if not meta["dropped_df"].empty:
            st.subheader("Dropped Tickers")
            st.dataframe(meta["dropped_df"], use_container_width=True)

        st.subheader("History Coverage")
        st.dataframe(meta["min_len_df"], use_container_width=True)
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

    except Exception as e:
        st.error(f"Error: {e}")
