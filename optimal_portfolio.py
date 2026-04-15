import io
import warnings
from datetime import datetime, timedelta

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
    return df[['ISIN NUMBER', 'SYMBOL', 'NAME OF COMPANY']].rename(columns={
        'ISIN NUMBER': 'ISIN',
        'SYMBOL': 'Symbol',
        'NAME OF COMPANY': 'Company Name'
    })


@st.cache_data(show_spinner=False)
def resolve_yahoo_tickers(symbols_base):
    resolved = {}
    for sym in symbols_base:
        for suffix in [".NS", ".BO"]:
            try:
                test = yf.download(sym + suffix, period="5d", progress=False, auto_adjust=False)
                if test is not None and not test.empty:
                    resolved[sym] = sym + suffix
                    break
            except Exception:
                continue
    return resolved
def safe_lower(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def find_header_row(file_bytes, sheet_name=0, search_terms=None, max_rows=40):
    if search_terms is None:
        search_terms = ["Stock Name", "ISIN", "Quantity", "Average"]

    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=max_rows)

    for i in range(len(raw)):
        row_vals = raw.iloc[i].tolist()
        row_vals_norm = [safe_lower(v) for v in row_vals]

        if all(any(safe_lower(term) in cell for cell in row_vals_norm) for term in search_terms):
            return i

    return None

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")

# =========================================================
# FILE PARSERS
# =========================================================

def parse_stocks_statement(file_bytes):
    header_row = find_header_row(
        file_bytes,
        sheet_name=0,
        search_terms=["Stock Name", "ISIN", "Quantity", "Average"]
    )
    if header_row is None:
        raise ValueError("Could not detect header row for Stocks_ statement.")

    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, skiprows=header_row)
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {
        "Average buy price": "Average Price",
        "Average buy Price": "Average Price",
    }
    df.rename(columns=rename_map, inplace=True)

    required_cols = ["Stock Name", "ISIN", "Quantity", "Average Price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Stocks_ file: {missing}")

    df = df[required_cols].copy()
    df["Quantity"] = safe_numeric(df["Quantity"])
    df["Average Price"] = safe_numeric(df["Average Price"])
    df = df.dropna(subset=["ISIN", "Quantity", "Average Price"])
    df = df[df["Quantity"] > 0].copy()

    equity_map = load_equity_mapping()
    df["ISIN"] = df["ISIN"].astype(str).str.strip()
    equity_map["ISIN"] = equity_map["ISIN"].astype(str).str.strip()

    df = df.merge(equity_map, on="ISIN", how="left")

    unmatched = df[df["Symbol"].isna()][["Stock Name", "ISIN", "Quantity", "Average Price"]].copy()
    if not unmatched.empty:
        unmatched["P&L (approx)"] = unmatched["Quantity"] * unmatched["Average Price"] * -0.05

    df = df.dropna(subset=["Symbol"]).copy()
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["Invested"] = df["Quantity"] * df["Average Price"]
    total = df["Invested"].sum()
    df["Weight"] = df["Invested"] / total if total != 0 else 0

    parsed = df[["Symbol", "Stock Name", "Quantity", "Average Price", "Weight"]].sort_values(
        by="Weight", ascending=False
    ).reset_index(drop=True)

    return parsed, unmatched

def parse_holdings_statement(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    possible_sheets = [s for s in xls.sheet_names if safe_lower(s) == "equity"]
    sheet = possible_sheets[0] if possible_sheets else xls.sheet_names[0]

    header_row = find_header_row(
        file_bytes,
        sheet_name=sheet,
        search_terms=["Symbol", "ISIN", "Quantity", "Average Price"]
    )
    if header_row is None:
        raise ValueError("Could not detect header row for holdings- file.")

    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    quantity_col = None
    for candidate in ["Quantity Available", "Quantity", "Available Quantity"]:
        if candidate in df.columns:
            quantity_col = candidate
            break

    if quantity_col is None:
        raise ValueError(f"Quantity column not found. Found: {df.columns.tolist()}")

    if "Symbol" not in df.columns or "Average Price" not in df.columns:
        raise ValueError(f"Required columns missing in holdings- file. Found: {df.columns.tolist()}")

    stock_name_col = None
    for candidate in ["Company Name", "Stock Name", "Name"]:
        if candidate in df.columns:
            stock_name_col = candidate
            break

    df = df.copy()
    df["Symbol"] = df["Symbol"].apply(lambda x: str(x).strip().upper() if pd.notna(x) else "")
    df["Quantity"] = safe_numeric(df[quantity_col])
    df["Average Price"] = safe_numeric(df["Average Price"])

    if stock_name_col:
        df["Stock Name"] = df[stock_name_col].apply(lambda x: str(x).strip() if pd.notna(x) else "")
    else:
        df["Stock Name"] = df["Symbol"]

    df = df.dropna(subset=["Quantity", "Average Price"])
    df = df[df["Quantity"] > 0].copy()
    df = df[df["Symbol"].apply(lambda x: safe_lower(x) not in ["", "nan", "total", "summary"])].copy()

    df["Invested"] = df["Quantity"] * df["Average Price"]
    total = df["Invested"].sum()
    df["Weight"] = df["Invested"] / total if total != 0 else 0

    parsed = df[["Symbol", "Stock Name", "Quantity", "Average Price", "Weight"]].sort_values(
        by="Weight", ascending=False
    ).reset_index(drop=True)

    return parsed, pd.DataFrame()

def extract_current_allocation(uploaded_file):
    file_name = str(uploaded_file.name).strip().lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.startswith("stocks_"):
        parsed, unmatched = parse_stocks_statement(file_bytes)
        return parsed, unmatched, "Stocks_ statement"

    if file_name.startswith("holdings-"):
        parsed, unmatched = parse_holdings_statement(file_bytes)
        return parsed, unmatched, "holdings- statement"

    try:
        parsed, unmatched = parse_holdings_statement(file_bytes)
        return parsed, unmatched, "holdings- style (auto-detected)"
    except Exception:
        parsed, unmatched = parse_stocks_statement(file_bytes)
        return parsed, unmatched, "Stocks_ style (fallback auto-detected)"

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

    df = yf.download(symbols, start=start_date, end=effective_end, progress=False, auto_adjust=False)["Close"]
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
    price_history = yf.download(latest_prices, period="15d", progress=False, auto_adjust=False)["Close"]

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

st.title("📊 Portfolio Rebalancer")

with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload holdings Excel file", type=["xlsx", "xls"])
    days_to_flip = st.number_input("Expected days to flip", min_value=1, value=30, step=1)
    max_dd = st.number_input("Max drawdown constraint", min_value=0.001, value=float(0.04 * 1.43), step=0.001, format="%.4f")
    drop_bottom_pct = st.slider("Drop bottom % tickers by history length", 0.0, 0.5, 0.1, 0.05)
    use_target_vol = st.checkbox("Use target volatility")
    target_volatility = st.number_input("Target volatility", min_value=0.001, value=0.10, step=0.01, format="%.3f") if use_target_vol else None
    run_btn = st.button("Run analysis", use_container_width=True)

if uploaded_file is None:
    st.info("Upload a holdings Excel file to begin.")
    st.stop()

if run_btn:
    try:
        with st.spinner("Parsing portfolio file..."):
            portfolio_df, unmatched_df, detected_format = extract_current_allocation(uploaded_file)

        st.success(f"Detected format: {detected_format}")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Parsed Allocation")
            st.dataframe(
                portfolio_df.style.format({
                    "Average Price": "₹{:,.2f}",
                    "Weight": "{:.2%}"
                }),
                use_container_width=True
            )

        with col2:
            total_invested = float((portfolio_df["Quantity"] * portfolio_df["Average Price"]).sum())
            st.metric("Stocks parsed", len(portfolio_df))
            st.metric("Total invested", f"₹{total_invested:,.2f}")

        if not unmatched_df.empty:
            st.subheader("Unmatched ISINs")
            st.dataframe(
                unmatched_df.style.format({"P&L (approx)": "₹{:,.2f}"}),
                use_container_width=True
            )

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
                drop_bottom_pct=drop_bottom_pct
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
                    axis=1
                )
                st.dataframe(current_df, use_container_width=True)

            with c2:
                st.markdown("**Optimized Portfolio**")
                optimal_df = metrics_df(optimal_stats)
                optimal_df["Value"] = optimal_df.apply(
                    lambda r: f"{r['Value']:.2%}" if r["Metric"] != "Sharpe Ratio" else f"{r['Value']:.2f}",
                    axis=1
                )
                st.dataframe(optimal_df, use_container_width=True)

        corr_matrix = log_returns.corr()
        corr_unstacked = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool)).stack()
        top_corrs = corr_unstacked.abs().sort_values(ascending=False).head(5).reset_index()
        top_corrs.columns = ["Ticker 1", "Ticker 2", "Abs Correlation"]
        st.subheader("Top Correlated Pairs")
        st.dataframe(top_corrs, use_container_width=True)

        with st.spinner("Fetching latest prices..."):
            latest_prices = log_returns.columns.tolist()
            price_map = get_latest_price_map(latest_prices)

        rebal_df, missing_prices, missing_alloc = rebalance_plan_multi(
            portfolio_df[portfolio_df["Symbol"].isin(price_map.keys())].copy(),
            optimal_weights,
            log_returns,
            price_map,
            days_to_flip
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
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")
