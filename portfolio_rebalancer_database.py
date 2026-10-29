import base64
import hmac
import importlib
import importlib.util
import io
import html
import json
import os
import re
import sqlite3
import time
import uuid
import warnings
import threading
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, time as clock_time, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pandas.tseries.offsets import BDay
from public_portfolio_trust import round_weights_to_whole_percent
from public_lumpsum_allocator import allocate_public_lumpsum
from public_basket_postgres import connect_public_basket_db, get_public_basket_database_url
from public_portfolio_publications import publish_approved_portfolio
from portfolio_risk_metrics import (
    bootstrap_tail_summary,
    empirical_expected_shortfall,
    moving_block_bootstrap_return_distribution,
)
from security_scenario_forecast import (
    forecast_executable_buys,
    normalize_saved_scenario_ranking,
)
from price_history_integrity import (
    DEFAULT_EXTREME_RETURN_THRESHOLD,
    PRICE_INTEGRITY_VERSION,
    extreme_return_dates,
    reconcile_price_frame,
)
from scalable_universe_preselection import (
    RISK_APPETITE_PAIRS,
    calculate_risk_appetite_regime,
    convert_candidate_history_to_inr,
    filter_candidates_by_market_cap,
    interpret_risk_appetite_regime,
    rank_scalable_candidates,
)
from search_surface_chart import (
    build_search_heatmap_figure,
    build_search_slice_figure,
    build_search_surface_figure,
)
from shortlist_search_checkpoint import (
    delete_checkpoints,
    ensure_checkpoint_schema,
    load_best_saved_result,
    load_latest_checkpoint,
    load_latest_checkpoint_for_research_scope,
    save_checkpoint,
)
from search_skipper import skipper_decision
import portfolio_optimizer_config as _optimizer_config
from optimization_run_timer import (
    abort_run_timer,
    estimate_run_seconds,
    finish_run_timer,
    format_elapsed,
    safe_stop_reason,
    serializable_timer,
    start_run_timer,
    update_run_stage,
)
from two_dimensional_search import (
    adaptive_anchor_values,
    best_feasible_result,
    coarse_to_fine_values,
    inclusive_values,
    local_refinement_values,
    missing_grid_pairs,
    neighbouring_grid_values,
    robust_feasible_result,
    search_convergence_summary,
)
from universal_portfolio_summary import summarize_universal_portfolio
from market_history_cache import sync_market_history
from trade_deferrals import (
    action_row_style,
    active_trade_deferrals,
    defer_trades_for_quarter,
    ensure_trade_deferral_schema,
    exclude_deferred_candidates,
    split_deferred_trades,
)
from portfolio_reserve import (
    calculate_portfolio_reserve_target,
)
from leverage_assessment import (
    LAMF_ADVERTISED_START_RATE,
    build_portfolio_leverage_assessment,
    estimate_drawdown_conditioned_debt,
)
from universal_portfolio_import import (
    apply_verified_overseas_replacement,
    ensure_import_schema,
    get_import_job,
    import_items_frame,
    latest_import_job,
    pending_import_items,
    prepare_import_job,
    record_validation_batch,
    safe_report_csv,
)
from market_cap_snapshot import (
    apply_market_cap_snapshot,
    ensure_market_cap_metadata_schema,
    prepare_market_cap_snapshot,
    refresh_market_cap_sample,
    refresh_market_cap_until_stalled,
    summarize_market_cap_coverage,
)
from universal_portfolio_cleaner import (
    apply_cleaner_exclusions,
    available_clusters,
    clear_optimizer_exclusions,
    cleaner_audit_frame,
    cleaner_preview_frame,
    cluster_members_frame,
    delete_cluster_snapshot,
    ensure_cleaner_schema,
    finalize_cleaner_job,
    filter_optimizer_candidates,
    get_cleaner_job,
    insider_sale_signal,
    latest_cleaner_job,
    optimizer_exclusions_frame,
    pending_history_items,
    pending_insider_items,
    prepare_cleaner_job,
    prepare_insider_shortlist,
    representative_cluster_sample,
    record_history_batch,
    record_insider_signal,
    score_price_history,
)

# Streamlit may retain an already-imported helper module across a hot deploy.
# Reload only when the running process still has the pre-momentum configuration;
# a fresh process takes the normal single-import path.
if not hasattr(_optimizer_config, "MOMENTUM_FILTER_CONFIG"):
    _optimizer_config = importlib.reload(_optimizer_config)

DEFAULT_HISTORY_START_DATE = _optimizer_config.DEFAULT_HISTORY_START_DATE
MAX_WEIGHT_PER_ASSET = _optimizer_config.MAX_WEIGHT_PER_ASSET
MOMENTUM_FILTER_CONFIG = _optimizer_config.MOMENTUM_FILTER_CONFIG
OPTIMIZER_CONFIG = _optimizer_config.OPTIMIZER_CONFIG
OPTIMIZER_CONFIG_VERSION = _optimizer_config.OPTIMIZER_CONFIG_VERSION
RISK_FREE_RATE_ANNUAL = _optimizer_config.RISK_FREE_RATE_ANNUAL
TRADING_DAYS_PER_YEAR = _optimizer_config.TRADING_DAYS_PER_YEAR
UNIVERSAL_PRESELECTION_CAP = 400
DEFAULT_EXACT_OPTIMIZER_ASSET_CAP = 300
DEFAULT_ADVANCED_CAP_FROM = 7700
DEFAULT_ADVANCED_CAP_THROUGH = 9400
DEFAULT_ADVANCED_CAP_STEP = 50
DEFAULT_ADVANCED_ASSETS_FROM = 600
DEFAULT_ADVANCED_ASSETS_THROUGH = 3000
DEFAULT_ADVANCED_ASSETS_STEP = 50
DEFAULT_ADVANCED_START_X = 8650
DEFAULT_ADVANCED_START_Y = 1250
MARKET_CAP_EXCLUSION_FRACTION = 0.20
MARKET_CAP_MAX_AGE_DAYS = 90
# Full-history downloads are the expensive part of a broad-universe run.  Twelve
# symbols per request made a 4,200-name shortlist issue roughly 700 sequential
# Yahoo requests for close and volume data, before any missing-symbol retries.
# Yahoo/yfinance already partitions work internally when ``threads=True``; a
# bounded 120-name request keeps the run recoverable while reducing request
# overhead by an order of magnitude.
FULL_HISTORY_BATCH_SIZE = 120
FULL_HISTORY_REQUEST_TIMEOUT_SECONDS = 20
MARKET_HISTORY_CACHE_PATH = Path(
    os.getenv(
        "PORTFOLIO_MARKET_HISTORY_CACHE",
        str(Path(__file__).resolve().parent / ".cache" / "market_history.duckdb"),
    )
)
from robust_momentum_filter import apply_robust_momentum_filter


def _percent_drop_count(total_tickers, drop_bottom_pct=0.2, min_tickers_to_keep=1):
    """Return the number to remove under a percentage rule while keeping at least one ticker when possible."""
    if total_tickers <= max(int(min_tickers_to_keep), 1):
        return 0

    count = int(np.floor(float(drop_bottom_pct) * total_tickers))
    if count == 0:
        return 1
    return min(count, max(total_tickers - max(int(min_tickers_to_keep), 1), 0))


def _select_drop_bottom_tickers_fallback(history, drop_bottom_pct=0.2, min_tickers_to_keep=1, volume_history=None):
    """Fallback selection logic used when the sibling helper file is not packaged in deployment."""
    if history is None or history.empty:
        return pd.Series(dtype=int), pd.Series(dtype=int), 0

    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame")

    valid_days = history.count().sort_values(ascending=False, kind="mergesort")
    if valid_days.empty:
        return pd.Series(dtype=int), pd.Series(dtype=int), 0

    if drop_bottom_pct <= 0:
        return valid_days, pd.Series(dtype=int), 0

    if volume_history is None:
        day_drop_count = _percent_drop_count(len(valid_days), drop_bottom_pct, min_tickers_to_keep)
        sorted_days = valid_days.sort_values(ascending=False, kind="mergesort")
        kept = sorted_days.head(len(valid_days) - day_drop_count)
        dropped = sorted_days.tail(day_drop_count)
        return kept, dropped, int(day_drop_count)

    if not isinstance(volume_history, pd.DataFrame):
        raise TypeError("volume_history must be a pandas DataFrame")

    # Price and volume downloads can recover different symbols. Restrict volume
    # evidence to the price-history universe before using its index; otherwise a
    # volume-only symbol can trigger ``not in index`` after an hour-long run.
    avg_volume = (
        volume_history.reindex(columns=valid_days.index)
        .mean()
        .sort_values(ascending=True, kind="mergesort")
    )
    liquidity_drop_count = _percent_drop_count(len(avg_volume), drop_bottom_pct, min_tickers_to_keep)
    liquidity_dropped = avg_volume.head(liquidity_drop_count)
    liquidity_kept = valid_days.loc[avg_volume.index[liquidity_drop_count:]]

    remaining = len(liquidity_kept)
    day_drop_count = _percent_drop_count(remaining, drop_bottom_pct, min_tickers_to_keep)
    ranked_history = liquidity_kept.to_frame("valid_days").join(
        volume_history.mean().rename("avg_volume"),
        how="left",
    )
    ranked_history = ranked_history.sort_values(
        ["valid_days", "avg_volume"],
        ascending=[False, False],
        kind="mergesort",
    )
    kept = ranked_history.head(remaining - day_drop_count)
    dropped = ranked_history.tail(day_drop_count)

    dropped_all = pd.concat([liquidity_dropped, dropped["valid_days"]]).sort_values(ascending=True, kind="mergesort")
    return kept["valid_days"], dropped_all, int(liquidity_drop_count + day_drop_count)


def _load_drop_logic_module():
    """Load the sibling module from the app directory, not just sys.path."""
    candidates = [
        Path(__file__).resolve().with_name("drop_logic.py"),
        Path.cwd() / "drop_logic.py",
        Path(__file__).resolve().parent.parent / "drop_logic.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("drop_logic", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    return None


drop_logic = _load_drop_logic_module()
select_drop_bottom_tickers = (
    drop_logic.select_drop_bottom_tickers if drop_logic is not None else _select_drop_bottom_tickers_fallback
)

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

APP_BUILD = "2026-10-02-run-to-plan-timer-v2"
OPTIMIZATION_TIMER_KEY = "optimization_run_to_plan_timer_v2"

# Lightweight, append-only diagnostics are kept outside the repository so a
# restart/redeploy does not erase the evidence needed to review a run.  The
# log deliberately records lifecycle/stage information only (never holdings,
# credentials, or downloaded price data).
APP_RUN_LOG_PATH = Path(
    os.getenv(
        "PORTFOLIO_APP_RUN_LOG",
        str(Path.home() / ".portfolio_rebalancer" / "app_run.log.jsonl"),
    )
)
_APP_RUN_LOG_LOCK = threading.Lock()


def _app_run_log(event, **details):
    """Append one JSONL diagnostic event; logging must never break the app."""
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        "app_build": APP_BUILD,
        "pid": os.getpid(),
    }
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[str(key)] = value
        else:
            payload[str(key)] = str(value)
    try:
        APP_RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _APP_RUN_LOG_LOCK, APP_RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        # Diagnostics are best-effort and must never affect user-facing work.
        pass


if not st.session_state.get("_app_run_log_started"):
    _app_run_log("app_session_started", streamlit_session=True)
    st.session_state["_app_run_log_started"] = True

# =========================================================
# HELPERS
# =========================================================

def trigger_download(data, file_name, mime_type):
    """Start a browser download using the supported iframe-based Streamlit API."""
    encoded_data = base64.b64encode(data).decode("utf-8")
    html_payload = f"""
    <html><body>
    <script>
    const bytes = Uint8Array.from(
        atob("{encoded_data}"),
        character => character.charCodeAt(0)
    );
    const blob = new Blob([bytes], {{ type: {json.dumps(mime_type)} }});
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = {json.dumps(file_name)};
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
    </script>
    </body></html>
    """
    st.components.v1.iframe(
        src="data:text/html;charset=utf-8," + quote(html_payload),
        height=0,
        scrolling=False,
    )

@st.cache_data(show_spinner=False)
def load_equity_mapping():
    local_path = Path(__file__).resolve().with_name("EQUITY_L.csv")
    url = "https://raw.githubusercontent.com/KPranaydeep/Finance/refs/heads/main/EQUITY_L.csv"
    # The deployed repository snapshot is deterministic and travels with the app.
    # Keep the remote file only as a fallback for unusual single-file executions.
    df = pd.read_csv(local_path if local_path.exists() else url)
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
    # NSE uses the -SM series marker for SME listings, while Yahoo history is
    # generally published under the base NSE symbol (for example, UCL.NS).
    if symbol.endswith("-SM"):
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
    if ticker.endswith(".WA"):
        return "WSE", "PLN"
    if ticker.endswith(".CN"):
        return "CSE", "CAD"
    if ticker.endswith(".ME"):
        return "MOEX", "RUB"
    if ticker.endswith(".F"):
        return "FRA", "EUR"
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

    kwargs.setdefault("timeout", FULL_HISTORY_REQUEST_TIMEOUT_SECONDS)
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
    threads=False,
    fallback_missing=True,
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
                "threads": threads,
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
            if not fallback_missing:
                failures.setdefault(ticker, "No usable history returned by the bulk request.")
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

    market_qualifier = None
    if ":" in raw:
        qualifier, qualified_ticker = raw.split(":", 1)
        qualifier_aliases = {
            "US": "US",
            "NASDAQ": "NASDAQ",
            "NMS": "NASDAQ",
            "NGM": "NASDAQ",
            "NYSE": "NYSE",
            "NYQ": "NYSE",
            "AMEX": "AMEX",
            "NYSEARCA": "NYSEARCA",
        }
        market_qualifier = qualifier_aliases.get(qualifier.strip())
        if market_qualifier and qualified_ticker.strip():
            raw = qualified_ticker.strip()
        else:
            market_qualifier = None

    nse_company_lookup = nse_company_lookup or get_nse_company_lookup()
    base = normalize_portfolio_symbol(raw)

    if base.endswith("-BE"):
        base = base[:-3].rstrip("-").strip()

    explicit = bool(market_qualifier) or (
        raw.endswith(".NS")
        or raw.endswith(".BO")
        or ("." in raw and not raw.endswith(".NS") and not raw.endswith(".BO"))
        or raw.endswith("=X")
        or raw.startswith("^")
    )

    if raw.endswith("-SM.NS"):
        # Yahoo's -SM alias can resolve but returns only a stale placeholder row.
        # Prefer the regular NSE ticker, retaining the alias as a last resort.
        candidates = [f"{base}.NS", raw]
    elif explicit:
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
        if market_qualifier:
            display_symbol = f"{market_qualifier}:{base}"
        elif ticker.endswith(".NS"):
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
    market_cap_millions REAL,
    market_cap_source TEXT,
    market_cap_as_of TEXT,
    market_cap_imported_at TEXT,
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
    ensure_checkpoint_schema(conn)
    ensure_import_schema(conn)
    ensure_cleaner_schema(conn)
    ensure_trade_deferral_schema(conn)
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
            "market_cap_millions": "NULL",
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
                 quantity, average_price, market_cap_millions, added_at, updated_at)
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
        "market_cap_millions": "ALTER TABLE master_holdings ADD COLUMN market_cap_millions REAL",
        "market_cap_source": "ALTER TABLE master_holdings ADD COLUMN market_cap_source TEXT",
        "market_cap_as_of": "ALTER TABLE master_holdings ADD COLUMN market_cap_as_of TEXT",
        "market_cap_imported_at": "ALTER TABLE master_holdings ADD COLUMN market_cap_imported_at TEXT",
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
                market_cap_millions AS "Market Cap Millions",
                market_cap_source AS "Market Cap Source",
                market_cap_as_of AS "Market Cap As Of",
                market_cap_imported_at AS "Market Cap Imported At",
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
            looks_like_sme_alias = ticker.endswith("-SM.NS")

            if (
                not metadata_missing
                and not looks_like_bad_legacy_nse_guess
                and not looks_like_sme_alias
            ):
                continue

            instrument = resolve_yahoo_instrument(symbol, lookup)
            if instrument is None:
                continue

            new_ticker = str(instrument["yahoo_ticker"]).upper()
            new_exchange = str(instrument["exchange"] or "").strip()
            new_currency = _normalize_currency_code(instrument["currency"])
            old_name = str(row["stock_name"] or "").strip()
            new_name = str(instrument["stock_name"] or symbol).strip()
            if (
                old_name
                and old_name.upper() != symbol.upper()
                and not looks_like_bad_legacy_nse_guess
                and not looks_like_sme_alias
            ):
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
    "trading symbol": "Ticker",
    "trading_symbol": "Ticker",
    "isin": "ISIN",
    "symbol isin": "ISIN",
    "symbol_isin": "ISIN",
    "quantity": "Quantity",
    "quantity available": "Quantity",
    "qty": "Quantity",
    "average buy price": "Average Buy Price",
    "average price": "Average Buy Price",
    "average_price": "Average Buy Price",
    "avg. price": "Average Buy Price",
    "avg. cost": "Average Buy Price",
    "avg cost": "Average Buy Price",
    "avg buy price ($)": "Average Buy Price",
    "average buy price ($)": "Average Buy Price",
    "ticker": "Ticker",
    "stock symbol": "Ticker",
    "exchange": "Exchange",
    "buy value": "Buy Value",
    "closing price": "Closing Price",
    "previous closing price": "Closing Price",
    "closing value": "Closing Value",
    "unrealised p&l": "Unrealised P&L",
    "unrealized p&l": "Unrealised P&L",
}


def _normalise_company_name(value):
    """Return a comparison key that ignores punctuation and legal suffix noise."""
    words = re.findall(r"[a-z0-9]+", str(value or "").lower())
    suffixes = {
        "co", "company", "corp", "corporation", "inc", "incorporated",
        "ltd", "limited", "plc", "group", "holdings", "holding",
    }
    return " ".join(word for word in words if word not in suffixes)


@st.cache_data(show_spinner=False, ttl=86400)
def resolve_us_instrument_by_name(stock_name):
    """Resolve a US company name through Yahoo without silently guessing.

    Tickertape's US CSV omits exchange tickers and ISINs.  A candidate is accepted
    only when Yahoo identifies it as a USD-listed equity/ETF and its normalized
    name closely matches the report.  Ambiguous or weak matches remain unresolved.
    """
    import yfinance as yf

    query = str(stock_name or "").strip()
    wanted = _normalise_company_name(query)
    if not wanted:
        return None

    quotes = []
    for search_query in dict.fromkeys([query, wanted]):
        try:
            quotes.extend(yf.Search(search_query, max_results=10, news_count=0).quotes or [])
        except Exception:
            continue

    candidates = []
    allowed_types = {"EQUITY", "ETF"}
    us_exchanges = {"ASE", "BTS", "NCM", "NGM", "NMS", "NYQ", "NYS", "PCX"}
    seen_symbols = set()
    for quote_row in quotes:
        symbol = str(quote_row.get("symbol") or "").strip().upper()
        quote_type = str(quote_row.get("quoteType") or "").strip().upper()
        exchange = str(quote_row.get("exchange") or "").strip().upper()
        candidate_name = quote_row.get("longname") or quote_row.get("shortname") or ""
        candidate_key = _normalise_company_name(candidate_name)
        if (
            not symbol or symbol in seen_symbols or quote_type not in allowed_types
            or exchange not in us_exchanges or not candidate_key
        ):
            continue
        seen_symbols.add(symbol)
        similarity = SequenceMatcher(None, wanted, candidate_key).ratio()
        wanted_tokens = set(wanted.split())
        candidate_tokens = set(candidate_key.split())
        token_score = (
            len(wanted_tokens & candidate_tokens) / len(wanted_tokens | candidate_tokens)
            if wanted_tokens and candidate_tokens else 0.0
        )
        candidates.append((max(similarity, token_score), symbol))

    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] < 0.78:
        return None
    if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] - 0.04:
        return None

    return resolve_yahoo_instrument(candidates[0][1], {})


def _read_tickertape_us_holdings_csv(uploaded_file):
    """Parse a Tickertape US holdings CSV into the broker-import schema."""
    if uploaded_file is None:
        raise ValueError("Choose a Tickertape US holdings CSV first.")
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected broker holdings file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    df.columns = [str(col).strip() for col in df.columns]
    renamed = {
        col: BROKER_HOLDINGS_COLUMN_ALIASES[col.lower()]
        for col in df.columns
        if col.lower() in BROKER_HOLDINGS_COLUMN_ALIASES
    }
    df = df.rename(columns=renamed)
    required = ["Stock Name", "Quantity", "Average Buy Price"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "The Tickertape US CSV is missing required columns: " + ", ".join(missing)
        )

    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    df["Stock Name"] = df["Stock Name"].fillna("").astype(str).str.strip()
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Buy Price"] = pd.to_numeric(df["Average Buy Price"], errors="coerce")
    df = df[
        df["Stock Name"].ne("")
        & df["Quantity"].notna() & (df["Quantity"] > 0)
        & df["Average Buy Price"].notna() & (df["Average Buy Price"] > 0)
    ].copy()
    if df.empty:
        raise ValueError("No valid positive-quantity holdings were found in the Tickertape US CSV.")
    df["Currency"] = "USD"
    return df[["Ticker", "Stock Name", "Quantity", "Average Buy Price", "Currency"]]


def _detect_broker_holdings_header_row(raw_df, max_scan_rows=25):
    """Locate an Indian broker header by scanning for ISIN + Quantity aliases.

    This tolerates broker exports that don't always place headers on row 11.
    """
    for row_idx in range(min(max_scan_rows, len(raw_df))):
        canonical_columns = {
            BROKER_HOLDINGS_COLUMN_ALIASES.get(str(value).strip().lower())
            for value in raw_df.iloc[row_idx].tolist()
            if pd.notna(value)
        }
        if {"ISIN", "Quantity"}.issubset(canonical_columns):
            return row_idx
    return None


def _read_holdings_preview(raw, file_name):
    """Read the first report rows without assigning a header."""
    if str(file_name or "").lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw), header=None, nrows=25)
        except Exception as exc:
            raise ValueError(f"Could not read the CSV file: {exc}") from exc

    try:
        return pd.read_excel(io.BytesIO(raw), header=None, nrows=25)
    except ImportError as exc:
        raise ValueError(
            "Reading Excel files requires the 'openpyxl' package. Install it with "
            "`pip install openpyxl` and retry."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read the Excel file: {exc}") from exc


def _detect_holdings_report_type(uploaded_file):
    """Classify a holdings report from its schema, never from its extension alone."""
    if uploaded_file is None:
        raise ValueError("Choose a broker holdings file first.")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected broker holdings file is empty.")

    file_name = str(getattr(uploaded_file, "name", "") or "").strip().lower()
    preview_df = _read_holdings_preview(raw, file_name)
    ambiguous_header_found = False

    for row_idx in range(min(25, len(preview_df))):
        raw_columns = {
            str(value).strip().lower()
            for value in preview_df.iloc[row_idx].tolist()
            if pd.notna(value) and str(value).strip()
        }
        canonical_columns = {
            BROKER_HOLDINGS_COLUMN_ALIASES.get(column)
            for column in raw_columns
        }

        if {"ISIN", "Quantity", "Average Buy Price"}.issubset(canonical_columns):
            return "INDIAN"

        us_required = {"Stock Name", "Quantity", "Average Buy Price"}
        if us_required.issubset(canonical_columns):
            has_us_currency_marker = any(
                "$" in column or "usd" in column for column in raw_columns
            )
            if has_us_currency_marker:
                return "TICKERTAPE_US"
            ambiguous_header_found = True

    if ambiguous_header_found:
        raise ValueError(
            "The CSV has generic holdings columns but no ISIN or explicit USD fields, "
            "so its market cannot be identified safely. Export the Groww holdings "
            "report with ISIN, or the Tickertape US report with USD-labelled columns."
        )
    raise ValueError(
        "The holdings report format was not recognized. Groww/Indian reports must "
        "include ISIN, Quantity, and Average Price; Tickertape US reports must include "
        "Stock Name, Quantity, and a USD-labelled average buy price."
    )


def _previous_run_estimate(owner):
    previous = load_latest_analysis(owner) or {}
    timing = previous.get("run_to_rebalancing_plan_timing") or {}
    preselection = (previous.get("settings") or {}).get("universe_preselection") or {}
    return timing.get("elapsed_seconds"), preselection.get("eligible")


def _start_optimization_plan_timer(owner):
    previous_elapsed, _ = _previous_run_estimate(owner)
    estimate = estimate_run_seconds(previous_elapsed, None, None)
    st.session_state[OPTIMIZATION_TIMER_KEY] = start_run_timer(
        estimated_total_seconds=estimate
    )


def _set_optimization_plan_stage(stage, progress, estimated_total_seconds=None):
    timer = st.session_state.get(OPTIMIZATION_TIMER_KEY) or {}
    updated = update_run_stage(
        timer,
        stage,
        progress,
        estimated_total_seconds=estimated_total_seconds,
    )
    st.session_state[OPTIMIZATION_TIMER_KEY] = updated
    return updated


def _display_optimization_stage(placeholder, stage, progress, estimated_total_seconds=None):
    previous_stage = (st.session_state.get(OPTIMIZATION_TIMER_KEY) or {}).get("stage")
    timer = _set_optimization_plan_stage(
        stage, progress, estimated_total_seconds=estimated_total_seconds
    )
    if previous_stage != stage:
        _app_run_log("optimization_stage", stage=str(stage), progress=float(progress))
    summary = _timer_summary(timer)
    if summary:
        placeholder.info(summary[1])
    return timer


def _finish_optimization_plan_timer():
    timer = st.session_state.get(OPTIMIZATION_TIMER_KEY) or {}
    finished = finish_run_timer(timer, "rebalancing plan generated")
    st.session_state[OPTIMIZATION_TIMER_KEY] = finished
    return finished


def _abort_optimization_plan_timer(reason):
    timer = st.session_state.get(OPTIMIZATION_TIMER_KEY) or {}
    stopped = abort_run_timer(timer, reason)
    st.session_state[OPTIMIZATION_TIMER_KEY] = stopped
    return stopped


def _timer_summary(timer):
    if not timer:
        return None
    status = timer.get("status")
    elapsed = format_elapsed(timer.get("elapsed_seconds"))
    if status == "running":
        stage = str(timer.get("stage") or "Running")
        progress = float(timer.get("progress") or 0.0)
        eta = timer.get("eta_seconds")
        eta_text = format_elapsed(eta) if eta is not None else "estimating"
        return (
            "running",
            f"**{stage}** · {progress:.0%} · estimated time remaining **{eta_text}**. "
            "The estimate adapts after each completed stage.",
        )
    if status == "finished":
        return "finished", f"Run-to-plan time: **{elapsed}** · Rebalancing plan generated"
    if status == "stopped_without_plan":
        stage = str(timer.get("stage") or "Unknown stage")
        reason = safe_stop_reason(timer.get("stop_reason"))
        return (
            "stopped",
            f"Run stopped during **{stage}** after **{elapsed}**. Reason: {reason}",
        )
    return None


def _read_indian_broker_holdings(uploaded_file):
    """Parse an Indian broker CSV/Excel with Stock Name/ISIN/Quantity/
    Average Buy Price/Buy Value/Closing Price/Closing Value/Unrealised P&L columns.

    The header row is auto-detected so Groww exports work as either CSV or Excel.
    """
    if uploaded_file is None:
        raise ValueError("Choose an Indian broker holdings file first.")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected broker holdings file is empty.")

    file_name = str(getattr(uploaded_file, "name", "") or "").strip().lower()
    preview_df = _read_holdings_preview(raw, file_name)
    header_row = _detect_broker_holdings_header_row(preview_df)
    if header_row is None:
        raise ValueError(
            "Could not locate an Indian holdings header containing ISIN and Quantity."
        )

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw), header=header_row)
        else:
            df = pd.read_excel(io.BytesIO(raw), header=header_row)
    except Exception as exc:
        raise ValueError(
            f"Could not read the holdings file with header row {header_row + 1}: {exc}"
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
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    if "Exchange" not in df.columns:
        df["Exchange"] = ""

    df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()
    df["Stock Name"] = df["Stock Name"].fillna("").astype(str).str.strip()
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip().str.upper()
    df["Exchange"] = df["Exchange"].fillna("").astype(str).str.strip().str.upper()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Buy Price"] = pd.to_numeric(df["Average Buy Price"], errors="coerce")

    df = df[df["ISIN"].str.match(r"^IN[A-Z0-9]{10}$", na=False)].copy()
    df = df[df["Quantity"].notna() & (df["Quantity"] > 0)]
    df = df[df["Average Buy Price"].notna() & (df["Average Buy Price"] > 0)]

    if df.empty:
        raise ValueError(
            "No valid holdings rows (with ISIN, Quantity, Average Buy Price) were found."
        )

    def _first_nonempty(values):
        return next((value for value in values if value), "")

    df["Total Buy Cost"] = df["Quantity"] * df["Average Buy Price"]
    df = df.groupby("ISIN", as_index=False).agg(**{
        "Stock Name": ("Stock Name", "first"),
        "Ticker": ("Ticker", _first_nonempty),
        "Exchange": ("Exchange", _first_nonempty),
        "Quantity": ("Quantity", "sum"),
        "Total Buy Cost": ("Total Buy Cost", "sum"),
    })
    df["Average Buy Price"] = df["Total Buy Cost"] / df["Quantity"]
    return df[
        ["ISIN", "Ticker", "Exchange", "Stock Name", "Quantity", "Average Buy Price"]
    ]


def _read_broker_holdings_excel(uploaded_file):
    """Backward-compatible wrapper for Indian broker report parsing."""
    return _read_indian_broker_holdings(uploaded_file)


def _indian_report_ticker_candidates(symbol, exchange_hint, nse_company_lookup):
    """Return Indian-only Yahoo candidates for a symbol from an ISIN report."""
    raw_symbol = str(symbol or "").strip().upper()
    if not raw_symbol:
        return []

    base_symbol = normalize_portfolio_symbol(raw_symbol)
    exchange = str(exchange_hint or "").strip().upper()
    if exchange in {"BSE", "BOM", "BO"}:
        return [f"{base_symbol}.BO"]
    if exchange in {"NSE", "NS"} or base_symbol in (nse_company_lookup or {}):
        return [f"{base_symbol}.NS"]
    return [f"{base_symbol}.NS", f"{base_symbol}.BO"]


def _instrument_from_official_nse_symbol(symbol, company_name=""):
    """Build identity from the bundled official NSE mapping without a price probe."""
    base_symbol = normalize_portfolio_symbol(symbol)
    if not base_symbol:
        return None

    ticker = f"{base_symbol}.NS"
    metadata = get_yahoo_metadata(ticker)
    return {
        "symbol": base_symbol,
        "yahoo_ticker": ticker,
        "stock_name": str(company_name or metadata["stock_name"] or base_symbol).strip(),
        "exchange": str(metadata["exchange"] or "NSE").strip(),
        "currency": _normalize_currency_code(metadata["currency"] or "INR"),
    }


def import_broker_holdings_excel(uploaded_file, owner, mode="merge"):
    """Resolve schema-detected Indian or Tickertape US holdings and upsert them.

    Returns (imported_row_count, unresolved_isins).
    """
    report_type = _detect_holdings_report_type(uploaded_file)
    is_tickertape_us = report_type == "TICKERTAPE_US"
    cleaned = (
        _read_tickertape_us_holdings_csv(uploaded_file)
        if is_tickertape_us else _read_indian_broker_holdings(uploaded_file)
    )
    normalized_mode = str(mode or "merge").strip().lower()
    if normalized_mode not in {"replace", "merge"}:
        raise ValueError("Import mode must be either 'replace' or 'merge'.")

    isin_to_symbol = {}
    if not is_tickertape_us:
        equity_map = load_equity_mapping().copy()
        equity_map["ISIN"] = equity_map["ISIN"].astype(str).str.strip().str.upper()
        isin_to_symbol = dict(
            zip(equity_map["ISIN"], equity_map["Symbol"].astype(str).str.strip().str.upper())
        )
    nse_lookup = get_nse_company_lookup()

    resolved_rows = []
    unresolved_isins = []

    for _, row in cleaned.iterrows():
        if is_tickertape_us:
            supplied_ticker = str(row.get("Ticker") or "").strip().upper()
            instrument = (
                resolve_yahoo_instrument(supplied_ticker, {})
                if supplied_ticker else resolve_us_instrument_by_name(row["Stock Name"])
            )
            unresolved_label = str(row["Stock Name"])
        else:
            isin = row["ISIN"]
            mapped_nse_symbol = str(isin_to_symbol.get(isin) or "").strip().upper()
            symbol_guess = str(mapped_nse_symbol or row.get("Ticker") or "").strip().upper()
            exchange_hint = str(row.get("Exchange") or "").strip().upper()
            # An ISIN-bearing report is Indian even when its exchange column is
            # absent. Probe Indian venues only; never let a same-name US ticker win.
            candidates = _indian_report_ticker_candidates(
                symbol_guess, exchange_hint, nse_lookup
            )

            instrument = None
            for candidate in candidates:
                instrument = resolve_yahoo_instrument(candidate, nse_lookup)
                if instrument is not None:
                    break
            if instrument is None and mapped_nse_symbol:
                # Price-provider availability must not erase an identity already
                # established by the official NSE symbol/ISIN mapping.
                instrument = _instrument_from_official_nse_symbol(
                    mapped_nse_symbol, nse_lookup.get(mapped_nse_symbol, row["Stock Name"])
                )
            unresolved_label = f"{isin} ({row['Stock Name']})"
        if instrument is None:
            unresolved_isins.append(unresolved_label)
            continue

        resolved_rows.append({
            "Symbol": instrument["symbol"],
            "Stock Name": row["Stock Name"] or instrument["stock_name"],
            "Yahoo Ticker": instrument["yahoo_ticker"],
            "Exchange": instrument["exchange"],
            "Currency": _normalize_currency_code(
                row.get("Currency", instrument["currency"])
            ),
            "Quantity": float(row["Quantity"]),
            "Average Price": float(row["Average Buy Price"]),
        })

    if not resolved_rows:
        raise ValueError(
            "None of the holdings in the broker file could be resolved. Unresolved: "
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
    if not np.isfinite(numeric):
        return "N/A"
    if metric_name == "Sharpe Ratio":
        return f"{numeric:.2f}"
    return f"{numeric:.2%}"


def _render_market_read_panel(market_read, *, title):
    """Render the market-regime read as compact context, not a trade instruction."""
    with st.container(border=True):
        st.markdown(f"**{title} · {market_read['headline']}**")
        if market_read.get("as_of"):
            st.caption(f"Benchmark data through {market_read['as_of']}")
        components = market_read.get("components", [])
        if components:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Signal": f"{item['pair']} · {item['label']}",
                            "Observed": item["reading"],
                            "Possible context (not proven cause)": item[
                                "plausible_causes"
                            ],
                        }
                        for item in components
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption(market_read["evidence"])
        st.write(f"**What this says:** {market_read['inference']}")
        st.caption(
            "Use as background when weighing buys, sells, or cash—not as a timing signal. "
            "These ratios do not measure market leverage, put/call open interest, your "
            "cash or margin, or the price/coverage of a hedge."
        )


def _render_saved_downside_tradeoff(current_stats, optimal_stats):
    """Show the optimizer's existing tail-loss context without implying insurance."""
    metric = "Block-Bootstrap ES 95% (20 Sessions)"
    current_es = (current_stats or {}).get(metric)
    proposed_es = (optimal_stats or {}).get(metric)
    current_return = (current_stats or {}).get("Annual Return")
    proposed_return = (optimal_stats or {}).get("Annual Return")
    if current_es is None and proposed_es is None:
        return

    def pct(value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "not available"
        return f"{numeric * 100:.2f}%" if np.isfinite(numeric) else "not available"

    with st.container(border=True):
        st.markdown("**Return / tail-loss check · current → proposed**")
        st.caption(
            f"Expected annual return: {pct(current_return)} → {pct(proposed_return)}  ·  "
            f"20-session 95% expected shortfall: {pct(current_es)} → {pct(proposed_es)}"
        )
        st.caption(
            "This bootstrap estimate is not a drawdown limit or hedge. Cash can soften "
            "losses on the cash slice but may miss market gains; option protection has a "
            "premium and may mismatch individual holdings. No hedge payoff or cost is "
            "modeled in this plan."
        )



def _render_saved_reserve_study(payload):
    """Show an automatic reserve percentage derived from optimized portfolio stats."""
    optimal_stats = payload.get("optimal_stats") or {}
    base_value_raw = payload.get("current_portfolio_value")
    base_source = "saved current market value"
    try:
        base_value = float(base_value_raw)
    except (TypeError, ValueError):
        base_value = float(payload.get("total_invested") or 0.0)
        base_source = "cost basis fallback (market value was not saved)"
    if not np.isfinite(base_value) or base_value <= 0:
        st.info("A positive portfolio value is needed to calculate a currency reference.")
        return

    try:
        reserve = calculate_portfolio_reserve_target(
            optimal_stats,
            invested_base=base_value,
        )
    except ValueError as exc:
        st.info(f"Portfolio reserve reference unavailable: {exc}.")
        return

    reserve_pct = reserve["reserve_fraction"]
    reserve_amount = reserve["reserve_amount"]
    with st.container(border=True):
        st.markdown(
            f"**Portfolio-derived reserve reference:** hold **{reserve_pct:.2%}** "
            f"(₹{reserve_amount:,.2f}) as additional reserve while keeping the "
            f"optimized portfolio 100% invested; it uses the {reserve['method']}’s "
            "central 20-session outcome plus one downside-tail standard deviation, "
            "as a deployment buffer—not insurance."
        )
        return_history = payload.get("leverage_return_history") or {}
        saved_returns = return_history.get("portfolio_log_returns") or []
        if saved_returns:
            try:
                leverage = estimate_drawdown_conditioned_debt(
                    saved_returns,
                    borrowing_rate=LAMF_ADVERTISED_START_RATE,
                )
                debt_pct = float(
                    leverage["recommended_debt_ratio_of_portfolio_base"]
                )
                debt_amount = base_value * debt_pct
                saved_dates = return_history.get("dates") or []
                leverage_as_of = (
                    str(saved_dates[-1])[:10]
                    if saved_dates
                    else "the latest saved portfolio data"
                )
                st.markdown(
                    f"**Leverage reference:** at 9.99% p.a., the drawdown-conditioned "
                    f"model indicates {debt_pct:.1%} additional debt/base "
                    f"(about ₹{debt_amount:,.0f}), with the portfolio "
                    f"{abs(float(leverage['current_drawdown'])):.2%} below its trailing "
                    f"252-session high as of {leverage_as_of}; this is a historical "
                    "scenario, not a safe limit or lender-approved credit amount."
                )
            except (TypeError, ValueError, KeyError, IndexError):
                st.markdown(
                    "Leverage reference unavailable: the saved portfolio history is "
                    "not long enough for a 252-session drawdown assessment."
                )
        else:
            st.markdown(
                "**Leverage reference:** unavailable because this saved analysis has no "
                "daily portfolio-return history; rerun optimization to calculate it."
            )
    with st.expander("How this reserve percentage is derived", expanded=False):
        st.caption(
            f"Sizing base: ₹{base_value:,.2f} ({base_source}). The reserve rule is "
            "x = max(0, −median 20-session portfolio return) + one standard deviation "
            "of losses in the worst 5% of simulated outcomes. This combines the typical "
            "central path with a pragmatic tail-dispersion buffer; it is not an optimizer "
            "or a guaranteed/optimal reserve amount."
        )
        st.caption(
            f"Central 20-session return: {reserve['median_return']:+.2%} · "
            f"95% downside-tail SD: {reserve['tail_loss_sd']:.2%} · "
            f"reserve x: {reserve_pct:.2%}. {reserve['method']} is based on the "
            "optimized portfolio’s saved history; it cannot bound future gaps or losses. "
            "A short-dated sovereign bill held to maturity in the likely deployment "
            "currency can reduce price sensitivity, but still has inflation, reinvestment, "
            "liquidity, tax, and possibly FX risk."
        )


def _render_saved_leverage_check(payload, *, owner, widget_scope):
    """Show portfolio-specific financing and downside scenarios without claiming safety."""
    optimal_stats = payload.get("optimal_stats") or {}
    tail_loss = optimal_stats.get("Block-Bootstrap ES 95% (20 Sessions)")
    annual_return = optimal_stats.get("Annual Return")
    try:
        base_value = float(
            payload.get("current_portfolio_value") or payload.get("total_invested") or 0.0
        )
        tail_loss = float(tail_loss)
        annual_return = float(annual_return)
    except (TypeError, ValueError):
        st.info("The saved portfolio is missing usable return or portfolio-value data for a leverage check.")
        return
    if (
        not np.isfinite(base_value)
        or base_value <= 0
        or not np.isfinite(tail_loss)
        or tail_loss < 0
        or tail_loss >= 1
        or not np.isfinite(annual_return)
    ):
        st.info("The saved portfolio statistics are outside the usable range for a leverage check.")
        return

    with st.expander("Portfolio-derived leverage assessment", expanded=False):
        st.caption(
            "This compares financing-adjusted historical scenarios for the saved optimal mix. "
            "It is a research diagnostic—not a safe borrowing limit or lender approval."
        )
        return_history = payload.get("leverage_return_history") or {}
        daily_log_returns = return_history.get("portfolio_log_returns") or []
        leverage_analysis = payload.get("leverage_assessment") or {}
        if st.button(
            "Run leverage analysis now",
            icon=":material/analytics:",
            disabled=not bool(daily_log_returns),
            key=f"run_leverage_analysis_{widget_scope}",
            help=(
                "Uses the latest saved optimal portfolio return history. Run optimization once "
                "to enable this for older saved analyses."
            ),
        ):
            try:
                with st.spinner("Recalculating leverage scenarios from the saved portfolio…"):
                    leverage_analysis = build_portfolio_leverage_assessment(
                        daily_log_returns,
                        invested_base=base_value,
                    )
                leverage_analysis["computed_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                payload["leverage_assessment"] = leverage_analysis
                save_latest_analysis(payload, owner)
                st.success("Leverage analysis updated and saved with this portfolio run.")
            except (TypeError, ValueError) as exc:
                st.error(f"Could not calculate portfolio leverage scenarios: {exc}")

        saved_scenarios = leverage_analysis.get("scenarios") or []
        if saved_scenarios:
            funding_rates = [float(item["rate"]) for item in saved_scenarios]
            base_scenario = next(
                (item for item in saved_scenarios if abs(float(item["rate"]) - LAMF_ADVERTISED_START_RATE) < 1e-9),
                saved_scenarios[0],
            )
            assessment = {
                "observations": base_scenario.get("observations", 0),
                "fold_count": base_scenario.get("growth_optimum", {}).get("segment_count", 3),
                "rows": base_scenario.get("frontier", []),
                "selection_bias_warning": base_scenario.get("selection_bias_warning", ""),
            }
            rate_rows = {
                float(scenario["rate"]): {
                    row["leverage_multiple"]: row
                    for row in scenario.get("frontier", [])
                }
                for scenario in saved_scenarios
            }
            if len(funding_rates) == 1:
                funding_rates.extend([funding_rates[0]] * 2)
            if len(funding_rates) == 2:
                funding_rates.append(funding_rates[-1])
            if len(funding_rates) > 3:
                funding_rates = funding_rates[:3]
            rows = []
            for row in assessment["rows"]:
                if not row.get("solvent_in_history"):
                    rows.append(
                        {
                            "Gross exposure": f"{row['leverage_multiple']:.2f}×",
                            "Additional debt / base": f"{row['debt_ratio']:.0%}",
                            "Historical result": "Would have exhausted equity in at least one session",
                        }
                    )
                    continue
                rows.append(
                    {
                        "Gross exposure": f"{row['leverage_multiple']:.2f}×",
                        "Additional debt / base": f"{row['debt_ratio']:.0%}",
                        f"Net annualized return* ({funding_rates[0]:.2%})": f"{row['annualized_net_return']:.2%}",
                        f"Return* ({funding_rates[1]:.2%} rate stress)": f"{rate_rows[funding_rates[1]][row['leverage_multiple']]['annualized_net_return']:.2%}",
                        f"Return* ({funding_rates[2]:.2%} rate stress)": f"{rate_rows[funding_rates[2]][row['leverage_multiple']]['annualized_net_return']:.2%}",
                        "Positive segments (base rate)": f"{row['positive_folds']}/{assessment['fold_count']}",
                        "Segments beating 1.00× (base rate)": f"{row['folds_beating_unlevered']}/{assessment['fold_count']}",
                        "20-session 95% ES loss": f"{row['tail_expected_shortfall']:.2%}",
                        "Historical max drawdown*": f"{row['historical_max_drawdown']:.2%}",
                        "Interest / ₹1L base / year": f"₹{row['annual_interest_per_100k']:,.0f}",
                    }
                )
            with st.container(border=True):
                st.markdown(
                    f"**Portfolio-only scenarios:** tested {assessment['observations']:,} "
                    "saved daily return observations at the advertised LAMF starting-rate "
                    f"assumption of **{LAMF_ADVERTISED_START_RATE:.2%} p.a.** Borrowing costs are charged daily; the "
                    "portfolio return is scaled to each exposure level."
                )
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption(
                    "*These are full-history annualized scenarios—not the borrowing recommendation. "
                    "Large annualized returns extrapolate the saved mix's in-sample average and do not "
                    "show outcomes specifically after drawdowns like the latest one. "
                    f"{assessment['selection_bias_warning']} "
                    "The 20-session expected shortfall is estimated from overlapping historical windows; "
                    "it is not a maximum-loss bound. Exposure is held constant by a simplified "
                    "daily-reset calculation. The +2/+4 percentage-point funding-rate changes are "
                    "stress assumptions, not Tickertape quotes; fees, changing collateral values, "
                    "repayment terms and actual liquidation rules are not modeled."
                )
            if daily_log_returns:
                try:
                    drawdown_recommendations = []
                    for scenario in saved_scenarios:
                        drawdown = estimate_drawdown_conditioned_debt(
                            daily_log_returns,
                            borrowing_rate=float(scenario["rate"]),
                        )
                        debt_ratio = float(
                            drawdown["recommended_debt_ratio_of_portfolio_base"]
                        )
                        drawdown["estimated_debt_amount"] = base_value * debt_ratio
                        drawdown_recommendations.append(drawdown)
                    current_dd = float(drawdown_recommendations[0]["current_drawdown"])
                    dates = return_history.get("dates") or []
                    as_of = str(dates[-1])[:10] if dates else "latest saved return"
                    base_drawdown = min(
                        drawdown_recommendations,
                        key=lambda item: abs(
                            float(item["borrowing_rate"]) - LAMF_ADVERTISED_START_RATE
                        ),
                    )
                    recommended_debt = float(
                        base_drawdown["recommended_debt_ratio_of_portfolio_base"]
                    )
                    recommended_amount = float(
                        base_drawdown["estimated_debt_amount"]
                    )
                    with st.container(border=True):
                        st.markdown("**Drawdown-conditioned borrowing scenario**")
                        st.markdown(
                            f"As of **{as_of}**, the saved optimal portfolio is "
                            f"**{abs(current_dd):.2%} below its trailing 252-session high**. "
                            f"Among historical periods with similar portfolio drawdowns, "
                            f"the 20-session lower-tail model indicates **{recommended_debt:.1%} "
                            f"additional debt/base (about ₹{recommended_amount:,.0f}** on this "
                            f"saved portfolio value) at **{LAMF_ADVERTISED_START_RATE:.2%} p.a.**"
                        )
                        dd_rows = [
                            {
                                "Funding rate": f"{float(item['borrowing_rate']):.2%}",
                                "Debt / base": f"{float(item['recommended_debt_ratio_of_portfolio_base']):.1%}",
                                "Illustrative debt": f"₹{float(item['estimated_debt_amount']):,.0f}",
                                "Comparable windows": int(item["analog_count"]),
                                "20-session 5% tail log growth": f"{float(item['lower_tail_log_growth']):.2%}",
                            }
                            for item in drawdown_recommendations
                        ]
                        st.dataframe(pd.DataFrame(dd_rows), hide_index=True)
                        st.caption(
                            f"Uses the latest return date saved by optimization ({as_of}), not a live broker "
                            "mark. It matches historical windows by drawdown depth and chooses the tested "
                            "debt ratio with the best empirical 5% lower-tail 20-session log growth; it "
                            "returns 0% unless that beats the unlevered case. Historical analogs overlap, "
                            "and the saved mix was optimized on its full history. This is a model scenario, "
                            "not a safe/allowed limit or Tickertape credit approval; actual LAMF capacity "
                            "depends on eligible mutual-fund collateral and live lender terms."
                        )
                        st.caption(
                            "Why this may differ from the large 2× annualized return above: this decision "
                            "compares financing-adjusted 5th-percentile outcomes over the next 20 sessions "
                            "after historical periods at a similar one-year drawdown. It does not maximize "
                            "the full-history average return."
                        )
                        candidate_rows = base_drawdown.get("candidate_rows") or []
                        if candidate_rows:
                            candidate_frame = pd.DataFrame(
                                [
                                    {
                                        "Additional debt / base": f"{float(row['debt_ratio']):.0%}",
                                        "Gross exposure": f"{float(row['gross_exposure']):.2f}×",
                                        "Matched 20-session 5% tail return": (
                                            f"{float(row['lower_tail_20_session_return']):+.2%}"
                                            if np.isfinite(float(row["lower_tail_20_session_return"]))
                                            else "Equity exhausted in sampled paths"
                                        ),
                                        "Change vs no debt": f"{float(row['change_vs_unlevered']):+.4%} log growth",
                                    }
                                    for row in candidate_rows
                                    if int(round(float(row["debt_ratio"]) * 100))
                                    in {0, 10, 25, 50, 75, 100}
                                ]
                            )
                            st.dataframe(candidate_frame, hide_index=True)
                            if recommended_debt <= 0:
                                baseline_candidate = candidate_rows[0]
                                best_candidate = max(
                                    candidate_rows,
                                    key=lambda row: float(row["lower_tail_log_growth"]),
                                )
                                st.info(
                                    f"The model shows 0% additional debt because the best tested debt "
                                    f"level ({float(best_candidate['debt_ratio']):.1%}) did not improve "
                                    f"matched 5% tail log-growth over no debt. The sampled 20-session "
                                    f"tail returns were {float(baseline_candidate['lower_tail_20_session_return']):+.2%} "
                                    f"without debt and {float(best_candidate['lower_tail_20_session_return']):+.2%} "
                                    "at that debt level. This is a conservative model comparison, not a "
                                    "claim that borrowing is categorically impossible."
                                )
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    st.warning(f"Drawdown-conditioned borrowing scenario unavailable: {exc}")
            for scenario in saved_scenarios:
                growth = scenario["growth_optimum"]
                recommended_debt = float(
                    growth["recommended_debt_ratio_of_portfolio_base"]
                )
                estimated_amount = growth.get("estimated_debt_amount")
                amount_text = (
                    f"₹{float(estimated_amount):,.0f}"
                    if estimated_amount is not None
                    else f"{recommended_debt:.1%} of portfolio value"
                )
                rate_label = f"{float(scenario['rate']):.2%} p.a."
                if abs(float(scenario["rate"]) - LAMF_ADVERTISED_START_RATE) < 1e-9:
                    if recommended_debt <= 0:
                        st.caption(
                            f"Full-history growth diagnostic at {rate_label}: 0% additional debt; "
                            "its all-history bootstrap result is retained for comparison only."
                        )
                    else:
                        boundary_note = (
                            " The model optimum hit the maximum tested debt ratio, so this is a lower bound—not a ceiling."
                            if growth.get("at_search_boundary")
                            else ""
                        )
                        st.caption(
                            f"Full-history growth diagnostic (not the drawdown-conditioned result): "
                            f"{amount_text} ({recommended_debt:.1%} of base; "
                            f"{float(growth['recommended_gross_exposure']):.2f}× gross exposure) at "
                            f"{rate_label}.{boundary_note}"
                        )
                else:
                    st.caption(
                        f"At the {rate_label} financing stress, the model-implied amount is {amount_text} "
                        f"({recommended_debt:.1%} of portfolio base)."
                    )
        else:
            if payload.get("leverage_assessment_error"):
                st.warning(
                    "Automatic leverage analysis could not be completed for this run: "
                    f"{payload['leverage_assessment_error']}"
                )
            else:
                st.info(
                    "No leverage calculation is saved for this analysis yet. The button needs a "
                    "saved daily portfolio-return history; run optimization once to enable the "
                    "on-demand analysis for this and future portfolio runs."
                )

        with st.container(border=True):
            st.markdown("**Tickertape LAMF financing reference**")
            st.markdown(
                f"Tickertape advertises LAMF from **{LAMF_ADVERTISED_START_RATE:.2%} p.a.** "
                f"on the amount used—about **₹{LAMF_ADVERTISED_START_RATE * 100_000:,.0f}/year "
                "per ₹1 lakh borrowed** (before any applicable fees). "
                "The credit is secured by eligible mutual-fund units, not the optimized "
                "stock basket; Tickertape lists indicative limits of 45% of eligible "
                "equity-MF value and 75% of eligible debt-MF value. The actual sanctioned "
                "rate, scheme eligibility and limit depend on the live offer. "
                "[Tickertape LAMF rate and terms](https://www.tickertape.in/loan-against-mutual-funds)."
            )
            st.caption(
                "The scenario assumes the advertised starting rate; confirm the current rate "
                "and fees before borrowing. The app has no eligible-MF "
                "collateral value, so it cannot infer your available LAMF credit limit."
            )

        if saved_scenarios:
            st.info(
                "The amount above is a model-implied growth scenario—not a guaranteed safe amount "
                "or lender-approved limit. The app cannot see your eligible mutual-fund collateral, "
                "actual LAMF haircut/fees, or forced-sale threshold. Confirm the live loan offer "
                "and collateral capacity separately before borrowing."
            )
        else:
            st.caption(f"Saved historical annualized return: {annual_return:.2%}.")
        st.caption(
            "A fall in pledged mutual-fund value can reduce the available collateral "
            "buffer and may require repayment or additional collateral; interest accrues "
            "even if the stock portfolio falls. See [Tickertape LAMF terms](https://www.tickertape.in/loan-against-mutual-funds)."
        )


def render_saved_analysis(placeholder, owner, *, instance_key="page"):
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
        widget_scope = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            f"{instance_key}_{saved_at}",
        ).strip("_")
        holdings_analyzed = int(payload.get("holdings_analyzed") or 0)
        total_invested = float(payload.get("total_invested") or 0.0)
        rebalancing_plan = payload.get("rebalancing_plan") or []
        executable_trade_count = int(
            payload.get("executable_trade_count")
            if payload.get("executable_trade_count") is not None
            else len(rebalancing_plan)
        )
        current_stats = payload.get("current_stats") or {}
        optimal_stats = payload.get("optimal_stats") or {}

        st.subheader("💾 Last Saved Analysis")
        st.caption(f"Saved at: {saved_at}")
        saved_preselection = (payload.get("settings") or {}).get(
            "universe_preselection", {}
        )
        if saved_preselection:
            saved_risk_appetite = (
                (saved_preselection.get("market_data") or {}).get("risk_appetite")
                or {}
            )
            saved_market_read = interpret_risk_appetite_regime(saved_risk_appetite)
            saved_market_cap = saved_preselection.get("market_cap_filter") or {}
            _render_market_read_panel(saved_market_read, title="Saved market backdrop")
            _render_saved_downside_tradeoff(current_stats, optimal_stats)
            if not int(saved_market_cap.get("fresh_known", saved_market_cap.get("known", 0))):
                st.warning(
                    "Market-cap screen was inactive for this saved run: no fresh "
                    "dated snapshot was available, so it excluded no symbols. "
                    f"{int(saved_market_cap.get('missing', saved_preselection.get('eligible', 0))):,} "
                    "market-cap values were missing."
                )
            else:
                st.caption(
                    "Market-cap coverage was present for some eligible symbols; "
                    f"{int(saved_market_cap.get('excluded', 0)):,} were excluded and "
                    f"{int(saved_market_cap.get('missing', 0)):,} missing values were retained."
                )
        else:
            st.info(
                "No saved universe-regime interpretation is available for this older "
                "analysis. It will be included with the next completed optimization."
            )
        _render_saved_reserve_study(payload)
        _render_saved_leverage_check(payload, owner=owner, widget_scope=widget_scope)
        saved_scenarios = payload.get("security_scenario_ranking") or []
        saved_scenario_df = normalize_saved_scenario_ranking(saved_scenarios)
        if not saved_scenario_df.empty:
            saved_leader = saved_scenario_df.iloc[0]
            st.success(
                f"Top single-buy candidate (research-only): {saved_leader.get('Security', '—')} · "
                f"expected net scenario return "
                f"{float(saved_leader.get('Expected net scenario return (%)', 0.0)):+.2f}% · "
                f"return × positive-path score "
                f"{float(saved_leader.get('Return × positive-probability score (%)', 0.0)):+.2f}% · "
                f"historical-bootstrap positive-path frequency "
                f"{float(saved_leader.get('Probability of positive net return (%)', 0.0)):.1f}%."
            )

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

        run_timing = payload.get("run_to_rebalancing_plan_timing") or {}
        if run_timing.get("status") == "finished":
            st.info(
                "**Run-to-plan time:** "
                + format_elapsed(run_timing.get("elapsed_seconds"))
                + " · Finished when the rebalancing plan was generated."
            )

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
            key="download_saved_analysis_main_" + widget_scope,
        )

        saved_lumpsum = payload.get("lumpsum_plan") or {}
        saved_lumpsum_orders = saved_lumpsum.get("orders") or []
        if saved_lumpsum_orders:
            saved_lumpsum_df = pd.DataFrame(saved_lumpsum_orders)
            saved_lumpsum_amount = float(saved_lumpsum.get("amount_inr") or 0.0)
            saved_unallocated_cash = float(saved_lumpsum.get("unallocated_cash") or 0.0)
            saved_execution_sheet_html = lumpsum_execution_sheet_html(
                saved_lumpsum_df,
                saved_lumpsum_amount,
                saved_unallocated_cash,
            )
            with st.expander("Saved Lumpsum Allocation", expanded=False):
                st.caption(
                    f"Lumpsum: ₹{saved_lumpsum_amount:,.2f} | "
                    f"Estimated investment: ₹{saved_lumpsum_df['Estimated Investment INR'].sum():,.2f} | "
                    f"Unallocated cash: ₹{saved_unallocated_cash:,.2f}"
                )
                st.dataframe(
                    saved_lumpsum_df.style.format({
                        "Optimal Weight": "{:.2%}",
                        "Target Amount INR": "₹{:,.2f}",
                        "Latest Price INR": "₹{:,.2f}",
                        "Suggested Quantity": "{:,.0f}",
                        "Estimated Investment INR": "₹{:,.2f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            saved_csv_col, saved_sheet_col, _ = st.columns([1, 1, 5])
            with saved_csv_col:
                st.download_button(
                    "Download CSV",
                    data=saved_lumpsum_df.to_csv(index=False).encode("utf-8"),
                    file_name="lumpsum_optimal_allocation.csv",
                    mime="text/csv",
                    width="content",
                    key="download_saved_lumpsum_allocation_csv_" + widget_scope,
                )
            with saved_sheet_col:
                st.download_button(
                    "Download buy sheet",
                    data=saved_execution_sheet_html.encode("utf-8"),
                    file_name="lumpsum_buy_orders.html",
                    mime="text/html",
                    width="content",
                    key="download_saved_lumpsum_execution_sheet_" + widget_scope,
                )

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
            saved_rebal_df = pd.DataFrame(rebalancing_plan)
            if not saved_rebal_df.empty:
                saved_rebal_df = _sort_rebalance_df_for_priority(saved_rebal_df)
            render_trade_deferral_editor(
                saved_rebal_df,
                owner,
                key_prefix=f"saved_rebalancing_plan_{widget_scope}",
            )

        scenario_config = (payload.get("settings") or {}).get(
            "security_scenario_forecast", {}
        )
        with st.expander("Saved buy scenario ranking", expanded=False):
            if not saved_scenario_df.empty:
                saved_scenario_error = payload.get("security_scenario_ranking_error")
                if saved_scenario_error:
                    st.warning(f"Scenario ranking reported an issue: {saved_scenario_error}")
                st.caption(
                    f"Historical scenarios · {int(scenario_config.get('horizon_sessions', 0))} "
                    f"sessions · {float(scenario_config.get('one_way_cost_bps', 0.0)):.0f} bp "
                    "modeled per side. Research-only; the displayed scenario frequencies "
                    "are not calibrated future probabilities and do not alter saved orders."
                )
                st.dataframe(
                    saved_scenario_df.drop(columns=["Scenario method"], errors="ignore"),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Optimized Weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Expected net scenario return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Median net return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                        "Probability of positive net return (%)": st.column_config.NumberColumn(format="%.1f%%"),
                        "Return × positive-probability score (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )
            elif payload.get("security_scenario_ranking_error"):
                st.info(
                    "No scenario ranking was saved. "
                    + str(payload["security_scenario_ranking_error"])
                )
            else:
                st.caption("No scenario ranking is present in this saved analysis.")


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


@st.cache_data(show_spinner=False, ttl=300, max_entries=64)
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


@st.cache_data(show_spinner=False, ttl="24h", max_entries=64)
def download_fx_history_to_inr(currency, start_date, end_date):
    """Return persistent daily INR-per-unit history with redundant Yahoo routes."""
    major, unit_factor = _currency_major_and_unit_factor(currency)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + timedelta(days=7)

    if major == "INR":
        idx = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
        return pd.Series(1.0, index=idx, name="INR")

    def close_series(ticker):
        try:
            closes, _, _ = sync_market_history(
                MARKET_HISTORY_CACHE_PATH,
                (ticker,),
                start,
                end,
                _fetch_combined_daily_history,
                batch_size=1,
                retry_after_hours=1,
            )
            normalized_ticker = str(ticker).strip().upper()
            if closes.empty or normalized_ticker not in closes.columns:
                return pd.Series(dtype=float)
            return (
                pd.to_numeric(closes[normalized_ticker], errors="coerce")
                .dropna()
                .sort_index()
            )
        except Exception:
            return pd.Series(dtype=float)

    direct_tickers = [f"{major}INR=X"]
    if major == "USD":
        # Yahoo's canonical USD/INR symbol is normally INR=X. Keep the verbose
        # alias too because provider behaviour has changed across releases.
        direct_tickers.append("INR=X")
    for ticker in dict.fromkeys(direct_tickers):
        direct = close_series(ticker)
        if not direct.empty:
            return direct * unit_factor

    inverse = close_series(f"INR{major}=X")
    inverse = inverse.where(inverse > 0).dropna()
    if not inverse.empty:
        return (1.0 / inverse) * unit_factor

    if major != "USD":
        to_usd = close_series(f"{major}USD=X")
        usd_inr = close_series("USDINR=X")
        if usd_inr.empty:
            usd_inr = close_series("INR=X")
        if not to_usd.empty and not usd_inr.empty:
            joined = pd.concat([to_usd.rename("to_usd"), usd_inr.rename("usd_inr")], axis=1)
            joined = joined.sort_index().ffill().dropna()
            if not joined.empty:
                return joined["to_usd"] * joined["usd_inr"] * unit_factor

    raise ValueError(f"No usable historical FX series was found for {currency}/INR.")


def convert_price_history_to_inr(prices, ticker_currency_pairs, fx_histories=None):
    """Convert every asset price series to INR before calculating returns."""
    if prices.empty:
        return prices

    currency_map = {str(t): _normalize_currency_code(c) for t, c in ticker_currency_pairs}
    converted = prices.copy().astype(float)
    fx_histories = fx_histories or {}

    for ticker in converted.columns:
        currency = currency_map.get(str(ticker), "INR")
        if currency == "INR":
            continue
        fx = fx_histories.get(currency)
        if fx is None:
            fx = download_fx_history_to_inr(
                currency, converted.index.min(), converted.index.max()
            )
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


def add_holding_with_quantity(symbol, quantity, owner, average_price=None):
    """Add one resolved holding with its bought quantity and cost basis.

    ``average_price`` is expressed in the security's native quoted currency. If
    omitted, the latest Yahoo price is stored as an explicit placeholder, just
    like the legacy manual-add flow. Existing holdings are never overwritten;
    their quantity/cost basis must be changed in the editor below the form.
    """
    entered = str(symbol or "").strip()
    if not entered:
        raise ValueError("Enter a ticker or Yahoo symbol.")

    try:
        bought_quantity = float(quantity)
    except (TypeError, ValueError):
        raise ValueError("Bought quantity must be a positive number.") from None
    if not np.isfinite(bought_quantity) or bought_quantity <= 0:
        raise ValueError("Bought quantity must be a positive number.")

    explicit_price = average_price not in (None, "")
    if explicit_price:
        try:
            saved_price = float(average_price)
        except (TypeError, ValueError):
            raise ValueError("Average buy price must be a positive number.") from None
        if not np.isfinite(saved_price) or saved_price <= 0:
            raise ValueError("Average buy price must be a positive number.")
    else:
        saved_price = None

    instrument = resolve_yahoo_instrument(entered, get_nse_company_lookup())
    if instrument is None:
        raise ValueError("Ticker could not be resolved on Yahoo Finance/NSE/BSE.")

    if saved_price is None:
        latest = get_latest_price_map((instrument["yahoo_ticker"],))
        saved_price = latest.get(instrument["yahoo_ticker"])
        if saved_price is None or not np.isfinite(saved_price) or saved_price <= 0:
            raise ValueError(
                "A current price was unavailable. Enter your average buy price manually."
            )
        saved_price = float(saved_price)

    now = datetime.now().isoformat(timespec="seconds")
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM master_holdings WHERE owner = ? AND symbol = ?",
            (owner, instrument["symbol"]),
        ).fetchone()
        if existing:
            raise ValueError(
                "This holding already exists. Edit its quantity and average price below."
            )
        conn.execute(
            """
            INSERT INTO master_holdings
                (owner, symbol, stock_name, yahoo_ticker, exchange, currency,
                 quantity, average_price, added_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner, instrument["symbol"], instrument["stock_name"],
                instrument["yahoo_ticker"], instrument["exchange"],
                _normalize_currency_code(instrument["currency"]),
                bought_quantity, saved_price, now, now,
            ),
        )
        conn.commit()

    return {
        "symbol": instrument["symbol"],
        "quantity": bought_quantity,
        "average_price": saved_price,
        "price_source": "entered" if explicit_price else "latest Yahoo placeholder",
    }


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
    seen_tickers = set()

    for entered_symbol in symbols:
        instrument = resolve_yahoo_instrument(entered_symbol, lookup)
        if instrument is None:
            invalid_symbols.append(entered_symbol)
            continue
        ticker = instrument["yahoo_ticker"]
        if ticker not in seen_tickers:
            instruments.append(instrument)
            seen_tickers.add(ticker)

    valid_tickers = [item["yahoo_ticker"] for item in instruments]
    with get_db_connection() as conn:
        existing = {
            row["yahoo_ticker"]
            for row in conn.execute(
                "SELECT yahoo_ticker FROM master_holdings WHERE owner = ? AND yahoo_ticker IN ({})".format(
                    ",".join("?" for _ in valid_tickers)
                ),
                [UNIVERSAL_OWNER, *valid_tickers],
            ).fetchall()
        } if valid_tickers else set()

    duplicates = [
        item["symbol"]
        for item in instruments
        if item["yahoo_ticker"] in existing
    ]
    new_instruments = [
        item for item in instruments if item["yahoo_ticker"] not in existing
    ]

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


def _probe_recent_import_batch(tickers):
    """Return Yahoo tickers with a usable recent close from one batched request."""
    normalized = [str(value).strip().upper() for value in tickers if str(value).strip()]
    if not normalized:
        return set()
    downloaded = _yf_download_quiet(
        normalized,
        period="10d",
        progress=False,
        auto_adjust=True,
        threads=True,
        max_retries=2,
    )
    close_frame = _extract_close_prices_frame(downloaded, normalized)
    return {
        ticker
        for ticker in normalized
        if ticker in close_frame.columns
        and pd.to_numeric(close_frame[ticker], errors="coerce").notna().any()
    }


def _import_validation_metadata(item, active_tickers):
    ticker = str(item["source_symbol"]).strip().upper()
    if ticker not in active_tickers:
        return None
    inferred_exchange, inferred_currency = _infer_market_from_ticker(ticker)
    # The supplied American-company files use unsuffixed Yahoo symbols. Avoid
    # thousands of metadata calls; USD is their quoted currency. Explicit
    # non-US suffixes are rare and are enriched individually for correct FX.
    if "." in ticker:
        metadata = get_yahoo_metadata(ticker)
        exchange = metadata.get("exchange") or inferred_exchange
        currency = metadata.get("currency") or inferred_currency
    else:
        exchange = inferred_exchange
        currency = inferred_currency
    return {
        "yahoo_ticker": ticker,
        "exchange": str(exchange),
        "currency": _normalize_currency_code(currency),
    }


def _format_import_eta(seconds):
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "estimating"
    seconds = int(round(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def validate_universal_import_with_progress(job_id, batch_size=180, max_seconds=720):
    """Validate a durable import job, checkpointing after every Yahoo batch."""
    started = time.perf_counter()
    with get_db_connection() as conn:
        initial = get_import_job(conn, job_id)
    total = max(int(initial["unique_symbols"]), 1)
    initial_processed = int(initial["processed"])
    progress_bar = st.progress(
        float(initial["progress"]),
        text=f"Processed {initial_processed:,} of {total:,} · ETA estimating",
    )
    with st.status("Validating recent Yahoo availability", expanded=True) as status:
        status_line = st.empty()
        batches = 0
        while True:
            with get_db_connection() as conn:
                batch = pending_import_items(conn, job_id, batch_size)
            if not batch:
                break
            tickers = [str(item["source_symbol"]).upper() for item in batch]
            try:
                active_tickers = _probe_recent_import_batch(tickers)
            except Exception as exc:
                status.update(
                    label="Yahoo validation paused; checkpoint preserved",
                    state="error",
                    expanded=True,
                )
                status_line.error(
                    "Yahoo could not complete this batch. No row was changed. "
                    "Use Resume validation later. " + str(exc)
                )
                with get_db_connection() as conn:
                    return get_import_job(conn, job_id)

            active_metadata = {}
            for item in batch:
                metadata = _import_validation_metadata(item, active_tickers)
                if metadata is not None:
                    active_metadata[str(item["source_symbol"]).upper()] = metadata
            with get_db_connection() as conn:
                current = record_validation_batch(
                    conn,
                    job_id,
                    tickers,
                    active_metadata,
                    max_attempts=3,
                )
            batches += 1
            elapsed = max(time.perf_counter() - started, 0.001)
            newly_processed = max(int(current["processed"]) - initial_processed, 0)
            remaining = int(current["unique_symbols"]) - int(current["processed"])
            rate = newly_processed / elapsed if newly_processed else 0.0
            eta = remaining / rate if rate > 0 else None
            progress_bar.progress(
                float(current["progress"]),
                text=(
                    f'Processed {int(current["processed"]):,} of '
                    f'{int(current["unique_symbols"]):,} · ETA {_format_import_eta(eta)}'
                ),
            )
            status_line.caption(
                f"Batch {batches:,} · {len(active_metadata):,} recently active · "
                f"checkpoint saved · elapsed {_format_import_eta(elapsed)}"
            )
            if elapsed >= max_seconds and current["counts"].get("pending", 0):
                status.update(
                    label="Validation paused at the safe time limit",
                    state="complete",
                    expanded=False,
                )
                return current

        with get_db_connection() as conn:
            finished = get_import_job(conn, job_id)
        progress_bar.progress(1.0, text=f"Processed {finished['unique_symbols']:,} of {total:,}")
        status.update(
            label="Validation complete; review before applying",
            state="complete",
            expanded=False,
        )
        return finished


def scan_universal_cleaner_with_progress(job_id, batch_size=180, max_seconds=720):
    """Resume a cleaner job with durable checkpoints and a bounded UI run."""
    started = time.perf_counter()
    with get_db_connection() as conn:
        initial = get_cleaner_job(conn, job_id)
    total = max(int(initial["total_symbols"]), 1)
    initial_done = total - int(initial["counts"]["history_pending"])
    progress_bar = st.progress(
        initial_done / total,
        text=f"Analysed {initial_done:,} of {total:,} price histories · ETA estimating",
    )
    with st.status("Screening the selected universe", expanded=True) as status:
        status_line = st.empty()
        while True:
            with get_db_connection() as conn:
                batch = pending_history_items(conn, job_id, batch_size)
            if not batch:
                break
            tickers = [item["yahoo_ticker"] for item in batch]
            history, failures = _download_close_prices_resilient(
                tickers,
                period="1y",
                batch_size=min(batch_size, 90),
                threads=True,
            )
            results = {}
            for item in batch:
                ticker = item["yahoo_ticker"]
                metrics = score_price_history(
                    history[ticker] if ticker in history.columns else pd.Series(dtype=float)
                )
                if metrics["history_status"] == "no_price" and ticker in failures:
                    metrics["reason"] = "No usable recent price history; review manually."
                results[item["symbol"]] = metrics
            with get_db_connection() as conn:
                current = record_history_batch(conn, job_id, results)
            done = total - int(current["counts"]["history_pending"])
            elapsed = max(time.perf_counter() - started, 0.001)
            rate = max(done - initial_done, 0) / elapsed
            eta = (total - done) / rate if rate > 0 else None
            progress_bar.progress(
                done / total,
                text=(
                    f"Analysed {done:,} of {total:,} price histories · "
                    f"ETA {_format_import_eta(eta)}"
                ),
            )
            status_line.caption(
                f"Price checkpoint saved · elapsed {_format_import_eta(elapsed)}"
            )
            if elapsed >= max_seconds and current["counts"]["history_pending"]:
                status.update(
                    label="Cleaner paused at the safe time limit; checkpoint preserved",
                    state="complete",
                    expanded=False,
                )
                return current

        with get_db_connection() as conn:
            current = get_cleaner_job(conn, job_id)
            if current["include_insider"] and not current["counts"]["insider_pending"]:
                prepare_insider_shortlist(conn, job_id)
                current = get_cleaner_job(conn, job_id)

        if current["counts"]["insider_pending"]:
            import yfinance as yf

            status_line.caption(
                "Checking recent management transactions for a small U.S. shortlist. "
                "Missing coverage remains neutral."
            )
            while True:
                with get_db_connection() as conn:
                    items = pending_insider_items(conn, job_id, limit=1)
                if not items:
                    break
                item = items[0]
                try:
                    transactions = yf.Ticker(item["yahoo_ticker"]).get_insider_transactions()
                    signal = insider_sale_signal(transactions)
                except Exception:
                    signal = {"status": "unavailable"}
                with get_db_connection() as conn:
                    record_insider_signal(conn, job_id, item["symbol"], signal)
                    current = get_cleaner_job(conn, job_id)
                if time.perf_counter() - started >= max_seconds:
                    status.update(
                        label="Cleaner paused at the safe time limit; checkpoint preserved",
                        state="complete",
                        expanded=False,
                    )
                    return current

        with get_db_connection() as conn:
            finished = finalize_cleaner_job(conn, job_id)
        progress_bar.progress(1.0, text=f"Analysed {total:,} of {total:,} symbols")
        if finished["status"] == "blocked":
            status.update(
                label="Cleaner stopped: market-data coverage was insufficient",
                state="error",
                expanded=True,
            )
        else:
            status.update(
                label="Cleaner proposal ready for review",
                state="complete",
                expanded=False,
            )
        return finished


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


def extend_allocation_with_universal_candidates(
    portfolio_df,
    maximum_candidates=UNIVERSAL_PRESELECTION_CAP,
    market_cap_exclusion_fraction=MARKET_CAP_EXCLUSION_FRACTION,
    owner=None,
    progress_callback=None,
):
    """Add zero-quantity rows for Universal Portfolio symbols not already held.

    This lets the optimizer treat every shared Universal Portfolio symbol as a
    candidate for the optimal allocation (current weight 0) alongside the user's
    real holdings, so the rebalancing plan can recommend buying new stocks too.
    Returns (extended_df, added_symbols, preselection_summary).
    """
    candidate_cap = max(int(maximum_candidates), 1)
    universal_df = load_master_holdings(UNIVERSAL_OWNER)
    if universal_df.empty:
        return portfolio_df, [], {
            "method": "scalable-preselection-v1", "eligible": 0,
            "shortlisted": 0, "cap": candidate_cap,
        }

    with get_db_connection() as conn:
        exclusions = optimizer_exclusions_frame(conn)
        quarterly_deferrals = (
            active_trade_deferrals(conn, owner) if owner else pd.DataFrame()
        )
    universal_df = filter_optimizer_candidates(universal_df, exclusions)
    universal_df, quarterly_excluded = exclude_deferred_candidates(
        universal_df,
        quarterly_deferrals,
    )
    if universal_df.empty:
        return portfolio_df, [], {
            "method": "scalable-preselection-v1", "eligible": 0,
            "shortlisted": 0, "cap": candidate_cap,
            "quarterly_excluded": quarterly_excluded,
        }

    held_symbols = set(portfolio_df["Symbol"]) if not portfolio_df.empty else set()
    candidates = universal_df[~universal_df["Symbol"].isin(held_symbols)].copy()
    if candidates.empty:
        return portfolio_df, [], {
            "method": "scalable-preselection-v1", "eligible": 0,
            "shortlisted": 0, "cap": candidate_cap,
        }

    candidates["Symbol"] = candidates["Symbol"].map(normalize_portfolio_symbol)
    candidates["Yahoo Ticker"] = candidates["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    candidates["Currency"] = candidates["Currency"].fillna("").map(_normalize_currency_code)
    candidates = candidates[candidates["Yahoo Ticker"] != ""]
    if candidates.empty:
        return portfolio_df, [], {
            "method": "scalable-preselection-v1", "eligible": 0,
            "shortlisted": 0, "cap": candidate_cap,
        }

    with get_db_connection() as conn:
        market_cap_refresh = refresh_market_cap_sample(
            conn,
            candidates,
            owner=UNIVERSAL_OWNER,
            sample_fraction=0.01,
            max_age_days=MARKET_CAP_MAX_AGE_DAYS,
        )
    if market_cap_refresh.get("updated"):
        refreshed_universal = load_master_holdings(UNIVERSAL_OWNER)
        refreshed_universal = filter_optimizer_candidates(refreshed_universal, exclusions)
        refreshed_universal, _ = exclude_deferred_candidates(refreshed_universal, quarterly_deferrals)
        candidates = refreshed_universal[~refreshed_universal["Symbol"].isin(held_symbols)].copy()
        candidates["Symbol"] = candidates["Symbol"].map(normalize_portfolio_symbol)
        candidates["Yahoo Ticker"] = candidates["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
        candidates["Currency"] = candidates["Currency"].fillna("").map(_normalize_currency_code)
        candidates = candidates[candidates["Yahoo Ticker"] != ""]

    candidates, market_cap_report = filter_candidates_by_market_cap(
        candidates,
        exclusion_fraction=market_cap_exclusion_fraction,
    )
    market_cap_report["auto_refresh"] = market_cap_refresh
    if candidates.empty:
        return portfolio_df, [], {
            "method": "scalable-preselection-v1", "eligible": 0,
            "shortlisted": 0, "cap": candidate_cap,
            "market_cap_filter": market_cap_report,
        }

    eligible_count = len(candidates)
    ticker_clusters = tuple(
        sorted(
            (
                str(row["Yahoo Ticker"]),
                f"{str(row.get('Exchange') or 'Unknown')} · {str(row.get('Currency') or 'Unknown')}",
                str(row.get("Currency") or "INR"),
            )
            for _, row in candidates.iterrows()
        )
    )
    (
        shortlisted_tickers,
        recent_price_map,
        shortlist_report,
        shortlist_diagnostics,
    ) = preselect_universal_candidates(
        ticker_clusters,
        maximum_candidates=candidate_cap,
        _progress_callback=progress_callback,
    )
    if shortlisted_tickers:
        candidates = candidates[candidates["Yahoo Ticker"].isin(shortlisted_tickers)].copy()
        latest_price_map = recent_price_map
    elif eligible_count <= candidate_cap:
        latest_price_map = get_latest_price_map(tuple(candidates["Yahoo Ticker"].tolist()))
    else:
        recovered = int(shortlist_diagnostics.get("recovered", 0))
        requests = int(shortlist_diagnostics.get("requests", 0))
        failed = int(shortlist_diagnostics.get("failed_or_empty_requests", 0))
        raise ValueError(
            "Broad-universe preselection recovered "
            f"{recovered:,} of {eligible_count:,} recent price histories across "
            f"{requests:,} batched Yahoo requests ({failed:,} failed or empty). "
            "No unranked candidate set was sent to the optimizer. Retry when Yahoo "
            "data is available."
        )
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
        return portfolio_df, [], {
            "method": "scalable-preselection-v1",
            "eligible": eligible_count,
            "shortlisted": 0,
            "cap": candidate_cap,
            "market_cap_filter": market_cap_report,
            "quarterly_excluded": quarterly_excluded,
        }

    extended_df = pd.concat([portfolio_df, pd.DataFrame(rows)], ignore_index=True)
    return extended_df, added_symbols, {
        "method": "scalable-preselection-v1",
        "eligible": eligible_count,
        "shortlisted": len(added_symbols),
        "cap": candidate_cap,
        "scored": int(len(shortlist_report)),
        "market_data": dict(shortlist_diagnostics),
        "market_cap_filter": market_cap_report,
        "quarterly_excluded": quarterly_excluded,
    }

# =========================================================
# RETURNS / OPTIMIZATION
# =========================================================

def _download_volume_history_resilient(
    tickers,
    *,
    start=None,
    end=None,
    period=None,
    batch_size=12,
    threads=False,
    fallback_missing=True,
):
    """Download volume history with the same resilient-per-ticker logic as price data."""
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
                "threads": threads,
            }
            if period is not None:
                kwargs["period"] = period
            else:
                kwargs["start"] = start
                kwargs["end"] = end

            downloaded = _yf_download_quiet(batch, **kwargs)
            batch_frame = _extract_volume_frame(downloaded, batch)
        except Exception as exc:
            batch_frame = pd.DataFrame()
            batch_error = str(exc) or exc.__class__.__name__
            for ticker in batch:
                failures.setdefault(ticker, batch_error)

        recovered = set(batch_frame.columns)
        for ticker in batch:
            if ticker in recovered:
                continue
            if not fallback_missing:
                failures.setdefault(ticker, "No usable volume returned by the bulk request.")
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
                single_frame = _extract_volume_frame(single_download, [ticker])
                if single_frame.empty:
                    failures.setdefault(ticker, "No usable volume history returned by Yahoo Finance.")
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


def _extract_volume_frame(data, expected_tickers):
    """Normalize yfinance output into a Volume DataFrame keyed by ticker."""
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
        if "Volume" in frame.columns.get_level_values(0):
            frame = frame["Volume"]
        else:
            return pd.DataFrame()
    else:
        if "Volume" not in frame.columns:
            return pd.DataFrame()
        if len(expected_tickers) == 1:
            frame = frame[["Volume"]].rename(columns={"Volume": expected_tickers[0]})
        else:
            frame = frame[["Volume"] * 0]
            return pd.DataFrame()

    frame.columns = [str(col).strip().upper() for col in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    return frame


def _fetch_combined_daily_history(tickers, start, end):
    """Fetch close and volume together for one persistent-cache batch."""
    normalized = [
        str(ticker).strip().upper()
        for ticker in dict.fromkeys(tickers)
        if str(ticker).strip()
    ]
    if not normalized:
        return pd.DataFrame(), pd.DataFrame(), {}
    try:
        downloaded = _yf_download_quiet(
            normalized,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        closes = _extract_close_prices_frame(downloaded, normalized)
        volumes = _extract_volume_frame(downloaded, normalized)
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        return pd.DataFrame(), pd.DataFrame(), {
            ticker: message for ticker in normalized
        }
    recovered = set(closes.columns) | set(volumes.columns)
    failures = {
        ticker: "No usable daily history returned by Yahoo Finance."
        for ticker in normalized
        if ticker not in recovered
    }
    return closes, volumes, failures


def _single_ticker_field(data, field, ticker):
    """Extract one numeric Yahoo field without depending on column orientation."""
    if data is None or getattr(data, "empty", True):
        return pd.Series(dtype=float)
    frame = data
    if isinstance(frame.columns, pd.MultiIndex):
        if field not in frame.columns.get_level_values(0):
            return pd.Series(dtype=float)
        value = frame[field]
        if isinstance(value, pd.DataFrame):
            if ticker in value.columns:
                value = value[ticker]
            elif len(value.columns) == 1:
                value = value.iloc[:, 0]
            else:
                return pd.Series(dtype=float)
    else:
        if field not in frame.columns:
            return pd.Series(dtype=float)
        value = frame[field]
    return pd.to_numeric(value, errors="coerce").sort_index()


def _download_price_integrity_evidence(
    ticker, start_date, end_date, integrity_version=PRICE_INTEGRITY_VERSION
):
    """Fetch raw corporate-action evidence only for a suspicious price series."""
    del integrity_version  # Included in the cache key for deterministic invalidation.
    ticker = str(ticker).strip().upper()
    data = _yf_download_quiet(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        actions=True,
        repair=False,
        threads=False,
    )
    raw_close = _single_ticker_field(data, "Close", ticker)
    adjusted_close = _single_ticker_field(data, "Adj Close", ticker)
    if adjusted_close.dropna().empty:
        adjusted_close = raw_close.copy()
    if raw_close.dropna().empty:
        raise ValueError("No raw close history was returned for integrity validation.")
    return {
        "raw_close": raw_close,
        "adjusted_close": adjusted_close,
        "dividends": _single_ticker_field(data, "Dividends", ticker),
        "stock_splits": _single_ticker_field(data, "Stock Splits", ticker),
    }


_download_price_integrity_evidence = st.cache_data(
    show_spinner=False,
    ttl="24h",
    max_entries=2048,
)(_download_price_integrity_evidence)


def _download_price_integrity_evidence_batch(
    tickers, start_date, end_date, integrity_version=PRICE_INTEGRITY_VERSION
):
    """Batch corporate-action evidence so integrity checks remain scalable."""
    del integrity_version  # Included in the cache key for deterministic invalidation.
    tickers = tuple(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())
    if not tickers:
        return {}
    data = _yf_download_quiet(
        list(tickers),
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        actions=True,
        repair=False,
        threads=True,
    )
    evidence = {}
    for ticker in tickers:
        raw_close = _single_ticker_field(data, "Close", ticker)
        if raw_close.dropna().empty:
            continue
        adjusted_close = _single_ticker_field(data, "Adj Close", ticker)
        evidence[ticker] = {
            "raw_close": raw_close,
            "adjusted_close": (
                adjusted_close if not adjusted_close.dropna().empty else raw_close.copy()
            ),
            "dividends": _single_ticker_field(data, "Dividends", ticker),
            "stock_splits": _single_ticker_field(data, "Stock Splits", ticker),
        }
    return evidence


_download_price_integrity_evidence_batch = st.cache_data(
    show_spinner=False,
    ttl="24h",
    max_entries=4096,
)(_download_price_integrity_evidence_batch)


def apply_price_integrity_gate(prices, owned_tickers=()):
    """Repair split discontinuities and quarantine unresolved candidate histories."""
    if prices is None or prices.empty:
        return prices, pd.DataFrame()
    flagged = [
        str(column).strip().upper()
        for column in prices.columns
        if extreme_return_dates(
            prices[column], threshold=DEFAULT_EXTREME_RETURN_THRESHOLD
        )
    ]
    if not flagged:
        return prices, pd.DataFrame()

    observed_start = pd.Timestamp(prices.index.min()).normalize().strftime("%Y-%m-%d")
    observed_end = (
        pd.Timestamp(prices.index.max()).normalize() + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    evidence = {}
    for batch in _chunked(flagged, 12):
        try:
            evidence.update(_download_price_integrity_evidence_batch(
                tuple(batch),
                observed_start,
                observed_end,
                integrity_version=PRICE_INTEGRITY_VERSION,
            ))
        except Exception:
            for ticker in batch:
                try:
                    evidence[ticker] = _download_price_integrity_evidence(
                        ticker,
                        observed_start,
                        observed_end,
                        integrity_version=PRICE_INTEGRITY_VERSION,
                    )
                except Exception:
                    evidence[ticker] = None
    return reconcile_price_frame(
        prices,
        evidence,
        owned_tickers=owned_tickers,
        threshold=DEFAULT_EXTREME_RETURN_THRESHOLD,
    )


def _download_recent_market_data_batch(
    tickers,
    period="2y",
    download_threads=16,
    integrity_version=PRICE_INTEGRITY_VERSION,
):
    """Download one cacheable shortlist batch without emitting Streamlit elements."""
    del integrity_version  # Included in the cache key for deterministic invalidation.
    batch = tuple(
        str(ticker).strip().upper()
        for ticker in tickers
        if str(ticker).strip()
    )
    if not batch:
        return pd.DataFrame(), pd.DataFrame(), True
    try:
        downloaded = _yf_download_quiet(
            batch,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=max(int(download_threads), 1),
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), True
    closes = _extract_close_prices_frame(downloaded, batch)
    volumes = _extract_volume_frame(downloaded, batch)
    return closes, volumes, bool(closes.empty)


_download_recent_market_data_batch = st.cache_data(
    show_spinner=False,
    ttl="24h",
    max_entries=256,
)(_download_recent_market_data_batch)


def _download_recent_market_data_bulk(
    tickers,
    period="2y",
    batch_size=120,
    fallback_batch_size=40,
    download_threads=16,
    integrity_version=PRICE_INTEGRITY_VERSION,
    _progress_callback=None,
):
    """Fetch recent data in fast batches, retrying missing names in smaller batches."""
    del integrity_version  # Included in the cache key for deterministic invalidation.
    unique_tickers = tuple(
        str(ticker).strip().upper()
        for ticker in dict.fromkeys(tickers)
        if str(ticker).strip()
    )
    close_frames = []
    volume_frames = []
    requested_batches = 0
    failed_batches = 0
    recovered_tickers = set()

    primary_size = max(int(batch_size), 1)
    retry_size = max(min(int(fallback_batch_size), primary_size), 1)
    primary_batches = list(_chunked(unique_tickers, primary_size))
    total_primary_batches = len(primary_batches)

    def report_progress(*, primary_batch, retry=False, batch_size_now=0):
        if _progress_callback is None:
            return
        fraction = (
            min(max(int(primary_batch), 0), total_primary_batches)
            / max(total_primary_batches, 1)
        )
        details = {
            "primary_batch": int(primary_batch),
            "primary_batches": int(total_primary_batches),
            "requests": int(requested_batches),
            "recovered": int(len(recovered_tickers)),
            "retry": bool(retry),
            "batch_size": int(batch_size_now),
        }
        try:
            _progress_callback(float(fraction), details)
        except Exception:
            # Progress rendering must never invalidate downloaded market data.
            pass

    def collect(batch, *, primary_batch, retry=False):
        nonlocal requested_batches, failed_batches
        requested_batches += 1
        closes, volumes, failed = _download_recent_market_data_batch(
            tuple(batch),
            period=period,
            download_threads=download_threads,
            integrity_version=PRICE_INTEGRITY_VERSION,
        )
        if failed:
            failed_batches += 1
        if not closes.empty:
            close_frames.append(closes)
        if not volumes.empty:
            volume_frames.append(volumes)
        recovered = set(closes.columns)
        recovered_tickers.update(recovered)
        report_progress(
            primary_batch=primary_batch,
            retry=retry,
            batch_size_now=len(batch),
        )
        return recovered

    report_progress(primary_batch=0)
    for primary_index, batch in enumerate(primary_batches, start=1):
        recovered = collect(batch, primary_batch=primary_index)
        missing = [ticker for ticker in batch if ticker not in recovered]
        if missing and retry_size < len(batch):
            for retry_batch in _chunked(missing, retry_size):
                collect(
                    retry_batch,
                    primary_batch=primary_index,
                    retry=True,
                )
    closes = (
        pd.concat(close_frames, axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()
        if close_frames
        else pd.DataFrame()
    )
    native_recent_prices = (
        closes.ffill().iloc[-1].to_dict() if not closes.empty else {}
    )
    volumes = (
        pd.concat(volume_frames, axis=1).loc[:, lambda frame: ~frame.columns.duplicated()].sort_index()
        if volume_frames
        else pd.DataFrame()
    )
    integrity_report = pd.DataFrame()
    if not closes.empty:
        if _progress_callback is not None:
            try:
                _progress_callback(
                    1.0,
                    {
                        "primary_batch": int(total_primary_batches),
                        "primary_batches": int(total_primary_batches),
                        "requests": int(requested_batches),
                        "recovered": int(len(closes.columns)),
                        "validating_integrity": True,
                    },
                )
            except Exception:
                pass
        closes, integrity_report = apply_price_integrity_gate(closes)
        volumes = volumes.reindex(columns=closes.columns)
    diagnostics = {
        "requested": len(unique_tickers),
        "recovered": int(len(closes.columns)),
        "requests": int(requested_batches),
        "failed_or_empty_requests": int(failed_batches),
        "primary_batch_size": primary_size,
        "fallback_batch_size": retry_size,
        "price_integrity_version": PRICE_INTEGRITY_VERSION,
        "price_integrity_flagged": int(len(integrity_report)),
        "price_integrity_quarantined": int(
            integrity_report["Status"].astype(str).str.startswith("QUARANTINED_").sum()
        )
        if not integrity_report.empty
        else 0,
    }
    return closes, volumes, diagnostics, native_recent_prices


def preselect_universal_candidates(
    ticker_cluster_currency_rows,
    maximum_candidates=400,
    _progress_callback=None,
):
    """Build an INR-adjusted recent-data shortlist before full-history analysis."""
    rows = tuple(
        (
            str(ticker).strip().upper(),
            str(cluster),
            _normalize_currency_code(currency),
        )
        for ticker, cluster, currency in ticker_cluster_currency_rows
        if str(ticker).strip()
    )
    tickers = tuple(dict.fromkeys(ticker for ticker, _, _ in rows))
    if not tickers:
        return [], {}, pd.DataFrame(), {"requested": 0, "recovered": 0, "requests": 0}
    if len(tickers) <= int(maximum_candidates):
        prices = get_latest_price_map(tickers)
        diagnostics = {
            "requested": len(tickers),
            "recovered": len(prices),
            "requests": 0,
            "method": "direct-price-map",
        }
        return (
            list(tickers),
            prices,
            pd.DataFrame({"Ticker": list(tickers), "Selected": True}),
            diagnostics,
        )

    regime_tickers = tuple(
        dict.fromkeys(
            ticker
            for numerator, denominator, _ in RISK_APPETITE_PAIRS
            for ticker in (numerator, denominator)
        )
    )
    download_tickers = tuple(dict.fromkeys((*tickers, *regime_tickers)))
    local_closes, volumes, diagnostics, native_recent_prices = _download_recent_market_data_bulk(
        download_tickers,
        period="2y",
        batch_size=120,
        fallback_batch_size=40,
        integrity_version=PRICE_INTEGRITY_VERSION,
        _progress_callback=_progress_callback,
    )
    if local_closes.empty:
        diagnostics.update({"ranking_currency": "INR", "inr_adjusted": True})
        return [], {}, pd.DataFrame(), diagnostics
    benchmark_closes = local_closes.reindex(
        columns=[ticker for ticker in regime_tickers if ticker in local_closes.columns]
    )
    risk_appetite = calculate_risk_appetite_regime(benchmark_closes)
    local_closes = local_closes.reindex(
        columns=[ticker for ticker in tickers if ticker in local_closes.columns]
    )
    volumes = volumes.reindex(columns=local_closes.columns)
    ticker_currencies = {ticker: currency for ticker, _, currency in rows}
    fx_histories = {}
    fx_download_failures = []
    for currency in sorted(set(ticker_currencies.values()) - {"INR"}):
        try:
            fx_histories[currency] = download_fx_history_to_inr(
                currency,
                local_closes.index.min(),
                local_closes.index.max(),
            )
        except Exception:
            fx_download_failures.append(currency)
    closes, omitted_currencies = convert_candidate_history_to_inr(
        local_closes,
        ticker_currencies,
        fx_histories,
    )
    omitted_currencies = sorted(set(omitted_currencies) | set(fx_download_failures))
    volumes = volumes.reindex(columns=closes.columns)
    diagnostics.update(
        {
            "ranking_currency": "INR",
            "inr_adjusted": True,
            "fx_currencies_requested": len(set(ticker_currencies.values()) - {"INR"}),
            "fx_currencies_omitted": omitted_currencies,
            "risk_appetite": risk_appetite,
        }
    )
    selected, report = rank_scalable_candidates(
        closes,
        volumes,
        {ticker: cluster for ticker, cluster, _ in rows},
        maximum_candidates=int(maximum_candidates),
        minimum_per_cluster=5,
        risk_appetite_regime=risk_appetite,
    )
    if closes.empty:
        return [], {}, report, diagnostics
    price_map = {
        ticker: float(native_recent_prices[ticker])
        for ticker in selected
        if ticker in native_recent_prices
        and pd.notna(native_recent_prices[ticker])
        and np.isfinite(float(native_recent_prices[ticker]))
        and float(native_recent_prices[ticker]) > 0
    }
    selected = [ticker for ticker in selected if ticker in price_map]
    diagnostics["scored"] = int(len(report))
    diagnostics["selected"] = int(len(selected))
    return selected, price_map, report, diagnostics


def effective_history_end(end_date=None, buffer_days=0):
    """Return (anchor, last included date), counting the buffer in business days.

    A calendar-day buffer would collapse (Mon-1, Mon-2, Mon-3 all land on Friday)
    and could display a Saturday or Sunday that has no market data. Exchange
    holidays are still not modelled, so treat the result as "on or before".
    """
    anchor = pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date)
    anchor = anchor.normalize()

    business_day = BDay()
    steps = max(int(buffer_days), 0)
    target = anchor if steps == 0 else anchor - steps * business_day
    return anchor, business_day.rollback(target)


def _resolve_history_window_end(end_date, buffer_days):
    """Return the anchor plus the exclusive `end` string yfinance expects.

    The anchor is returned unadjusted so callers can pass it back down together
    with buffer_days without applying the buffer twice.
    """
    anchor, last_included = effective_history_end(end_date, buffer_days)
    return anchor, (last_included + timedelta(days=1)).strftime("%Y-%m-%d")


MARKET_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE = clock_time(15, 30)
# Yahoo needs a little time after the close before the daily bar stops moving.
BAR_SETTLE_MINUTES = 30


def auto_history_buffer_days(now=None):
    """Return (trading days to skip, reason) so the last bar is always a closed session.

    Weekends need no buffer because `effective_history_end` already rolls back to
    Friday. A holiday needs none either: Yahoo simply returns no row for it, and the
    intersection logic then ends on the previous real session.
    """
    now = datetime.now(MARKET_TIMEZONE) if now is None else now

    if now.weekday() >= 5:
        return 0, f"{now:%A} is not a trading day — the previous session is used."

    settled_at = (
        datetime.combine(now.date(), MARKET_CLOSE)
        + timedelta(minutes=BAR_SETTLE_MINUTES)
    ).time()

    if now.time() < settled_at:
        return 1, (
            f"The {now:%A} session has not settled yet "
            f"(NSE closes {MARKET_CLOSE:%H:%M} IST) — today is skipped."
        )

    return 0, f"The {now:%A} session has closed — today is included."


def _download_close_history_batch(symbols, start_date, effective_end):
    """Download one stable history batch so adjacent shortlist caps can reuse it."""
    return _download_close_prices_resilient(
        list(symbols),
        start=start_date,
        end=effective_end,
        batch_size=max(len(symbols), 1),
        threads=True,
        # A broad candidate universe must not fan a partially empty bulk response
        # out into hundreds of serial per-ticker requests. Owned holdings receive
        # a small, explicit recovery pass in ``download_close_history`` below.
        fallback_missing=False,
    )


_download_close_history_batch = st.cache_data(
    show_spinner=False,
    ttl="24h",
    max_entries=256,
)(_download_close_history_batch)


def download_close_history(
    symbols,
    start_date=DEFAULT_HISTORY_START_DATE,
    end_date=None,
    buffer_days=0,
    owned_tickers=(),
):
    """Download closing-price history once and reuse it during the same analysis."""
    _, effective_end = _resolve_history_window_end(end_date, buffer_days)

    frames = []
    failures = {}
    normalized_symbols = tuple(
        str(symbol).strip().upper()
        for symbol in dict.fromkeys(symbols)
        if str(symbol).strip()
    )
    for batch in _chunked(normalized_symbols, FULL_HISTORY_BATCH_SIZE):
        batch_prices, batch_failures = _download_close_history_batch(
            tuple(batch),
            start_date,
            effective_end,
        )
        if batch_prices is not None and not batch_prices.empty:
            frames.append(batch_prices)
        failures.update(batch_failures or {})

    recovered = {
        str(column).strip().upper()
        for frame in frames
        for column in frame.columns
    }
    owned_missing = [
        str(ticker).strip().upper()
        for ticker in dict.fromkeys(owned_tickers)
        if str(ticker).strip() and str(ticker).strip().upper() not in recovered
    ]
    # Protect real holdings from a partial broad-market response. This recovery
    # pass is deliberately limited to owned symbols; unavailable zero-quantity
    # candidates are safely omitted and can be reconsidered on the next run.
    for batch in _chunked(owned_missing, 12):
        owned_prices, owned_failures = _download_close_prices_resilient(
            list(batch),
            start=start_date,
            end=effective_end,
            batch_size=max(len(batch), 1),
            threads=True,
            fallback_missing=True,
        )
        if owned_prices is not None and not owned_prices.empty:
            frames.append(owned_prices)
        failures.update(owned_failures or {})

    prices = (
        pd.concat(frames, axis=1)
        .loc[:, lambda frame: ~frame.columns.duplicated()]
        .sort_index()
        if frames
        else pd.DataFrame()
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

    prices, integrity_report = apply_price_integrity_gate(
        prices, owned_tickers=owned_tickers
    )
    prices.attrs["price_integrity_report"] = integrity_report.to_dict(orient="records")
    prices.attrs["price_integrity_version"] = PRICE_INTEGRITY_VERSION
    return prices


def _download_volume_history_batch(symbols, start_date, effective_end):
    """Download one stable volume batch so adjacent shortlist caps can reuse it."""
    return _download_volume_history_resilient(
        list(symbols),
        start=start_date,
        end=effective_end,
        batch_size=max(len(symbols), 1),
        threads=True,
        # Volume is ranking evidence, not a reason to retry thousands of
        # candidates one by one. Missing volume remains NaN and is handled by the
        # existing conservative liquidity/history filters.
        fallback_missing=False,
    )


_download_volume_history_batch = st.cache_data(
    show_spinner=False,
    ttl="24h",
    max_entries=256,
)(_download_volume_history_batch)


def download_volume_history(
    symbols,
    start_date=DEFAULT_HISTORY_START_DATE,
    end_date=None,
    buffer_days=0,
):
    """Download daily volume history so low-liquidity names can be dropped first."""
    _, effective_end = _resolve_history_window_end(end_date, buffer_days)
    frames = []
    failures = {}
    normalized_symbols = tuple(
        str(symbol).strip().upper()
        for symbol in dict.fromkeys(symbols)
        if str(symbol).strip()
    )
    for batch in _chunked(normalized_symbols, FULL_HISTORY_BATCH_SIZE):
        batch_volumes, batch_failures = _download_volume_history_batch(
            tuple(batch),
            start_date,
            effective_end,
        )
        if batch_volumes is not None and not batch_volumes.empty:
            frames.append(batch_volumes)
        failures.update(batch_failures or {})

    volumes = (
        pd.concat(frames, axis=1)
        .loc[:, lambda frame: ~frame.columns.duplicated()]
        .sort_index()
        if frames
        else pd.DataFrame()
    )

    volumes = volumes.dropna(axis=1, how="all")
    if volumes.empty:
        raise ValueError(
            _format_download_failure_message(
                [str(symbol).strip().upper() for symbol in symbols],
                failures,
                context="historical volume history",
            )
        )

    return volumes


@st.cache_data(show_spinner=False)
def find_drop_bottom_pct_nearest_target(
    symbols,
    target_trading_days=TRADING_DAYS_PER_YEAR,
    start_date=DEFAULT_HISTORY_START_DATE,
    end_date=None,
    buffer_days=0,
):
    """Return the 0.01 drop fraction producing days nearest the configured year.

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


def select_redundant_tickers(
    log_returns,
    avg_volume=None,
    correlation_threshold=0.80,
    aggregation_days=5,
    owned_tickers=(),
):
    """Keep owned assets, then one liquid representative per candidate cluster.

    Eight gold ETFs are one bet, not eight, but a per-ticker weight cap cannot see
    that: blocked at the cap on one ticker, the optimizer simply rebuilds the same
    exposure out of clones. Collapsing them first is what makes the cap meaningful.
    Existing holdings are never dropped by this statistical redundancy rule: that
    would manufacture an exit merely because a correlated candidate is more liquid.
    """
    empty_report = pd.DataFrame(columns=["Dropped", "Kept Instead", "Correlation"])
    tickers = list(log_returns.columns)
    if len(tickers) < 2 or not 0.0 < correlation_threshold < 1.0:
        return tickers, empty_report

    # A thinly traded ETF repeats its previous close, so its daily returns are full
    # of artificial zeros that hide a duplicate. Non-overlapping multi-day sums let
    # the shared underlying factor show through.
    aggregated = log_returns
    if aggregation_days > 1 and len(log_returns) >= aggregation_days * 10:
        buckets = np.arange(len(log_returns)) // int(aggregation_days)
        aggregated = log_returns.groupby(buckets).sum()

    correlation = aggregated.corr().abs()

    if avg_volume is None:
        liquidity = pd.Series(0.0, index=tickers)
    else:
        liquidity = pd.to_numeric(avg_volume, errors="coerce").reindex(tickers).fillna(0.0)

    owned = {str(ticker).strip().upper() for ticker in owned_tickers}
    # Owned holdings precede candidates and can never remove one another. Among
    # candidates, volume still chooses the more tradable representative.
    liquidity_order = liquidity.sort_values(
        ascending=False, kind="mergesort").index.tolist()
    ordered = (
        [ticker for ticker in liquidity_order if str(ticker).upper() in owned]
        + [ticker for ticker in liquidity_order if str(ticker).upper() not in owned]
    )

    removed = set()
    rows = []
    for position, ticker in enumerate(ordered):
        if ticker in removed:
            continue
        for other in ordered[position + 1:]:
            if other in removed:
                continue
            pair_correlation = correlation.at[ticker, other]
            if pd.notna(pair_correlation) and pair_correlation >= correlation_threshold:
                if str(other).upper() in owned:
                    continue
                removed.add(other)
                rows.append({
                    "Dropped": other,
                    "Kept Instead": ticker,
                    "Correlation": float(pair_correlation),
                })

    kept = [ticker for ticker in tickers if ticker not in removed]
    if len(kept) < 2:
        return tickers, empty_report

    report = pd.DataFrame(rows, columns=["Dropped", "Kept Instead", "Correlation"])
    if not report.empty:
        report = report.sort_values("Correlation", ascending=False).reset_index(drop=True)
    return kept, report


def get_daily_log_returns(
    symbols,
    start_date=None,
    end_date=None,
    buffer_days=0,
    drop_bottom_pct=0.2,
    ticker_currency_pairs=(),
    redundancy_corr_threshold=0.80,
    owned_tickers=(),
    progress_callback=None,
):
    pipeline_started = time.perf_counter()
    stage_timings = {}
    _, effective_end = _resolve_history_window_end(end_date, buffer_days)

    if start_date is None:
        start_date = DEFAULT_HISTORY_START_DATE

    requested_tickers = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]

    # Resolve the much smaller set of required FX histories before downloading
    # thousands of asset histories. Successful series live in the persistent
    # DuckDB cache, so a temporary Yahoo FX outage cannot discard a long run.
    fx_histories = {}
    foreign_currencies = sorted({
        _normalize_currency_code(currency)
        for _, currency in ticker_currency_pairs
        if _normalize_currency_code(currency) != "INR"
    })
    if foreign_currencies and progress_callback is not None:
        progress_callback(
            "Preparing persistent FX histories",
            0.02,
            {"currencies": int(len(foreign_currencies))},
        )
    for currency in foreign_currencies:
        fx_histories[currency] = download_fx_history_to_inr(
            currency, start_date, effective_end
        )

    history_started = time.perf_counter()
    df, volume_df, history_cache = sync_market_history(
        MARKET_HISTORY_CACHE_PATH,
        tuple(symbols),
        start_date,
        effective_end,
        _fetch_combined_daily_history,
        batch_size=FULL_HISTORY_BATCH_SIZE,
        retry_after_hours=20,
        on_progress=progress_callback,
    )
    df = df.copy()
    volume_df = volume_df.copy()
    stage_timings["history_cache"] = time.perf_counter() - history_started

    if progress_callback is not None:
        progress_callback(
            "Checking price integrity",
            0.80,
            {"symbols": int(len(df.columns))},
        )
    integrity_started = time.perf_counter()
    df, price_integrity_report = apply_price_integrity_gate(
        df, owned_tickers=owned_tickers
    )
    stage_timings["price_integrity"] = time.perf_counter() - integrity_started

    price_integrity_report = pd.DataFrame(price_integrity_report)

    if df.empty:
        raise ValueError("No data available for the given tickers.")
    available_tickers = {str(col).strip().upper() for col in df.columns}
    missing_history_tickers = sorted(set(requested_tickers) - available_tickers)

    # Convert foreign-market price histories into INR before calculating returns.
    filter_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback(
            "Converting currencies and applying quality filters",
            0.84,
            {"symbols": int(len(df.columns))},
        )
    if ticker_currency_pairs:
        df = convert_price_history_to_inr(
            df,
            ticker_currency_pairs,
            fx_histories=fx_histories,
        )

    lengths = df.count().sort_values(ascending=False)
    kept, dropped, num_to_drop = select_drop_bottom_tickers(
        df,
        drop_bottom_pct=drop_bottom_pct,
        volume_history=volume_df,
    )

    dropped_df = pd.DataFrame()
    if num_to_drop > 0:
        df = df[kept.index]
        dropped_df = pd.DataFrame({
            "Ticker": dropped.index,
            "Valid Days of Data": dropped.values,
        }).reset_index(drop=True)
    else:
        kept = lengths

    if len(kept) == 0:
        raise ValueError("History filtering removed every ticker.")

    momentum_kept, momentum_report_df = apply_robust_momentum_filter(
        df[kept.index],
        owned_tickers=owned_tickers,
        config=MOMENTUM_FILTER_CONFIG,
    )
    momentum_dropped_df = momentum_report_df.loc[
        momentum_report_df["Excluded"],
        [
            "Ticker",
            "Stable Momentum Score",
            "3-1 INR Return",
            "6-1 INR Return",
            "12-1 INR Return",
            "Below 200-Session Trend",
            "Reason",
        ],
    ].reset_index(drop=True)
    if not momentum_kept:
        raise ValueError("The robust momentum filter removed every ticker.")
    df = df[momentum_kept]
    kept = df.count().sort_values(ascending=False, kind="mergesort")
    stage_timings["quality_and_momentum_filters"] = (
        time.perf_counter() - filter_started
    )

    if progress_callback is not None:
        progress_callback(
            "Aligning common history",
            0.90,
            {"surviving_symbols": int(len(kept))},
        )
    alignment_started = time.perf_counter()
    valid_start = df[kept.index].apply(lambda x: x.first_valid_index()).max()
    valid_end = df[kept.index].apply(lambda x: x.last_valid_index()).min()

    if valid_start is None or valid_end is None or valid_start >= valid_end:
        raise ValueError("No overlapping date range found across tickers after filtering.")

    df_aligned = df.loc[valid_start:valid_end].ffill().dropna(axis=0, how="any")
    log_returns = np.log(df_aligned / df_aligned.shift(1)).dropna()
    stage_timings["common_period_alignment"] = (
        time.perf_counter() - alignment_started
    )

    if progress_callback is not None:
        progress_callback(
            "Removing near-duplicate exposures",
            0.94,
            {"aligned_assets": int(log_returns.shape[1])},
        )
    redundancy_started = time.perf_counter()
    kept_tickers, redundant_df = select_redundant_tickers(
        log_returns,
        volume_df.mean() if not volume_df.empty else None,
        redundancy_corr_threshold,
        owned_tickers=owned_tickers,
    )
    log_returns = log_returns[kept_tickers]
    stage_timings["near_duplicate_filter"] = (
        time.perf_counter() - redundancy_started
    )

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
        "redundant_df": redundant_df,
        "momentum_report_df": momentum_report_df,
        "momentum_dropped_df": momentum_dropped_df,
        "momentum_filter_config": dict(MOMENTUM_FILTER_CONFIG),
        "price_integrity_report": price_integrity_report,
        "price_integrity_version": PRICE_INTEGRITY_VERSION,
        "market_history_cache": history_cache,
        "pipeline_timings_seconds": {
            key: round(float(value), 3) for key, value in stage_timings.items()
        },
        "pipeline_total_seconds": round(
            float(time.perf_counter() - pipeline_started), 3
        ),
    }
    return log_returns, meta

# Cap on any single asset's optimized weight. Sample-estimate mean-variance
# optimization concentrates into whichever names had the luckiest realized mean,
# so an explicit cap is what keeps the result diversified.
def shrunk_covariance(log_returns):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity.

    With roughly as many assets as trading days the sample covariance is
    near-singular, and optimizers amplify that estimation error. Shrinkage keeps
    the matrix well-conditioned without adding a scikit-learn dependency.
    """
    returns = np.asarray(log_returns, dtype=float)
    num_days, num_assets = returns.shape
    if num_days < 2 or num_assets == 0:
        return np.atleast_2d(np.cov(returns, rowvar=False))

    centered = returns - returns.mean(axis=0)
    sample = centered.T @ centered / num_days

    average_variance = np.trace(sample) / num_assets
    target = average_variance * np.eye(num_assets)

    dispersion = np.sum((sample - target) ** 2) / num_assets
    if dispersion <= 0:
        return sample

    squared = centered ** 2
    noise = (squared.T @ squared / num_days - sample ** 2).sum() / (num_assets * num_days)
    shrinkage = float(np.clip(noise / dispersion, 0.0, 1.0))

    return shrinkage * target + (1.0 - shrinkage) * sample


def screen_log_returns_for_exact_optimizer(
    log_returns,
    *,
    maximum_assets=DEFAULT_EXACT_OPTIMIZER_ASSET_CAP,
    owned_tickers=(),
):
    """Reduce solver dimensionality while preserving broad candidate discovery."""
    cap = max(int(maximum_assets), 1)
    columns = [str(column) for column in log_returns.columns]
    owned = {str(ticker).strip().upper() for ticker in owned_tickers}
    protected = [ticker for ticker in columns if ticker.upper() in owned]
    if len(columns) <= cap:
        return log_returns, {
            "eligible_assets": len(columns),
            "selected_assets": len(columns),
            "owned_assets_protected": len(protected),
            "maximum_assets": cap,
            "method": "all-assets-fit",
        }

    numeric = log_returns.apply(pd.to_numeric, errors="coerce")
    annual_mean = numeric.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = numeric.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    downside = numeric.clip(upper=0.0)
    annual_downside = np.sqrt(downside.pow(2).mean()) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    positive_fraction = (numeric > 0).mean()

    finite_means = annual_mean.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_means.empty:
        clipped_mean = annual_mean.fillna(-np.inf)
    else:
        clipped_mean = annual_mean.clip(
            lower=float(finite_means.quantile(0.05)),
            upper=float(finite_means.quantile(0.95)),
        )
    blended_risk = (
        annual_volatility.fillna(np.inf) * 0.5
        + annual_downside.fillna(np.inf) * 0.5
    ).clip(lower=0.05)
    score = (
        (clipped_mean - RISK_FREE_RATE_ANNUAL) / blended_risk
        + 0.10 * (positive_fraction - 0.5)
    ).replace([np.inf, -np.inf], np.nan)

    candidate_slots = max(cap - len(protected), 0)
    ranked_candidates = (
        pd.DataFrame({"Ticker": columns, "Score": score.reindex(columns).to_numpy()})
        .loc[lambda frame: ~frame["Ticker"].str.upper().isin(owned)]
        .sort_values(
            ["Score", "Ticker"],
            ascending=[False, True],
            na_position="last",
            kind="mergesort",
        )
    )
    selected_candidates = ranked_candidates.head(candidate_slots)["Ticker"].tolist()
    selected_set = set(protected) | set(selected_candidates)
    selected_columns = [ticker for ticker in columns if ticker in selected_set]

    return log_returns.loc[:, selected_columns], {
        "eligible_assets": len(columns),
        "selected_assets": len(selected_columns),
        "owned_assets_protected": len(protected),
        "maximum_assets": cap,
        "method": "winsorized-excess-return-to-blended-downside-v1",
    }


def weight_bounds(num_assets, tickers=(), frozen_weights=None):
    """Per-asset bounds, with quarantined owned holdings frozen in place."""
    cap = max(MAX_WEIGHT_PER_ASSET, 1.0 / num_assets) if num_assets else 1.0
    frozen = {str(key).upper(): float(value) for key, value in (frozen_weights or {}).items()}
    # ``log_returns.columns`` is a pandas Index. Its truth value is deliberately
    # undefined, so never use ``if tickers`` here; large runs only reached this
    # branch after completing their expensive history pipeline.
    names = list(tickers) if tickers is not None else []
    if not names:
        names = [str(index) for index in range(num_assets)]
    bounds = []
    for ticker in names:
        fixed = frozen.get(str(ticker).upper())
        if fixed is None:
            bounds.append((0.0, min(cap, 1.0)))
        else:
            fixed = float(np.clip(fixed, 0.0, 1.0))
            bounds.append((fixed, fixed))
    return tuple(bounds)


def feasible_initial_weights(bounds):
    """Construct a deterministic feasible starting point for bounded SLSQP."""
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    if lower.sum() > 1.0 + 1e-9 or upper.sum() < 1.0 - 1e-9:
        raise ValueError("Optimizer bounds cannot form a fully invested portfolio.")
    weights = lower.copy()
    remaining = max(1.0 - float(weights.sum()), 0.0)
    while remaining > 1e-12:
        headroom = upper - weights
        available = np.flatnonzero(headroom > 1e-12)
        if not len(available):
            break
        increment = remaining / len(available)
        additions = np.minimum(headroom[available], increment)
        weights[available] += additions
        used = float(additions.sum())
        if used <= 1e-15:
            break
        remaining -= used
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Unable to construct feasible initial optimizer weights.")
    return weights

def enforce_min_weight_postprocess(weights, min_weight=0.01, frozen_positions=None):
    """Set tiny weights to zero and renormalize the rest.

    This does NOT change the optimization problem itself; it only cleans up
    the final solution for reporting and execution.
    """
    w = np.asarray(weights, dtype=float).copy()

    frozen = {
        int(position): float(value)
        for position, value in (frozen_positions or {}).items()
    }
    for position, value in frozen.items():
        w[position] = value
    adjustable = np.ones(len(w), dtype=bool)
    if frozen:
        adjustable[list(frozen)] = False
    w[adjustable & (np.abs(w) < min_weight)] = 0.0

    fixed_total = float(sum(frozen.values()))
    adjustable_total = float(w[adjustable].sum())
    if fixed_total > 1.0 + 1e-9 or adjustable_total <= 0:
        # If everything got zeroed, just return the original weights unchanged
        # (or you could fall back to the raw optimizer solution).
        return weights
    w[adjustable] *= max(1.0 - fixed_total, 0.0) / adjustable_total
    return w

class OptimizerTimeBudgetExceeded(RuntimeError):
    """Raised from a SciPy callback when one exact solve exceeds its budget."""


def _optimizer_deadline_callback(timeout_seconds):
    if timeout_seconds is None or float(timeout_seconds) <= 0:
        return None
    started = time.monotonic()

    def enforce_deadline(_weights):
        if time.monotonic() - started >= float(timeout_seconds):
            raise OptimizerTimeBudgetExceeded(
                f"Exact optimizer exceeded its {float(timeout_seconds) / 60:.1f}-minute budget."
            )

    return enforce_deadline


def optimize_portfolio_max_return_given_daily_risk(
    log_returns, max_drawdown=0.1, frozen_weights=None, timeout_seconds=None
):
    from scipy.optimize import minimize
    mean_returns = log_returns.mean().to_numpy(dtype=float)
    cov_matrix = np.asarray(shrunk_covariance(log_returns), dtype=float)
    cumulative_asset_returns = np.cumsum(
        log_returns.to_numpy(dtype=float), axis=0
    )
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        risk_free_rate_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        port_return = np.dot(weights, mean_returns)
        excess_return = port_return - risk_free_rate_daily
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_volatility == 0:
            return np.inf
        return -excess_return / port_volatility

    def negative_sharpe_jac(weights):
        risk_free_rate_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        covariance_times_weights = cov_matrix @ weights
        variance = float(weights @ covariance_times_weights)
        volatility = np.sqrt(max(variance, 1e-24))
        excess_return = float(weights @ mean_returns) - risk_free_rate_daily
        return (
            -mean_returns / volatility
            + excess_return * covariance_times_weights / (volatility ** 3)
        )

    drawdown_cache = {"weights": None, "value": None, "jac": None}

    def drawdown_value_and_jac(weights):
        cached_weights = drawdown_cache["weights"]
        if cached_weights is not None and np.array_equal(weights, cached_weights):
            return drawdown_cache["value"], drawdown_cache["jac"]
        cumulative = cumulative_asset_returns @ weights
        running_peak = np.maximum.accumulate(cumulative)
        trough_index = int(np.argmax(running_peak - cumulative))
        peak_index = int(np.argmax(cumulative[: trough_index + 1]))
        value = float(cumulative[peak_index] - cumulative[trough_index])
        jac = (
            cumulative_asset_returns[peak_index]
            - cumulative_asset_returns[trough_index]
        )
        drawdown_cache.update(
            weights=np.asarray(weights, dtype=float).copy(),
            value=value,
            jac=jac,
        )
        return value, jac

    def portfolio_drawdown(weights):
        return drawdown_value_and_jac(weights)[0]

    def portfolio_drawdown_jac(weights):
        return -drawdown_value_and_jac(weights)[1]

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1,
            "jac": lambda x: np.ones_like(x),
        },
        {
            "type": "ineq",
            "fun": lambda x: max_drawdown - portfolio_drawdown(x),
            "jac": portfolio_drawdown_jac,
        },
    ]
    bounds = weight_bounds(num_assets, log_returns.columns, frozen_weights)
    initial = feasible_initial_weights(bounds)

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        jac=negative_sharpe_jac,
        bounds=bounds,
        constraints=constraints,
        callback=_optimizer_deadline_callback(timeout_seconds),
        options={"maxiter": 100, "ftol": 1e-6, "disp": False},
    )
    return result.x if result.success else None


def optimize_max_sharpe_ratio(
    log_returns, frozen_weights=None, timeout_seconds=None
):
    from scipy.optimize import minimize
    mean_returns = log_returns.mean().to_numpy(dtype=float)
    cov_matrix = np.asarray(shrunk_covariance(log_returns), dtype=float)
    num_assets = len(mean_returns)

    def negative_sharpe(weights):
        risk_free_rate = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        port_return = np.dot(weights, mean_returns)
        excess_return = port_return - risk_free_rate
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -excess_return / port_volatility if port_volatility != 0 else np.inf

    def negative_sharpe_jac(weights):
        risk_free_rate = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        covariance_times_weights = cov_matrix @ weights
        variance = float(weights @ covariance_times_weights)
        volatility = np.sqrt(max(variance, 1e-24))
        excess_return = float(weights @ mean_returns) - risk_free_rate
        return (
            -mean_returns / volatility
            + excess_return * covariance_times_weights / (volatility ** 3)
        )

    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1,
        "jac": lambda x: np.ones_like(x),
    }]
    bounds = weight_bounds(num_assets, log_returns.columns, frozen_weights)
    initial = feasible_initial_weights(bounds)

    result = minimize(
        negative_sharpe,
        initial,
        method="SLSQP",
        jac=negative_sharpe_jac,
        bounds=bounds,
        constraints=constraints,
        callback=_optimizer_deadline_callback(timeout_seconds),
        options={"maxiter": 100, "ftol": 1e-6, "disp": False},
    )
    return result.x if result.success else None


def optimize_portfolio_target_volatility(
    log_returns,
    target_volatility=0.1,
    frozen_weights=None,
    timeout_seconds=None,
):
    from scipy.optimize import minimize
    mean_returns = log_returns.mean().to_numpy(dtype=float)
    cov_matrix = np.asarray(shrunk_covariance(log_returns), dtype=float)
    num_assets = len(mean_returns)

    def objective(weights):
        return -np.dot(weights, mean_returns)

    def objective_jac(_weights):
        return -mean_returns

    def constraint_sum(weights):
        return np.sum(weights) - 1

    def constraint_sum_jac(weights):
        return np.ones_like(weights)

    def constraint_volatility(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return target_volatility - portfolio_vol

    def constraint_volatility_jac(weights):
        covariance_times_weights = cov_matrix @ weights
        volatility = np.sqrt(max(float(weights @ covariance_times_weights), 1e-24))
        return -covariance_times_weights / volatility

    constraints = [
        {"type": "eq", "fun": constraint_sum, "jac": constraint_sum_jac},
        {
            "type": "ineq",
            "fun": constraint_volatility,
            "jac": constraint_volatility_jac,
        },
    ]
    bounds = weight_bounds(num_assets, log_returns.columns, frozen_weights)
    initial = feasible_initial_weights(bounds)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=objective_jac,
        bounds=bounds,
        constraints=constraints,
        callback=_optimizer_deadline_callback(timeout_seconds),
        options={"maxiter": 100, "ftol": 1e-6, "disp": False},
    )
    return result.x if result.success else None


def portfolio_stats(weights, log_returns):
    portfolio_returns = log_returns @ weights
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    annualized_return = mean * TRADING_DAYS_PER_YEAR
    annualized_vol = std * np.sqrt(TRADING_DAYS_PER_YEAR)
    simple_returns = np.expm1(np.asarray(portfolio_returns, dtype=float))
    expected_shortfall_1 = empirical_expected_shortfall(
        simple_returns,
        confidence=0.95,
    )
    bootstrap_scenarios_20 = moving_block_bootstrap_return_distribution(
        np.asarray(portfolio_returns, dtype=float),
        horizon_sessions=20,
    )
    bootstrap_summary_20 = bootstrap_tail_summary(
        bootstrap_scenarios_20,
        confidence=0.95,
    )
    expected_shortfall_20 = empirical_expected_shortfall(
        bootstrap_scenarios_20,
        confidence=0.95,
    )

    excess_return = annualized_return - RISK_FREE_RATE_ANNUAL
    sharpe = excess_return / annualized_vol if annualized_vol != 0 else 0

    return {
        "Annual Return": annualized_return,
        "Annual Volatility": annualized_vol,
        "Historical ES 95% (1 Session)": expected_shortfall_1,
        "Block-Bootstrap ES 95% (20 Sessions)": expected_shortfall_20,
        "Block-Bootstrap Median Return (20 Sessions)": bootstrap_summary_20[
            "median_return"
        ],
        "Block-Bootstrap Tail Loss SD 95% (20 Sessions)": bootstrap_summary_20[
            "tail_loss_sd"
        ],
        "Sharpe Ratio": sharpe,
    }


def portfolio_stats_comparison(current_alloc, log_returns, optimal_weights):
    allocation_weights = (
        current_alloc.assign(
            _weight=pd.to_numeric(current_alloc["Weight"], errors="coerce").fillna(0.0)
        )
        .groupby("Yahoo Ticker", sort=False)["_weight"]
        .sum()
    )
    aligned_tickers = [
        ticker
        for ticker in log_returns.columns
        if ticker in allocation_weights.index and allocation_weights[ticker] > 0
    ]
    current_stats = None
    if aligned_tickers:
        current_weights = allocation_weights.reindex(aligned_tickers).to_numpy(dtype=float)
        current_weight_total = float(current_weights.sum())
        if current_weight_total > 0:
            current_weights = current_weights / current_weight_total
            current_log_returns = log_returns.loc[:, aligned_tickers]
            current_stats = portfolio_stats(current_weights, current_log_returns)

    optimal_stats = portfolio_stats(optimal_weights, log_returns)
    return current_stats, optimal_stats


def optimize_prepared_return_matrix(
    prepared_log_returns,
    current_alloc,
    *,
    maximum_assets,
    max_dd=0.05,
    target_volatility=None,
    integrity_report=None,
    timeout_seconds=None,
):
    """Solve one exact-asset limit without rebuilding market history.

    The prepared matrix must be the matrix produced for the largest exact-
    optimizer limit in the current shortlist-cap evaluation.  Screening that
    matrix again for a smaller limit is equivalent to screening the original
    ranked universe, while avoiding repeated downloads, FX conversion,
    continuity checks, common-period alignment and duplicate detection.
    """
    owned_tickers = tuple(
        sorted(
            str(row["Yahoo Ticker"])
            for _, row in current_alloc.iterrows()
            if str(row.get("Yahoo Ticker", "")).strip()
            and pd.notna(pd.to_numeric(row.get("Quantity", 0), errors="coerce"))
            and float(pd.to_numeric(row.get("Quantity", 0), errors="coerce")) > 0
        )
    )
    log_returns, solver_screen = screen_log_returns_for_exact_optimizer(
        prepared_log_returns,
        maximum_assets=int(maximum_assets),
        owned_tickers=owned_tickers,
    )

    quarantined_owned = set()
    if isinstance(integrity_report, pd.DataFrame) and not integrity_report.empty:
        quarantined_owned = set(
            integrity_report.loc[
                integrity_report["Owned"].astype(bool)
                & integrity_report["Status"].astype(str).str.startswith("QUARANTINED_"),
                "Ticker",
            ].astype(str)
        )
    allocation_weights = (
        current_alloc.assign(
            _weight=pd.to_numeric(current_alloc["Weight"], errors="coerce").fillna(0.0)
        )
        .groupby("Yahoo Ticker", sort=False)["_weight"]
        .sum()
    )
    frozen_weights = {
        ticker: float(allocation_weights.get(ticker, 0.0))
        for ticker in log_returns.columns
        if ticker in quarantined_owned
        and float(allocation_weights.get(ticker, 0.0)) > 0.0
    }

    try:
        if target_volatility is not None:
            optimal_weights = optimize_portfolio_target_volatility(
                log_returns,
                target_volatility=target_volatility,
                frozen_weights=frozen_weights,
                timeout_seconds=timeout_seconds,
            )
        else:
            optimal_weights = None
        if optimal_weights is None:
            optimal_weights = optimize_portfolio_max_return_given_daily_risk(
                log_returns,
                max_drawdown=max_dd,
                frozen_weights=frozen_weights,
                timeout_seconds=timeout_seconds,
            )
        if optimal_weights is None:
            optimal_weights = optimize_max_sharpe_ratio(
                log_returns,
                frozen_weights=frozen_weights,
                timeout_seconds=timeout_seconds,
            )
    except OptimizerTimeBudgetExceeded:
        solver_screen = dict(solver_screen or {})
        solver_screen["timed_out"] = True
        return None, log_returns, None, None, solver_screen
    if optimal_weights is None:
        return None, log_returns, None, None, solver_screen

    frozen_positions = {
        int(log_returns.columns.get_loc(ticker)): weight
        for ticker, weight in frozen_weights.items()
    }
    optimal_weights = enforce_min_weight_postprocess(
        optimal_weights,
        min_weight=0.01,
        frozen_positions=frozen_positions,
    )
    current_stats, optimal_stats = portfolio_stats_comparison(
        current_alloc,
        log_returns,
        optimal_weights,
    )
    return optimal_weights, log_returns, current_stats, optimal_stats, solver_screen


def solve_prepared_search_finalist(
    prepared,
    current_alloc,
    *,
    maximum_assets,
    max_dd,
    target_volatility,
):
    """Run the selected final solve against the matrix already built by search.

    This avoids re-running candidate preselection, market-history preparation,
    FX conversion and quality filters in the automatic search-to-plan handoff.
    """
    if not prepared or len(prepared) < 3:
        raise ValueError("The selected search finalist has no reusable history matrix.")
    extended_df, prepared_log_returns, prepared_meta = prepared
    if prepared_log_returns is None or prepared_log_returns.empty:
        raise ValueError("The selected search finalist has an empty history matrix.")

    started = time.perf_counter()
    weights, selected_returns, current_stats, optimal_stats, solver_screen = (
        optimize_prepared_return_matrix(
            prepared_log_returns,
            extended_df.copy(),
            maximum_assets=int(maximum_assets),
            max_dd=max_dd,
            target_volatility=target_volatility,
            integrity_report=(prepared_meta or {}).get("price_integrity_report"),
        )
    )
    if weights is None:
        return None, selected_returns, current_stats, optimal_stats, dict(prepared_meta or {})

    meta = dict(prepared_meta or {})
    meta["exact_optimizer_screen"] = solver_screen or {}
    timings = dict(meta.get("pipeline_timings_seconds") or {})
    timings["final_optimizer_solver"] = round(time.perf_counter() - started, 3)
    meta["pipeline_timings_seconds"] = timings
    meta["analysis_total_seconds"] = round(float(sum(timings.values())), 3)
    owned_tickers = {
        str(row.get("Yahoo Ticker", "")).strip().upper()
        for _, row in current_alloc.iterrows()
        if str(row.get("Yahoo Ticker", "")).strip()
        and pd.notna(pd.to_numeric(row.get("Quantity", 0), errors="coerce"))
        and float(pd.to_numeric(row.get("Quantity", 0), errors="coerce")) > 0
    }
    selected_tickers = {str(ticker).strip().upper() for ticker in selected_returns.columns}
    meta["missing_owned_return_tickers"] = sorted(owned_tickers - selected_tickers)
    return weights, selected_returns, current_stats, optimal_stats, meta


def validate_search_finalists_walk_forward(
    owner,
    search_rows,
    *,
    max_dd,
    target_volatility,
    drop_bottom_pct,
    history_buffer_days,
    redundancy_corr_threshold,
    transaction_cost_bps=22.0,
    finalist_count=2,
    prepared_matrices=None,
    minimum_train_sessions=252,
    minimum_test_sessions=42,
    solver_timeout_seconds=60,
    total_timeout_seconds=900,
    progress_callback=None,
):
    """Check leading full-history finalists on expanding chronological windows.

    This is a finalist stability gate, not a fully point-in-time backtest: the
    current candidate preselection and exact-optimizer screen are reused for
    every fold. The gate deliberately reports that limitation to callers.
    """
    started = time.monotonic()
    available_matrices = prepared_matrices if isinstance(prepared_matrices, dict) else None
    all_feasible = [
        dict(row) for row in search_rows
        if row.get("Status") == "Feasible"
        and np.isfinite(float(row.get("Annual Return", np.nan)))
    ]
    feasible = list(all_feasible)
    feasible.sort(key=lambda row: float(row["Annual Return"]), reverse=True)
    finalists = []
    seen_pairs = set()
    for row in feasible:
        pair = (int(row["Shortlist cap"]), int(row["Maximum assets"]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        finalists.append(row)
        if len(finalists) >= max(int(finalist_count), 1):
            break
    if not finalists:
        return {
            "status": "no_feasible_finalists",
            "finalists": [],
            "winner": None,
            "pairs_without_reusable_matrix": 0,
        }

    invalid_rows = []
    if available_matrices is not None:
        # Search hands over already-prepared, exact-universe matrices. In
        # particular, do not redownload an 8k-name history universe after the
        # user has already waited for the cap search to finish.
        prepared_by_cap = {
            int(cap): item for cap, item in available_matrices.items()
            if item and item[1] is not None and not item[1].empty
        }
    else:
        prepared_by_cap = {}

    # A process restart discards the in-memory price matrices while retaining
    # completed search rows. Rebuild only finalist caps missing from that cache;
    # otherwise a resumed search could silently validate a lower-ranked cached
    # pair instead of the actual top-return finalists.
    finalists_by_cap = {}
    for row in finalists:
        finalists_by_cap.setdefault(int(row["Shortlist cap"]), []).append(row)
    missing_caps = [cap for cap in finalists_by_cap if cap not in prepared_by_cap]
    if missing_caps:
        portfolio_df, invalid_rows = build_current_allocation_from_db(owner)
        if portfolio_df.empty:
            raise ValueError("No usable holdings are available for walk-forward validation.")
        for cap_index, cap in enumerate(missing_caps, start=1):
            if time.monotonic() - started >= float(total_timeout_seconds):
                break
            if progress_callback:
                progress_callback(
                    f"Preparing finalist universe {cap_index}/{len(missing_caps)} (cap {cap:,})",
                    0.05 + 0.15 * (cap_index - 1) / max(len(missing_caps), 1),
                )
            extended_df, _, _ = extend_allocation_with_universal_candidates(
                portfolio_df.copy(), maximum_candidates=cap, owner=owner
            )
            tickers = extended_df["Yahoo Ticker"].dropna().astype(str).tolist()
            max_assets = max(
                int(row["Maximum assets"]) for row in finalists_by_cap[cap]
            )
            _, returns, _, _, meta = run_portfolio_analysis_multi(
                tickers,
                extended_df.copy(),
                max_dd=max_dd,
                target_volatility=target_volatility,
                drop_bottom_pct=drop_bottom_pct,
                buffer_days=history_buffer_days,
                redundancy_corr_threshold=redundancy_corr_threshold,
                exact_optimizer_asset_cap=max_assets,
                solve_portfolio=False,
                optimizer_timeout_seconds=solver_timeout_seconds,
            )
            if returns is not None and not returns.empty:
                prepared_by_cap[cap] = (extended_df, returns, meta)
                if available_matrices is not None:
                    available_matrices[cap] = prepared_by_cap[cap]

    finalist_pairs = {
        (int(row["Shortlist cap"]), int(row["Maximum assets"]))
        for row in finalists
    }
    reusable_pairs = {
        (int(row["Shortlist cap"]), int(row["Maximum assets"]))
        for row in finalists
        if int(row["Shortlist cap"]) in prepared_by_cap
    }

    result_rows = []
    attempted = 0
    total_attempts = len(finalists) * 3
    cost_rate = max(float(transaction_cost_bps), 0.0) / 10000.0
    for finalist_index, finalist in enumerate(finalists, start=1):
        cap = int(finalist["Shortlist cap"])
        asset_limit = int(finalist["Maximum assets"])
        prepared = prepared_by_cap.get(cap)
        record = {
            "Shortlist cap": cap,
            "Maximum assets": asset_limit,
            "Full-history annual return": float(finalist["Annual Return"]),
            "Fold net annual returns": [],
            "Positive folds": 0,
            "Status": "Not evaluated",
        }
        if prepared is None:
            record["Status"] = "History preparation unavailable or timed out"
            result_rows.append(record)
            continue
        extended_df, all_returns, meta = prepared
        n_sessions = int(all_returns.shape[0])
        block = max(int(minimum_test_sessions), n_sessions // 5)
        if n_sessions < int(minimum_train_sessions) + 3 * int(minimum_test_sessions):
            record["Status"] = f"Insufficient history ({n_sessions} sessions)"
            result_rows.append(record)
            continue
        block = min(block, (n_sessions - int(minimum_train_sessions)) // 3)
        fold_ranges = [
            (n_sessions - 2 * block, n_sessions - block),
            (n_sessions - block, n_sessions),
        ]
        # First fold is an earlier holdout; later fold expands the training set.
        fold_ranges.insert(0, (n_sessions - 3 * block, n_sessions - 2 * block))
        fold_values = []
        failure = None
        for fold_index, (test_start, test_end) in enumerate(fold_ranges, start=1):
            attempted += 1
            if time.monotonic() - started >= float(total_timeout_seconds):
                failure = "Validation runtime budget reached"
                break
            if test_start < int(minimum_train_sessions) or test_end - test_start < int(minimum_test_sessions):
                failure = "Insufficient chronological fold coverage"
                break
            if progress_callback:
                progress_callback(
                    f"Validating finalist {finalist_index}/{len(finalists)} · fold {fold_index}/3",
                    0.20 + 0.78 * attempted / max(total_attempts, 1),
                )
            train_returns = all_returns.iloc[:test_start]
            weights, selected_train, _, _, screen = optimize_prepared_return_matrix(
                train_returns,
                extended_df.copy(),
                maximum_assets=asset_limit,
                max_dd=max_dd,
                target_volatility=target_volatility,
                integrity_report=(meta or {}).get("price_integrity_report"),
                timeout_seconds=solver_timeout_seconds,
            )
            if weights is None or selected_train is None or selected_train.empty:
                failure = "Training optimizer infeasible or timed out"
                break
            test_returns = all_returns.iloc[test_start:test_end].reindex(
                columns=selected_train.columns
            )
            test_returns = test_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            if len(test_returns) < int(minimum_test_sessions):
                failure = "Too few complete test sessions after ticker alignment"
                break
            weights = np.asarray(weights, dtype=float)
            gross_daily = test_returns.to_numpy(dtype=float) @ weights
            current_weights = (
                extended_df.assign(
                    _weight=pd.to_numeric(extended_df["Weight"], errors="coerce").fillna(0.0)
                )
                .groupby("Yahoo Ticker", sort=False)["_weight"].sum()
            )
            current_total = float(current_weights.clip(lower=0.0).sum())
            if current_total > 0:
                current_weights = current_weights.clip(lower=0.0) / current_total
            current_map = current_weights.to_dict()
            optimal_map = dict(zip(selected_train.columns, weights.tolist()))
            turnover_tickers = set(current_map) | set(optimal_map)
            one_way_turnover = 0.5 * sum(
                abs(float(optimal_map.get(ticker, 0.0)) - float(current_map.get(ticker, 0.0)))
                for ticker in turnover_tickers
            )
            cost_drag_daily = (
                np.log1p(-min(cost_rate * one_way_turnover, 0.999999))
                / len(test_returns)
            )
            net_annual = float((gross_daily.mean() + cost_drag_daily) * TRADING_DAYS_PER_YEAR)
            fold_values.append(net_annual)
        record["Fold net annual returns"] = fold_values
        record["Positive folds"] = int(sum(value > 0 for value in fold_values))
        record["Median net annual return"] = (
            float(np.median(fold_values)) if fold_values else np.nan
        )
        record["Worst fold net annual return"] = (
            float(min(fold_values)) if fold_values else np.nan
        )
        record["Status"] = failure or (
            "Passed" if len(fold_values) == 3 and record["Positive folds"] >= 2
            else "Did not pass consistency gate"
        )
        result_rows.append(record)

    passed = [row for row in result_rows if row["Status"] == "Passed"]
    winner = max(
        passed,
        key=lambda row: float(row["Median net annual return"]),
        default=None,
    )
    return {
        "status": "passed" if winner else "no_finalist_passed",
        "finalists": result_rows,
        "winner": winner,
        "transaction_cost_bps": float(transaction_cost_bps),
        "elapsed_seconds": float(time.monotonic() - started),
        "invalid_holdings": invalid_rows,
        "pairs_without_reusable_matrix": len(finalist_pairs - reusable_pairs),
        "method_note": (
            "Expanding chronological folds; weights fit on prior sessions only; "
            "estimated one-way-turnover cost applied to each test fold. Finalists "
            "were nominated by full-history search results and the current candidate "
            "universe is reused, so this is a finalist robustness gate—not an "
            "unbiased point-in-time backtest."
        ),
    }

def run_portfolio_analysis_multi(
    symbols,
    current_alloc,
    max_dd=0.05,
    target_volatility=None,
    drop_bottom_pct=0.2,
    buffer_days=0,
    redundancy_corr_threshold=0.80,
    exact_optimizer_asset_cap=300,
    progress_callback=None,
    solve_portfolio=True,
    optimizer_timeout_seconds=None,
):
    ticker_currency_pairs = tuple(
        sorted(
            (str(row["Yahoo Ticker"]), _normalize_currency_code(row["Currency"]))
            for _, row in current_alloc.iterrows()
            if str(row.get("Yahoo Ticker", "")).strip()
        )
    )
    owned_tickers = tuple(
        sorted(
            str(row["Yahoo Ticker"])
            for _, row in current_alloc.iterrows()
            if str(row.get("Yahoo Ticker", "")).strip()
            and pd.notna(pd.to_numeric(row.get("Quantity", 0), errors="coerce"))
            and float(pd.to_numeric(row.get("Quantity", 0), errors="coerce")) > 0
        )
    )
    log_returns, meta = get_daily_log_returns(
        tuple(symbols),
        drop_bottom_pct=drop_bottom_pct,
        buffer_days=buffer_days,
        ticker_currency_pairs=ticker_currency_pairs,
        redundancy_corr_threshold=redundancy_corr_threshold,
        owned_tickers=owned_tickers,
        progress_callback=progress_callback,
    )
    screen_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback(
            "Selecting the exact optimizer universe",
            0.96,
            {"eligible_assets": int(log_returns.shape[1])},
        )
    log_returns, solver_screen = screen_log_returns_for_exact_optimizer(
        log_returns,
        maximum_assets=exact_optimizer_asset_cap,
        owned_tickers=owned_tickers,
    )
    meta.setdefault("pipeline_timings_seconds", {})[
        "exact_universe_screen"
    ] = round(float(time.perf_counter() - screen_started), 3)
    meta["exact_optimizer_screen"] = solver_screen
    integrity_report = meta.get("price_integrity_report")
    quarantined_owned = set()
    if isinstance(integrity_report, pd.DataFrame) and not integrity_report.empty:
        quarantined_owned = set(
            integrity_report.loc[
                integrity_report["Owned"].astype(bool)
                & integrity_report["Status"].astype(str).str.startswith("QUARANTINED_"),
                "Ticker",
            ].astype(str)
        )
    allocation_weights = (
        current_alloc.assign(
            _weight=pd.to_numeric(current_alloc["Weight"], errors="coerce").fillna(0.0)
        )
        .groupby("Yahoo Ticker", sort=False)["_weight"]
        .sum()
    )
    frozen_weights = {
        ticker: float(allocation_weights.get(ticker, 0.0))
        for ticker in log_returns.columns
        if ticker in quarantined_owned and float(allocation_weights.get(ticker, 0.0)) > 0.0
    }
    meta["price_integrity_frozen_owned_weights"] = dict(frozen_weights)

    if not solve_portfolio:
        meta["analysis_total_seconds"] = round(
            float(sum(meta.get("pipeline_timings_seconds", {}).values())), 3
        )
        meta["missing_owned_return_tickers"] = sorted(
            set(owned_tickers)
            - {str(ticker).strip().upper() for ticker in log_returns.columns}
        )
        return None, log_returns, None, None, meta

    if progress_callback is not None:
        progress_callback(
            "Solving the constrained portfolio",
            0.98,
            {"assets": int(log_returns.shape[1])},
        )
    solver_started = time.perf_counter()
    try:
        if target_volatility is not None:
            optimal_weights = optimize_portfolio_target_volatility(
                log_returns,
                target_volatility=target_volatility,
                frozen_weights=frozen_weights,
                timeout_seconds=optimizer_timeout_seconds,
            )
        else:
            optimal_weights = None

        if optimal_weights is None:
            optimal_weights = optimize_portfolio_max_return_given_daily_risk(
                log_returns,
                max_drawdown=max_dd,
                frozen_weights=frozen_weights,
                timeout_seconds=optimizer_timeout_seconds,
            )
        if optimal_weights is None:
            optimal_weights = optimize_max_sharpe_ratio(
                log_returns,
                frozen_weights=frozen_weights,
                timeout_seconds=optimizer_timeout_seconds,
            )
    except OptimizerTimeBudgetExceeded as exc:
        meta["optimizer_timed_out"] = True
        meta["optimizer_timeout_reason"] = str(exc)
        return None, log_returns, None, None, meta
    if optimal_weights is None:
        return None, log_returns, None, None, meta
    meta.setdefault("pipeline_timings_seconds", {})["optimizer_solver"] = round(
        float(time.perf_counter() - solver_started), 3
    )

    # Enforce minimum 1% weight on any non-zero position, then renormalize
    frozen_positions = {
        int(log_returns.columns.get_loc(ticker)): weight
        for ticker, weight in frozen_weights.items()
    }
    optimal_weights = enforce_min_weight_postprocess(
        optimal_weights,
        min_weight=0.01,
        frozen_positions=frozen_positions,
    )

    current_stats, optimal_stats = portfolio_stats_comparison(
        current_alloc, log_returns, optimal_weights
    )
    meta["analysis_total_seconds"] = round(
        float(sum(meta.get("pipeline_timings_seconds", {}).values())), 3
    )
    meta["missing_owned_return_tickers"] = sorted(
        set(owned_tickers) - {str(ticker).strip().upper() for ticker in log_returns.columns}
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
    allocation_result = allocate_public_lumpsum(
        [{"ticker":ticker,"target_weight":float(weight)} for ticker,weight in zip(common_tickers,aligned_weights)],
        {ticker:float(price) for ticker,price in zip(common_tickers,price_inr)},
        float(lumpsum_inr),max_weight_per_asset=MAX_WEIGHT_PER_ASSET,
    )
    order_by_ticker={row["ticker"]:row for row in allocation_result["orders"]}
    target_amounts = float(lumpsum_inr) * aligned_weights
    quantities=np.array([order_by_ticker.get(ticker,{}).get("quantity",0) for ticker in common_tickers],dtype=int)
    invested_amounts=quantities*price_inr

    plan = pd.DataFrame({
    "Symbol": alloc_df.loc[common_tickers, "Symbol"].values,
    "Yahoo Ticker": common_tickers,
    "Optimal Weight": aligned_weights,
    "Target Amount INR": target_amounts,
    "Latest Price INR": price_inr,
    "Suggested Quantity": quantities,
    "Estimated Investment INR": invested_amounts,
    })
    
    # Keep only rows with non-zero optimal weight
    plan = plan[plan["Optimal Weight"] > 0].sort_values(
        "Optimal Weight", ascending=False
    ).reset_index(drop=True)

    unallocated_cash = float(allocation_result["residual_cash_inr"])
    return plan, unallocated_cash, missing_prices, missing_alloc


def lumpsum_execution_sheet_html(lumpsum_df, lumpsum_inr, unallocated_cash):
    """Build a compact shareable buy-order sheet without portfolio analysis details."""
    orders = lumpsum_df[lumpsum_df["Suggested Quantity"] > 0].copy()
    rows = "".join(
        "<tr>"
        f"<td data-label=\"Symbol\">{html.escape(str(row['Symbol']))}</td>"
        f"<td data-label=\"Buy Quantity\">{int(row['Suggested Quantity']):,}</td>"
        f"<td data-label=\"Estimated Value\">INR {float(row['Estimated Investment INR']):,.2f}</td>"
        "</tr>"
        for _, row in orders.iterrows()
    )
    if not rows:
        rows = '<tr><td colspan="3">No whole-share buy orders for this amount.</td></tr>'

    estimated_investment = float(orders["Estimated Investment INR"].sum())
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lumpsum Buy Orders</title>
<style>
body {{ font-family: Arial, sans-serif; color: #17212b; margin: 20px auto; max-width: 760px; padding: 0 16px; font-size: 14px; }}
h1 {{ margin: 0; font-size: 22px; }}
.generated {{ margin: 2px 0 10px; color: #5d6872; font-size: 12px; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 10px 0; padding: 9px 12px; background: #f2f6f4; border-left: 4px solid #1f7a5b; }}
.summary p {{ margin: 0; white-space: nowrap; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d7dce0; padding: 7px 8px; text-align: left; }}
th {{ background: #17212b; color: white; }}
.footnote {{ margin: 10px 0 0; color: #5d6872; font-size: 11px; }}
@media (max-width: 600px) {{
body {{ margin: 12px auto; padding: 0 10px; font-size: 12px; }}
h1 {{ font-size: 19px; }}
.summary {{ gap: 4px 10px; margin: 8px 0; padding: 7px 9px; }}
.summary p {{ white-space: normal; }}
th, td {{ padding: 6px 4px; }}
}}
@media print {{ body {{ margin: 12px; max-width: none; }} .summary {{ margin: 8px 0; }} th, td {{ padding: 5px 6px; }} }}
</style>
</head>
<body>
<h1>Lumpsum Buy Orders</h1>
<p class="generated">Generated {generated_at}</p>
<div class="summary">
<p><strong>Lumpsum:</strong> INR {float(lumpsum_inr):,.2f}</p>
<p><strong>Estimated order value:</strong> INR {estimated_investment:,.2f}</p>
<p><strong>Cash remaining:</strong> INR {float(unallocated_cash):,.2f}</p>
</div>
<table>
<thead><tr><th>Symbol</th><th>Buy Quantity</th><th>Estimated Value</th></tr></thead>
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
    annualized_mean = log_returns[common_tickers].mean().to_numpy() * TRADING_DAYS_PER_YEAR
    annualized_vol = log_returns[common_tickers].std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS_PER_YEAR)
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

    # Turnover-control policy: keep buys and complete exits, but do not generate
    # partial sell orders for securities that remain in the target portfolio.
    # Their non-zero optimizer weight is still published and the next analysis
    # can reassess them without realizing costs merely to trim the position.
    partial_sell = (
        rebal_df["Action"].eq("Sell")
        & rebal_df["Optimal Weight"].gt(1e-12)
    )
    rebal_df = (
        rebal_df[(rebal_df["Executable Quantity"] != 0) & ~partial_sell]
        .sort_values(by="Expected Annual Return Lift (%)", ascending=False)
        .reset_index(drop=True)
    )
    return rebal_df, missing_prices, missing_alloc


def build_holdings_action_summary(
    portfolio_df,
    rebal_df,
    optimal_weight_by_ticker,
    price_map,
    meta,
    invalid_holding_rows=(),
    missing_price_tickers=(),
    deferred_trade_expiry=None,
):
    """Report one row per owned holding, including the ones no trade was produced for.

    The rebalancing plan lists executable trades only, so a holding removed by a
    filter looks exactly like one that needs no action. This states which it was.
    """
    columns = [
        "Symbol", "Action", "Quantity", "Price INR", "Trade Value INR",
        "Current Weight", "Optimal Weight", "Status",
    ]

    quantity = pd.to_numeric(portfolio_df.get("Quantity"), errors="coerce").fillna(0)
    owned = portfolio_df[quantity > 0].copy()
    if owned.empty:
        return pd.DataFrame(columns=columns)

    trades = {}
    if rebal_df is not None and not rebal_df.empty:
        trades = {str(row["Yahoo Ticker"]): row for _, row in rebal_df.iterrows()}

    redundant_df = meta.get("redundant_df")
    merged_into = (
        dict(zip(redundant_df["Dropped"].astype(str), redundant_df["Kept Instead"].astype(str)))
        if redundant_df is not None and not redundant_df.empty
        else {}
    )

    dropped_df = meta.get("dropped_df")
    short_history = (
        set(dropped_df["Ticker"].astype(str))
        if dropped_df is not None and not dropped_df.empty
        else set()
    )

    missing_history = {str(t) for t in (meta.get("missing_history_tickers") or [])}
    missing_price = {str(t) for t in (missing_price_tickers or [])}
    invalid_symbols = {str(s) for s in (invalid_holding_rows or [])}
    analysed = {str(t) for t in optimal_weight_by_ticker}
    deferred_trade_expiry = {
        str(ticker).upper(): str(expiry)
        for ticker, expiry in (deferred_trade_expiry or {}).items()
    }

    rows = []
    for _, holding in owned.iterrows():
        ticker = str(holding["Yahoo Ticker"])
        symbol = str(holding["Symbol"])
        trade = trades.get(ticker)

        if trade is not None:
            rows.append({
                "Symbol": symbol,
                "Action": str(trade["Action"]),
                "Quantity": int(trade["Executable Quantity"]),
                "Price INR": float(trade["Native Price"]) * float(trade["FX to INR"]),
                "Trade Value INR": float(trade["Executable Value"]),
                "Current Weight": float(trade["Current Weight"]),
                "Optimal Weight": float(trade["Optimal Weight"]),
                "Status": "Listed in the rebalancing plan",
            })
            continue

        native_price = price_map.get(ticker)
        price_inr = (
            float(native_price) * float(holding["FX to INR"])
            if native_price is not None
            else float(holding["Latest Price INR"])
        )

        if ticker.upper() in deferred_trade_expiry:
            action = "Hold"
            optimal_weight = float(optimal_weight_by_ticker.get(ticker, 0.0))
            status = (
                "Trade paused; excluded from universal selection until "
                + deferred_trade_expiry[ticker.upper()]
            )
        elif ticker in analysed:
            action = "Hold"
            optimal_weight = float(optimal_weight_by_ticker.get(ticker, 0.0))
            current_weight = float(holding.get("Weight", 0.0))
            if optimal_weight > 1e-12 and current_weight > optimal_weight + 1e-12:
                status = "Partial sell suppressed to reduce trading; target weight remains non-zero"
            else:
                status = "Optimal weight is less than one whole share away"
        elif ticker in merged_into:
            action = "Not analysed"
            status = f"Near-duplicate of {merged_into[ticker]}, which carries this exposure"
        elif ticker in short_history:
            action = "Not analysed"
            status = "Removed by the liquidity / history-length filter"
        elif ticker in missing_history:
            action = "Not analysed"
            status = "Yahoo Finance returned no usable price history"
        elif ticker in missing_price:
            action = "Not analysed"
            status = "No latest price was available for sizing"
        elif symbol in invalid_symbols:
            action = "Not analysed"
            status = "Quantity or average price is missing or invalid"
        else:
            action = "Not analysed"
            status = "No overlapping history with the rest of the portfolio"

        rows.append({
            "Symbol": symbol,
            "Action": action,
            "Quantity": 0,
            "Price INR": price_inr,
            "Trade Value INR": 0.0,
            "Current Weight": float(holding.get("Weight", 0.0)),
            "Optimal Weight": float(optimal_weight_by_ticker.get(ticker, 0.0)),
            "Status": status,
        })

    action_order = {"Sell": 0, "Buy": 1, "Hold": 2, "Not analysed": 3}
    summary = pd.DataFrame(rows, columns=columns)
    summary["_order"] = summary["Action"].map(action_order).fillna(4)
    return (
        summary.sort_values(["_order", "Trade Value INR"], ascending=[True, False], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def style_holdings_action_summary(df):
    def color_action_row(row):
        colors = {
            "Buy": "background-color: #d4edda; color: #155724;",
            "Sell": "background-color: #f8d7da; color: #721c24;",
            "Hold": "background-color: #e2e3e5; color: #383d41;",
        }
        return [colors.get(row["Action"], "background-color: #fff3cd; color: #856404;")] * len(row)

    formatters = {
        "Quantity": "{:,.0f}",
        "Price INR": "₹{:,.2f}",
        "Trade Value INR": "₹{:,.0f}",
        "Current Weight": "{:.2%}",
        "Optimal Weight": "{:.2%}",
    }
    return df.style.apply(color_action_row, axis=1).format(formatters, na_rep="N/A")

@st.cache_data(show_spinner=False, ttl=300, max_entries=32)
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


def render_trade_deferral_editor(plan_df, owner, key_prefix):
    """Render executable rows with a persistent 90-day pause control."""
    plan_df = _sort_rebalance_df_for_priority(plan_df)
    with get_db_connection() as conn:
        deferrals = active_trade_deferrals(conn, owner)
    executable, deferred = split_deferred_trades(plan_df, deferrals)

    if not executable.empty:
        editable = executable.copy()
        editable.insert(0, "Skip for one quarter", False)
        disabled_columns = [
            column for column in editable.columns if column != "Skip for one quarter"
        ]
        with st.form(f"{key_prefix}_deferral_form", border=False):
            edited = st.data_editor(
                editable.style.apply(action_row_style, axis=1),
                key=f"{key_prefix}_deferral_editor",
                width="stretch",
                hide_index=True,
                disabled=disabled_columns,
                column_config={
                    "Skip for one quarter": st.column_config.CheckboxColumn(
                        "Skip 3 months",
                        help=(
                            "Excludes this security from the universal candidate pool "
                            "and suppresses its trade for 90 days."
                        ),
                        default=False,
                        pinned=True,
                    ),
                    "Current Weight": st.column_config.NumberColumn(format="percent"),
                    "Optimal Weight": st.column_config.NumberColumn(format="percent"),
                    "Change": st.column_config.NumberColumn(format="percent"),
                    "Expected Annual Return Lift (%)": st.column_config.NumberColumn(
                        format="%.2f"
                    ),
                    "Native Price": st.column_config.NumberColumn(format="%.2f"),
                    "FX to INR": st.column_config.NumberColumn(format="%.4f"),
                    "Executable Quantity": st.column_config.NumberColumn(format="%.0f"),
                    "Executable Value": st.column_config.NumberColumn(format="₹%.0f"),
                },
            )
            submitted = st.form_submit_button(
                "Pause selected securities for 90 days",
                icon=":material/pause_circle:",
            )
        if submitted:
            selected = edited.loc[edited["Skip for one quarter"].fillna(False)].drop(
                columns="Skip for one quarter"
            )
            if selected.empty:
                st.warning("Select at least one security to pause.")
            else:
                with get_db_connection() as conn:
                    saved = defer_trades_for_quarter(
                        conn,
                        owner,
                        selected.to_dict(orient="records"),
                    )
                st.success(
                    f"Paused {len(saved)} securities for 90 days. "
                    "They will be excluded automatically on the next optimization."
                )
                st.rerun()
    else:
        st.success("No currently executable trades remain after active pauses.")

    if not deferrals.empty:
        paused = deferrals.rename(
            columns={
                "symbol": "Symbol",
                "yahoo_ticker": "Yahoo Ticker",
                "requested_action": "Action when paused",
                "expires_at": "Excluded until",
            }
        )
        paused["Excluded until"] = pd.to_datetime(
            paused["Excluded until"], utc=True, errors="coerce"
        )
        st.caption(
            "Paused securities remain in your holdings records, but are excluded from "
            "universal selection and executable trades until expiry."
        )
        st.dataframe(
            paused[["Symbol", "Yahoo Ticker", "Action when paused", "Excluded until"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Excluded until": st.column_config.DatetimeColumn(
                    format="DD MMM YYYY, hh:mm a"
                )
            },
        )

    return executable, deferred

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

    volume_history, _ = _download_volume_history_resilient(
        list(yahoo_tickers),
        period="5y",
        batch_size=12,
    )
    volume_history = volume_history.dropna(axis=1, how="all")

    kept, dropped, num_to_drop = select_drop_bottom_tickers(
        history,
        drop_bottom_pct=float(drop_bottom_pct),
        volume_history=volume_history,
    )
    total_tickers = len(yahoo_tickers)
    tickers_kept = max(total_tickers - num_to_drop, 0)
    available_tickers = len(kept)
    actual_kept = max(min(tickers_kept, available_tickers), 0)
    filtered_tickers = kept.head(actual_kept).index.tolist()
    available_days = int(kept.head(actual_kept).min()) if actual_kept > 0 else 0
    dropped_tickers = dropped.index.tolist() if num_to_drop > 0 else []

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


def search_universal_shortlist_caps(
    owner,
    *,
    starting_cap,
    minimum_cap=None,
    starting_asset_limit=None,
    max_dd,
    target_volatility,
    drop_bottom_pct,
    history_buffer_days,
    redundancy_corr_threshold,
    exact_optimizer_asset_cap=300,
    exact_optimizer_asset_caps=None,
    minimum_trading_days=252,
    step=50,
    maximum_cap=None,
    adaptive=True,
    asset_evaluation_order="ascending",
    alternating_refinement=True,
    gradient_return_scale=0.001,
    runtime_brake_seconds=900,
    solver_timeout_seconds=600,
    prior_results=(),
    prepared_matrix_store=None,
    on_result=None,
    on_progress=None,
):
    """Explore shortlist capacity and exact-solver dimensionality.

    ``step`` is the minimum resolution, not the jump used for every evaluation.
    The search starts with a larger jump and adjusts it from the common-history
    headroom. It always evaluates the selected lower endpoint and, when the
    history floor is never breached, the eligible-universe upper endpoint.
    Every evaluated shortlist cap is crossed with every requested exact-asset
    limit. Market-history preparation is performed once per shortlist cap.
    """
    search_started = time.monotonic() if on_progress is not None else None

    def report_progress(stage, fraction, details=None):
        if on_progress is None:
            return
        payload = dict(details or {})
        payload.setdefault("elapsed_seconds", time.monotonic() - search_started)
        on_progress(stage, float(np.clip(fraction, 0.0, 1.0)), payload)

    report_progress("Loading current holdings", 0.0)
    portfolio_df, invalid_rows = build_current_allocation_from_db(owner)
    if portfolio_df.empty:
        raise ValueError("No usable holdings are available for shortlist-cap search.")

    minimum_increment = max(int(step), 50)
    lower_cap = max(int(minimum_cap if minimum_cap is not None else starting_cap), minimum_increment)
    cap = max(int(starting_cap), lower_cap)
    if maximum_cap is not None:
        cap = min(cap, int(maximum_cap))
    rows = [dict(item) for item in prior_results]
    exact_asset_limits = tuple(
        sorted(
            {
                max(int(value), 1)
                for value in (
                    exact_optimizer_asset_caps
                    if exact_optimizer_asset_caps is not None
                    else (exact_optimizer_asset_cap,)
                )
            }
        )
    )
    if not exact_asset_limits:
        raise ValueError("At least one Maximum-assets value is required.")
    preferred_asset_limit = min(
        exact_asset_limits,
        key=lambda value: (
            abs(int(value) - int(starting_asset_limit)) if starting_asset_limit is not None else 0,
            int(value),
        ),
    )
    two_dimensional = len(exact_asset_limits) > 1
    if adaptive and two_dimensional and alternating_refinement:
        # Start with the low-cost lower-bound stencil. The saved best and
        # configured warm start are evaluated as additional seeds below.
        cap = lower_cap
    completed_pairs = {
        (int(row["Shortlist cap"]), int(row["Maximum assets"]))
        for row in rows
        if row.get("Shortlist cap") is not None
        and row.get("Maximum assets") is not None
    }
    if rows and not two_dimensional:
        last_row = rows[-1]
        last_cap = int(last_row.get("Shortlist cap") or 0)
        last_eligible = int(last_row.get("Eligible universe") or 0)
        active_pair_completed = (int(cap), int(preferred_asset_limit)) in completed_pairs
        if active_pair_completed and (
            int(last_row.get("Trading days") or 0) < int(minimum_trading_days)
            or (last_eligible and last_cap >= last_eligible)
            or (maximum_cap is not None and last_cap >= int(maximum_cap))
        ):
            return rows, invalid_rows
        previous_jump = max(
            int(last_row.get("Next adaptive jump") or minimum_increment),
            minimum_increment,
        )
        cap = max(cap, last_cap + previous_jump)
        if last_eligible:
            cap = min(cap, last_eligible)
        if maximum_cap is not None:
            cap = min(cap, int(maximum_cap))

    def aligned_jump(raw_jump):
        raw_jump = max(int(raw_jump), minimum_increment)
        return (
            (raw_jump + minimum_increment - 1) // minimum_increment
        ) * minimum_increment

    def next_adaptive_jump(current_jump, trading_days):
        if not adaptive:
            return minimum_increment
        headroom = max(int(trading_days) - int(minimum_trading_days), 0)
        if headroom >= int(minimum_trading_days):
            raw_jump = int(current_jump) * 2
        elif headroom >= int(minimum_trading_days) // 2:
            raw_jump = int(current_jump) * 3 // 2
        elif headroom < max(int(minimum_trading_days) // 8, 1):
            raw_jump = int(current_jump) // 2
        else:
            raw_jump = int(current_jump)
        return aligned_jump(raw_jump)

    current_jump = (
        max(int(rows[-1].get("Next adaptive jump") or 0), minimum_increment)
        if rows
        else aligned_jump(minimum_increment * 4 if adaptive else minimum_increment)
    )

    while True:
        pending_asset_limits = [
            value
            for value in exact_asset_limits
            if (int(cap), int(value)) not in completed_pairs
        ]
        if not pending_asset_limits:
            if maximum_cap is not None and cap >= int(maximum_cap):
                break
            cap += minimum_increment
            if maximum_cap is not None:
                cap = min(cap, int(maximum_cap))
            continue
        cap_started = time.monotonic()
        report_progress(
            "Shortlisting the universal candidate pool",
            0.02,
            {"cap": int(cap)},
        )
        cached_prepared = (
            prepared_matrix_store.get(int(cap))
            if isinstance(prepared_matrix_store, dict)
            else None
        )
        if cached_prepared is not None:
            extended_df, prepared_log_returns, analysis_meta = cached_prepared
            extended_df = extended_df.copy()
            prepared_log_returns = prepared_log_returns.copy()
            analysis_meta = dict(analysis_meta or {})
            preselection = dict(analysis_meta.get("universal_preselection") or {})
            added_symbols = [None] * int(preselection.get("shortlisted") or 0)
        else:
            extended_df, added_symbols, preselection = (
                extend_allocation_with_universal_candidates(
                    portfolio_df.copy(),
                    maximum_candidates=cap,
                    owner=owner,
                )
            )
            tickers = extended_df["Yahoo Ticker"].dropna().astype(str).tolist()
            if not tickers:
                raise ValueError("The selected shortlist cap produced no analysable tickers.")

        report_progress(
            "Shortlist ready; preparing full-history analysis",
            0.08,
            {
                "cap": int(cap),
                "shortlisted": int(
                    preselection.get("shortlisted") or len(added_symbols)
                ),
                "eligible": int(preselection.get("eligible") or 0),
            },
        )

        def report_cap_analysis(stage, fraction, details=None):
            payload = dict(details or {})
            payload["cap"] = int(cap)
            report_progress(stage, 0.08 + 0.90 * float(fraction), payload)

        largest_exact_limit = max(exact_asset_limits)
        prepare_without_largest_solve = bool(
            adaptive and two_dimensional and alternating_refinement
        )
        if cached_prepared is None:
            optimal_weights, prepared_log_returns, _, optimal_stats, analysis_meta = (
                run_portfolio_analysis_multi(
                    tickers,
                    extended_df.copy(),
                    max_dd=max_dd,
                    target_volatility=target_volatility,
                    drop_bottom_pct=drop_bottom_pct,
                    buffer_days=history_buffer_days,
                    redundancy_corr_threshold=redundancy_corr_threshold,
                    exact_optimizer_asset_cap=largest_exact_limit,
                    progress_callback=report_cap_analysis,
                    solve_portfolio=not prepare_without_largest_solve,
                    optimizer_timeout_seconds=solver_timeout_seconds,
                )
            )
        else:
            optimal_weights, optimal_stats = None, None
        if prepared_log_returns is None:
            raise ValueError(f"Optimization returned no usable return matrix at cap {cap:,}.")

        trading_days = int(prepared_log_returns.shape[0])
        eligible = int(preselection.get("eligible") or 0)
        elapsed_seconds = float(time.monotonic() - cap_started)
        next_jump = (
            0
            if trading_days < int(minimum_trading_days)
            else next_adaptive_jump(current_jump, trading_days)
        )
        runtime_brake_applied = False
        if (
            adaptive
            and next_jump
            and elapsed_seconds >= float(runtime_brake_seconds)
        ):
            reduced_jump = aligned_jump(max(next_jump // 2, minimum_increment))
            runtime_brake_applied = reduced_jump < next_jump
            next_jump = reduced_jump
        largest_result = (
            (
                optimal_weights,
                prepared_log_returns,
                optimal_stats,
                (analysis_meta or {}).get("exact_optimizer_screen", {}),
            )
            if not prepare_without_largest_solve
            else None
        )
        solved_by_effective_limit = {}

        def evaluate_asset_limit(asset_limit, search_phase):
            pair_started = time.monotonic() if two_dimensional else None
            effective_limit = min(int(asset_limit), int(prepared_log_returns.shape[1]))
            if int(asset_limit) == largest_exact_limit and largest_result is not None:
                pair_weights, pair_returns, pair_stats, solver_screen = largest_result
            elif effective_limit in solved_by_effective_limit:
                pair_weights, pair_returns, pair_stats, solver_screen = (
                    solved_by_effective_limit[effective_limit]
                )
            else:
                pair_weights, pair_returns, _, pair_stats, solver_screen = (
                    optimize_prepared_return_matrix(
                        prepared_log_returns,
                        extended_df.copy(),
                        maximum_assets=int(asset_limit),
                        max_dd=max_dd,
                        target_volatility=target_volatility,
                        integrity_report=(analysis_meta or {}).get(
                            "price_integrity_report"
                        ),
                        timeout_seconds=solver_timeout_seconds,
                    )
                )
                solved_by_effective_limit[effective_limit] = (
                    pair_weights,
                    pair_returns,
                    pair_stats,
                    solver_screen,
                )

            feasible = (
                trading_days >= int(minimum_trading_days)
                and pair_weights is not None
                and pair_stats is not None
            )
            solver_timed_out = bool(
                (solver_screen or {}).get("timed_out")
                or (analysis_meta or {}).get("optimizer_timed_out")
            )
            status = (
                f"Below {int(minimum_trading_days)}-session floor"
                if trading_days < int(minimum_trading_days)
                else (
                    "Timed out; resumable pair skipped"
                    if solver_timed_out
                    else ("Feasible" if feasible else "Solver failed")
                )
            )
            pair_elapsed = (
                float(time.monotonic() - pair_started)
                if pair_started is not None
                else elapsed_seconds
            )
            cap_elapsed = (
                float(time.monotonic() - cap_started)
                if two_dimensional
                else elapsed_seconds
            )
            pair_returns = pair_returns if pair_returns is not None else prepared_log_returns
            pair_stats = pair_stats or {}
            last_index_value = pair_returns.index[-1] if len(pair_returns.index) else None
            analysis_cutoff = (
                last_index_value.date().isoformat()
                if hasattr(last_index_value, "date")
                else None
            )
            row = {
                "Search phase": search_phase,
                "Shortlist cap": int(cap),
                "Maximum assets": int(asset_limit),
                "Effective maximum assets": int(
                    (solver_screen or {}).get("selected_assets")
                    or pair_returns.shape[1]
                ),
                "Shortlisted": int(
                    preselection.get("shortlisted") or len(added_symbols)
                ),
                "Assets": int(pair_returns.shape[1]),
                "Trading days": trading_days,
                "Analysis cutoff": analysis_cutoff,
                "Annual Return": float(pair_stats.get("Annual Return", np.nan)),
                "Annual Volatility": float(
                    pair_stats.get("Annual Volatility", np.nan)
                ),
                "Historical ES 95% (1 Session)": float(
                    pair_stats.get("Historical ES 95% (1 Session)", np.nan)
                ),
                "Block-Bootstrap ES 95% (20 Sessions)": float(
                    pair_stats.get("Block-Bootstrap ES 95% (20 Sessions)", np.nan)
                ),
                "Sharpe Ratio": float(pair_stats.get("Sharpe Ratio", np.nan)),
                "Elapsed seconds": pair_elapsed,
                "Cap elapsed seconds": cap_elapsed,
                "Eligible universe": eligible,
                "Next adaptive jump": int(next_jump),
                "Runtime brake": (
                    "Applied"
                    if adaptive and runtime_brake_applied
                    else ("Not needed" if adaptive else "Not applicable")
                ),
                "Status": status,
            }
            rows.append(row)
            completed_pairs.add((int(cap), int(asset_limit)))
            if on_result is not None:
                on_result(rows, row)
            return row

        if adaptive and two_dimensional:
            if alternating_refinement:
                initial_limits = (
                    [exact_asset_limits[0]]
                    if exact_asset_limits[0] in pending_asset_limits
                    else []
                )
            else:
                anchor_limits = [
                    value for value in adaptive_anchor_values(
                        exact_asset_limits, target_points=8,
                    )
                    if value in pending_asset_limits
                ]
                initial_limits = (
                    [preferred_asset_limit]
                    if preferred_asset_limit in pending_asset_limits
                    else []
                ) + [value for value in anchor_limits if value != preferred_asset_limit]
            initial_phase = (
                "Adaptive lower-bound seed"
                if alternating_refinement
                else "Derivative-free anchor"
            )
        elif str(asset_evaluation_order) == "coarse_to_fine":
            ordered_limits = [
                value for value in coarse_to_fine_values(exact_asset_limits)
                if value in pending_asset_limits
            ]
            initial_limits = ([preferred_asset_limit] if preferred_asset_limit in pending_asset_limits else []) + [
                value for value in ordered_limits if value != preferred_asset_limit
            ]
            initial_phase = "Complete grid"
        else:
            initial_limits = ([preferred_asset_limit] if preferred_asset_limit in pending_asset_limits else []) + [
                value for value in pending_asset_limits if value != preferred_asset_limit
            ]
            initial_phase = "Complete grid"

        for asset_limit in initial_limits:
            evaluate_asset_limit(asset_limit, initial_phase)

        if adaptive and two_dimensional and not alternating_refinement:
            for _ in range(6):
                cap_rows = [
                    row for row in rows
                    if int(row.get("Shortlist cap") or -1) == int(cap)
                    and str(row.get("Status")) == "Feasible"
                ]
                current_best = best_feasible_result(cap_rows)
                if current_best is None:
                    break
                evaluated_limits = {
                    int(row["Maximum assets"])
                    for row in rows
                    if int(row.get("Shortlist cap") or -1) == int(cap)
                    and row.get("Maximum assets") is not None
                }
                refinement_limits = local_refinement_values(
                    exact_asset_limits,
                    evaluated_limits,
                    int(current_best["Maximum assets"]),
                    radius=2,
                )
                if not refinement_limits:
                    break
                for asset_limit in refinement_limits:
                    evaluate_asset_limit(asset_limit, "Derivative-free refinement")

        if isinstance(prepared_matrix_store, dict) and prepared_log_returns is not None:
            if isinstance(analysis_meta, dict):
                preselection_snapshot = dict(preselection or {})
                preselection_snapshot["cap"] = int(cap)
                analysis_meta["universal_preselection"] = preselection_snapshot
            prepared_matrix_store[int(cap)] = (
                extended_df.copy(),
                prepared_log_returns.copy(),
                dict(analysis_meta or {}),
            )
            # Keep all matrices for a small configured cap range so a completed
            # local search can hand every prepared cap to the final optimizer.
            # On broad scans retain only the strongest completed caps to bound
            # memory use during walk-forward scoring.
            cap_scores = {}
            for result_row in rows:
                if result_row.get("Status") != "Feasible":
                    continue
                result_cap = int(result_row["Shortlist cap"])
                result_return = float(result_row.get("Annual Return", -np.inf))
                cap_scores[result_cap] = max(
                    result_return, cap_scores.get(result_cap, -np.inf)
                )
            # A recursive coordinate solve prepares its next cap before the
            # parent search merges that result row. Keep already-cached matrices
            # eligible during that short handoff window, while retaining the
            # same three-cap memory bound.
            for cached_cap in prepared_matrix_store:
                cap_scores.setdefault(int(cached_cap), -np.inf)
            configured_cap_count = (
                len(inclusive_values(
                    int(lower_cap), int(maximum_cap), minimum_increment,
                    minimum_step=50,
                ))
                if maximum_cap is not None else len(cap_scores)
            )
            retained_caps = set(cap_scores) if configured_cap_count <= 3 else {
                result_cap for result_cap, _ in sorted(
                    cap_scores.items(), key=lambda item: item[1], reverse=True
                )[:3]
            }
            for cached_cap in list(prepared_matrix_store):
                if int(cached_cap) not in retained_caps:
                    prepared_matrix_store.pop(cached_cap, None)
        report_progress(
            "Cap evaluation complete",
            1.0,
            {
                "cap": int(cap),
                "trading_days": trading_days,
                "assets": int(prepared_log_returns.shape[1]),
            },
        )

        if trading_days < int(minimum_trading_days):
            break
        if adaptive and two_dimensional and alternating_refinement:
            break
        range_ceiling = (
            min(eligible, int(maximum_cap))
            if eligible and maximum_cap is not None
            else (eligible or int(maximum_cap or cap))
        )
        if cap >= range_ceiling:
            break
        current_jump = next_jump
        cap = min(cap + current_jump, range_ceiling)

    if not adaptive and lower_cap < int(starting_cap):
        for candidate_cap in range(lower_cap, int(starting_cap), minimum_increment):
            refined, _ = search_universal_shortlist_caps(
                owner, starting_cap=candidate_cap, minimum_cap=candidate_cap,
                starting_asset_limit=preferred_asset_limit, maximum_cap=candidate_cap,
                max_dd=max_dd, target_volatility=target_volatility,
                drop_bottom_pct=drop_bottom_pct, history_buffer_days=history_buffer_days,
                redundancy_corr_threshold=redundancy_corr_threshold,
                exact_optimizer_asset_cap=max(exact_asset_limits),
                exact_optimizer_asset_caps=exact_asset_limits,
                minimum_trading_days=minimum_trading_days, step=step,
                adaptive=False, asset_evaluation_order=asset_evaluation_order,
                alternating_refinement=False,
            )
            for new_row in refined:
                pair = (int(new_row["Shortlist cap"]), int(new_row["Maximum assets"]))
                if pair in completed_pairs:
                    continue
                rows.append(dict(new_row))
                completed_pairs.add(pair)
                if on_result is not None:
                    on_result(rows, rows[-1])

    if (
        adaptive
        and two_dimensional
        and alternating_refinement
        and maximum_cap is not None
    ):
        adaptive_saturated = False
        saturation_reason = None
        cap_grid = inclusive_values(
            int(lower_cap), int(maximum_cap), minimum_increment,
            minimum_step=50,
        )

        def merge_refinement(new_rows, phase, model_predictions=None):
            model_predictions = model_predictions or {}
            for new_row in new_rows:
                pair = (int(new_row["Shortlist cap"]), int(new_row["Maximum assets"]))
                if pair in completed_pairs:
                    continue
                merged = dict(new_row)
                merged["Search phase"] = phase
                if pair in model_predictions:
                    predicted_return = float(model_predictions[pair])
                    merged["Linear fit predicted return"] = predicted_return
                    merged["Deviation from linear fit"] = (
                        float(merged["Annual Return"]) - predicted_return
                        if str(merged.get("Status")) == "Feasible"
                        and np.isfinite(float(merged.get("Annual Return", np.nan)))
                        else np.nan
                    )
                rows.append(merged)
                completed_pairs.add(pair)
                if on_result is not None:
                    on_result(rows, merged)

        def evaluate_coordinate(
            candidate_x, candidate_y, phase, *, model_prediction=None
        ):
            pair = (int(candidate_x), int(candidate_y))
            if pair in completed_pairs:
                return False
            refined, _ = search_universal_shortlist_caps(
                owner,
                starting_cap=int(candidate_x),
                minimum_cap=int(candidate_x),
                maximum_cap=int(candidate_x),
                starting_asset_limit=int(candidate_y),
                max_dd=max_dd,
                target_volatility=target_volatility,
                drop_bottom_pct=drop_bottom_pct,
                history_buffer_days=history_buffer_days,
                redundancy_corr_threshold=redundancy_corr_threshold,
                exact_optimizer_asset_cap=int(candidate_y),
                exact_optimizer_asset_caps=(int(candidate_y),),
                minimum_trading_days=minimum_trading_days,
                step=step,
                adaptive=False,
                asset_evaluation_order="coarse_to_fine",
                alternating_refinement=False,
                prior_results=rows,
                prepared_matrix_store=prepared_matrix_store,
            )
            prediction_map = (
                {pair: model_prediction}
                if model_prediction is not None else None
            )
            merge_refinement(refined, phase, prediction_map)
            return True

        def gradient_target(line_rows, coordinate_key, grid, center, coordinate_step):
            samples = sorted(
                (
                    int(row[coordinate_key]),
                    float(row["Annual Return"]),
                )
                for row in line_rows
                if str(row.get("Status")) == "Feasible"
                and np.isfinite(float(row.get("Annual Return", np.nan)))
            )
            lower = [sample for sample in samples if sample[0] < center]
            upper = [sample for sample in samples if sample[0] > center]
            if lower and upper:
                left, right = lower[-1], upper[0]
            elif upper:
                left = next((sample for sample in samples if sample[0] == center), None)
                right = upper[0]
            elif lower:
                left = lower[-1]
                right = next((sample for sample in samples if sample[0] == center), None)
            else:
                return None
            if left is None or right is None or right[0] == left[0]:
                return None
            slope = (right[1] - left[1]) / (right[0] - left[0])
            if not np.isfinite(slope) or abs(slope) < 1e-15:
                return None
            response_per_step = abs(slope) * max(int(coordinate_step), 1)
            scale = max(float(gradient_return_scale), 1e-6)
            step_count = int(np.clip(round(response_per_step / scale), 1, 8))
            raw_target = (
                center + (1 if slope > 0 else -1) * step_count * coordinate_step
            )
            return min(grid, key=lambda value: (abs(value - raw_target), value))

        # A shared baseline plus one X and one Y probe gives two observations
        # for each one-dimensional linear fit. New solver residuals refit the
        # relevant axis after every point, so the search stays incremental.
        lower_x = cap_grid[0]
        lower_y = exact_asset_limits[0]
        seed_pairs = [(lower_x, lower_y)]
        if len(cap_grid) > 1:
            seed_pairs.append((cap_grid[1], lower_y))
        if len(exact_asset_limits) > 1:
            seed_pairs.append((lower_x, exact_asset_limits[1]))
        saved_best_seed = best_feasible_result(rows)
        if saved_best_seed is not None:
            saved_x = int(saved_best_seed["Shortlist cap"])
            saved_y = int(saved_best_seed["Maximum assets"])
            if (
                cap_grid[0] <= saved_x <= cap_grid[-1]
                and exact_asset_limits[0] <= saved_y <= exact_asset_limits[-1]
            ):
                seed_pairs.append((saved_x, saved_y))
        # The explicitly configured warm start is also useful when its saved
        # result is absent from this checkpoint.
        if (
            cap_grid[0] <= int(starting_cap) <= cap_grid[-1]
            and exact_asset_limits[0] <= preferred_asset_limit <= exact_asset_limits[-1]
        ):
            seed_pairs.append((int(starting_cap), preferred_asset_limit))

        for seed_x, seed_y in dict.fromkeys(seed_pairs):
            seed_pair = (int(seed_x), int(seed_y))
            if seed_pair not in completed_pairs:
                report_progress(
                    f"Running initial seed solver at X {seed_x:,} × Y {seed_y:,}",
                    0.98,
                    {
                        "cap": seed_x,
                        "running_pair": {"x": seed_x, "y": seed_y},
                    },
                )
            evaluate_coordinate(seed_x, seed_y, "Adaptive lower-bound seed")

        accepted_incumbent = best_feasible_result(rows)
        material_improvement = max(float(gradient_return_scale), 0.0)
        axis_trust_radius = {"X": 1, "Y": 1}
        axis_direction = {"X": 0, "Y": 0}

        def fit_local_linear_response(incumbent, axis, maximum_samples=12):
            """Fit a one-axis response from two or more observations on its line."""
            coordinate_key = "Shortlist cap" if axis == "X" else "Maximum assets"
            fixed_key = "Maximum assets" if axis == "X" else "Shortlist cap"
            fixed_value = int(incumbent[fixed_key])
            center = int(incumbent[coordinate_key])
            samples = []
            for row in rows:
                if str(row.get("Status")) != "Feasible":
                    continue
                try:
                    coordinate = float(row[coordinate_key])
                    fixed = int(row[fixed_key])
                    result = float(row["Annual Return"])
                except (KeyError, TypeError, ValueError):
                    continue
                if fixed == fixed_value and np.isfinite([coordinate, result]).all():
                    samples.append((abs(coordinate - center), coordinate, result))
            samples.sort(key=lambda item: item[0])
            samples = samples[:max(2, int(maximum_samples))]
            if len(samples) < 2:
                return None
            coordinates = np.asarray([item[1] for item in samples], dtype=float)
            returns = np.asarray([item[2] for item in samples], dtype=float)
            centered = coordinates - float(np.mean(coordinates))
            variance = float(np.dot(centered, centered))
            if variance <= 0:
                return None
            slope = float(np.dot(centered, returns - float(np.mean(returns))) / variance)
            intercept = float(np.mean(returns) - slope * np.mean(coordinates))
            residuals = returns - (intercept + slope * coordinates)
            if not np.isfinite([slope, intercept]).all():
                return None
            return {
                "slope": slope,
                "intercept": intercept,
                "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                "sample_count": len(samples),
            }

        def predict_linear_response(model, candidate_x, candidate_y, axis):
            if model is None:
                return None
            coordinate = candidate_x if axis == "X" else candidate_y
            predicted = float(model["intercept"] + model["slope"] * coordinate)
            return predicted if np.isfinite(predicted) else None

        def refine_axis(axis, max_iterations=16):
            nonlocal accepted_incumbent
            improved_any = False
            coordinate_key = "Shortlist cap" if axis == "X" else "Maximum assets"
            grid = cap_grid if axis == "X" else exact_asset_limits
            coordinate_step = min(
                (abs(int(right) - int(left)) for left, right in zip(grid, grid[1:])),
                default=minimum_increment,
            )
            for _ in range(max_iterations):
                incumbent = accepted_incumbent
                if incumbent is None:
                    break
                fixed_x = int(incumbent["Shortlist cap"])
                fixed_y = int(incumbent["Maximum assets"])
                line_rows = [
                    row for row in rows
                    if (
                        int(row.get("Maximum assets") or -1) == fixed_y
                        if axis == "X"
                        else int(row.get("Shortlist cap") or -1) == fixed_x
                    )
                ]
                center = int(incumbent[coordinate_key])
                center_index = min(
                    range(len(grid)), key=lambda index: abs(int(grid[index]) - center)
                )
                model = fit_local_linear_response(incumbent, axis)
                model_slope = float(model["slope"]) if model is not None else 0.0
                slope_target = None
                if abs(model_slope) > 1e-12:
                    # Translate the fitted return slope into a bounded jump in
                    # this axis's own grid units. This avoids walking one cell
                    # at a time when the LR clearly points farther away, while
                    # the solver residual recalibrates the next jump.
                    response_per_step = abs(model_slope) * coordinate_step
                    scale = max(float(gradient_return_scale), 1e-6)
                    step_count = int(np.clip(round(response_per_step / scale), 1, 8))
                    raw_target = center + (
                        1 if model_slope > 0 else -1
                    ) * step_count * coordinate_step
                    slope_target = min(
                        grid, key=lambda value: (abs(value - raw_target), value)
                    )
                if slope_target is None:
                    slope_target = gradient_target(
                        line_rows, coordinate_key, grid, center, coordinate_step
                    )
                if axis_direction[axis] == 0:
                    if abs(model_slope) > 1e-12:
                        axis_direction[axis] = 1 if model_slope > 0 else -1
                    elif slope_target is not None and int(slope_target) != center:
                        axis_direction[axis] = (
                            1 if int(slope_target) > center else -1
                        )
                    else:
                        axis_direction[axis] = 1

                radius = max(1, int(axis_trust_radius[axis]))
                direction = int(axis_direction[axis])
                candidate_indices = [
                    center_index + direction * radius,
                    center_index - direction * radius,
                ]
                candidates = []
                for candidate_index in candidate_indices:
                    if not 0 <= candidate_index < len(grid):
                        continue
                    coordinate = int(grid[candidate_index])
                    candidate_x = coordinate if axis == "X" else fixed_x
                    candidate_y = fixed_y if axis == "X" else coordinate
                    pair = (candidate_x, candidate_y)
                    if pair not in completed_pairs and pair not in candidates:
                        candidates.append(pair)
                # A model-directed slope proposal is retained as a fallback
                # when a resumed/off-grid center has no point at this radius.
                if slope_target is not None and int(slope_target) != center:
                    coordinate = int(slope_target)
                    pair = (
                        (coordinate, fixed_y)
                        if axis == "X" else (fixed_x, coordinate)
                    )
                    if pair not in completed_pairs and pair not in candidates:
                        candidates.append(pair)
                if not candidates:
                    axis_trust_radius[axis] = max(1, radius // 2)
                    if radius <= 1:
                        break
                    continue

                before = float(incumbent["Annual Return"])
                predictions = {
                    pair: predict_linear_response(model, *pair, axis)
                    for pair in candidates
                }
                candidates.sort(
                    key=lambda pair: (
                        predictions[pair]
                        if predictions[pair] is not None
                        else -np.inf,
                        pair,
                    ),
                    reverse=True,
                )
                candidate_x, candidate_y = candidates[0]
                predicted_return = predictions[(candidate_x, candidate_y)]
                delta_steps = max(
                    1,
                    round(
                        abs(
                            (candidate_x - fixed_x)
                            if axis == "X"
                            else (candidate_y - fixed_y)
                        ) / max(coordinate_step, 1)
                    ),
                )
                report_progress(
                    f"LR proposes {axis} move of {delta_steps} grid interval(s) "
                    f"toward {candidate_x:,} × {candidate_y:,}",
                    0.98,
                    {
                        "cap": candidate_x,
                        "next_probe": {
                            "from_x": fixed_x,
                            "from_y": fixed_y,
                            "x": candidate_x,
                            "y": candidate_y,
                            "axis": axis,
                            "delta_steps": delta_steps,
                            "predicted_return": predicted_return,
                        },
                        "running_pair": {"x": candidate_x, "y": candidate_y},
                    },
                )
                added = evaluate_coordinate(
                    candidate_x,
                    candidate_y,
                    f"Alternating {axis} linear-fit refinement",
                    model_prediction=predicted_return,
                )
                if not added:
                    break

                evaluated = next(
                    (
                        row for row in reversed(rows)
                        if int(row.get("Shortlist cap") or -1) == candidate_x
                        and int(row.get("Maximum assets") or -1) == candidate_y
                    ),
                    None,
                )
                if evaluated is None or str(evaluated.get("Status")) != "Feasible":
                    axis_trust_radius[axis] = max(1, radius // 2)
                    axis_direction[axis] = -direction
                    break
                actual_return = float(evaluated["Annual Return"])
                actual_improvement = actual_return - before
                prediction_error = (
                    actual_return - predicted_return
                    if predicted_return is not None else 0.0
                )
                error_threshold = max(
                    material_improvement,
                    float(model.get("rmse", 0.0)) * 0.5 if model else 0.0,
                )
                if actual_improvement > material_improvement:
                    accepted_incumbent = best_feasible_result(rows)
                    improved_any = True
                    if prediction_error > error_threshold:
                        # The actual optimizer beat the plane: explore farther
                        # in this direction while the region remains promising.
                        axis_trust_radius[axis] = min(radius * 2, 8)
                        axis_direction[axis] = direction
                    elif prediction_error < -error_threshold:
                        # The plane was optimistic: reduce the move and retest
                        # around the observed surface instead of extrapolating.
                        axis_trust_radius[axis] = max(1, radius // 2)
                        axis_direction[axis] = direction
                    else:
                        axis_trust_radius[axis] = radius
                        axis_direction[axis] = direction
                else:
                    axis_trust_radius[axis] = max(1, radius // 2)
                    axis_direction[axis] = -direction
            return improved_any

        def incumbent_has_timed_out_neighbour():
            if accepted_incumbent is None:
                return False
            best_x = int(accepted_incumbent["Shortlist cap"])
            best_y = int(accepted_incumbent["Maximum assets"])
            neighbour_pairs = set()
            for grid, center, axis in (
                (cap_grid, best_x, "X"),
                (exact_asset_limits, best_y, "Y"),
            ):
                for neighbour in neighbouring_grid_values(grid, center):
                    neighbour_pairs.add(
                        (int(neighbour), best_y)
                        if axis == "X"
                        else (best_x, int(neighbour))
                    )
            return any(
                (int(row.get("Shortlist cap") or -1), int(row.get("Maximum assets") or -1))
                in neighbour_pairs
                and str(row.get("Status", "")).startswith("Timed out")
                for row in rows
            )

        for _ in range(12):
            before_pair = accepted_incumbent
            before_key = (
                int(before_pair["Shortlist cap"]),
                int(before_pair["Maximum assets"]),
            ) if before_pair is not None else None
            moved_x = refine_axis("X")
            moved_y = refine_axis("Y")
            after_pair = accepted_incumbent
            after_key = (
                int(after_pair["Shortlist cap"]),
                int(after_pair["Maximum assets"]),
            ) if after_pair is not None else None
            if not moved_x and not moved_y:
                blocked_by_timeout = incumbent_has_timed_out_neighbour()
                adaptive_saturated = (
                    accepted_incumbent is not None and not blocked_by_timeout
                )
                saturation_reason = (
                    "A neighbouring pair timed out, so saturation was not declared."
                    if blocked_by_timeout
                    else "A complete X/Y pass produced no material improvement."
                )
                break
            if before_key == after_key:
                adaptive_saturated = accepted_incumbent is not None
                saturation_reason = "The alternating X/Y pass returned to the same best pair."
                break

        if (
            rows
            and isinstance(prepared_matrix_store, dict)
            and maximum_cap is not None
            and adaptive
            and two_dimensional
            and len(cap_grid) <= 3
        ):
            # Tiny configured searches are cheap enough to retain a prepared
            # history for every cap, even when the LR converges early. This
            # keeps final auto-run and resume consumers fully reusable.
            for missing_cap in cap_grid:
                if int(missing_cap) in prepared_matrix_store:
                    continue
                prepared_df, missing_symbols, missing_preselection = (
                    extend_allocation_with_universal_candidates(
                        portfolio_df.copy(),
                        maximum_candidates=int(missing_cap),
                        owner=owner,
                    )
                )
                missing_tickers = prepared_df["Yahoo Ticker"].dropna().astype(str).tolist()
                if not missing_tickers:
                    continue
                _, missing_log_returns, _, _, missing_meta = run_portfolio_analysis_multi(
                    missing_tickers,
                    prepared_df.copy(),
                    max_dd=max_dd,
                    target_volatility=target_volatility,
                    drop_bottom_pct=drop_bottom_pct,
                    buffer_days=history_buffer_days,
                    redundancy_corr_threshold=redundancy_corr_threshold,
                    exact_optimizer_asset_cap=max(exact_asset_limits),
                    solve_portfolio=False,
                    optimizer_timeout_seconds=solver_timeout_seconds,
                )
                if missing_log_returns is not None:
                    missing_snapshot = dict(missing_preselection or {})
                    missing_snapshot["cap"] = int(missing_cap)
                    missing_meta = dict(missing_meta or {})
                    missing_meta["universal_preselection"] = missing_snapshot
                    prepared_matrix_store[int(missing_cap)] = (
                        prepared_df.copy(), missing_log_returns.copy(), missing_meta
                    )
        if rows:
            rows[-1]["Search saturated"] = bool(adaptive_saturated)
            rows[-1]["Saturation reason"] = saturation_reason

    return rows, invalid_rows


def local_refinement_cap_pair(
    results,
    *,
    minimum_cap,
    maximum_cap,
    minimum_step=50,
):
    """Return untested midpoint caps immediately left and right of the best cap."""
    if not results:
        return []
    frame = pd.DataFrame(results)
    required = {"Shortlist cap", "Annual Return", "Status"}
    if not required.issubset(frame.columns):
        return []
    feasible = frame.loc[frame["Status"].eq("Feasible")].copy()
    if feasible.empty:
        return []

    feasible["Annual Return"] = pd.to_numeric(
        feasible["Annual Return"], errors="coerce"
    )
    feasible = feasible.dropna(subset=["Annual Return"])
    if feasible.empty:
        return []
    center = int(feasible.loc[feasible["Annual Return"].idxmax(), "Shortlist cap"])
    increment = max(int(minimum_step), 50)
    lower_bound = max(int(minimum_cap), increment)
    upper_bound = max(int(maximum_cap), lower_bound)
    tested = sorted(
        {
            int(value)
            for value in pd.to_numeric(frame["Shortlist cap"], errors="coerce").dropna()
        }
    )
    tested_set = set(tested)

    left_boundary = max(
        [value for value in tested if value < center] + [lower_bound]
    )
    right_boundary = min(
        [value for value in tested if value > center] + [upper_bound]
    )

    def aligned_midpoint(low, high):
        if high - low <= increment:
            return None
        raw = (int(low) + int(high)) / 2.0
        candidate = int(np.floor(raw / increment + 0.5) * increment)
        candidate = max(candidate, int(low) + increment)
        candidate = min(candidate, int(high) - increment)
        if candidate <= low or candidate >= high or candidate in tested_set:
            return None
        return candidate

    pair = []
    left = aligned_midpoint(left_boundary, center)
    right = aligned_midpoint(center, right_boundary)
    if left is not None:
        pair.append(left)
    if right is not None:
        pair.append(right)
    return pair


def refinement_is_material(
    baseline,
    candidates,
    *,
    minimum_return_improvement=0.01,
    maximum_es_worsening=0.005,
):
    """Require a useful return gain without materially worse 20-session ES."""
    if not baseline or not candidates:
        return False
    baseline_return = float(baseline.get("Annual Return", np.nan))
    baseline_es = float(
        baseline.get("Block-Bootstrap ES 95% (20 Sessions)", np.nan)
    )
    if not np.isfinite(baseline_return):
        return False
    for row in candidates:
        if row.get("Status") != "Feasible":
            continue
        candidate_return = float(row.get("Annual Return", np.nan))
        candidate_es = float(
            row.get("Block-Bootstrap ES 95% (20 Sessions)", np.nan)
        )
        return_improved = (
            np.isfinite(candidate_return)
            and candidate_return >= baseline_return + float(minimum_return_improvement)
        )
        es_acceptable = (
            not np.isfinite(baseline_es)
            or not np.isfinite(candidate_es)
            or candidate_es <= baseline_es + float(maximum_es_worsening)
        )
        if return_improved and es_acceptable:
            return True
    return False


def _apply_best_shortlist_cap(cap):
    st.session_state["universal_preselection_cap"] = int(cap)


def _apply_best_search_pair(cap, maximum_assets):
    cap = int(cap)
    maximum_assets = int(maximum_assets)
    # Keep the research warm start and the ordinary optimizer controls aligned.
    # This callback runs before Streamlit recreates the widgets on the rerun.
    st.session_state["shortlist_cap_search_start_x"] = cap
    st.session_state["shortlist_cap_search_start_y"] = maximum_assets
    st.session_state["universal_preselection_cap"] = cap
    st.session_state["exact_optimizer_asset_cap"] = maximum_assets


def _search_results_are_compatible(previous_config, current_config):
    """Allow grid edits to reuse results from the same analysis snapshot."""
    analysis_keys = (
        "max_dd",
        "target_volatility",
        "drop_bottom_pct",
        "history_buffer_days",
        "redundancy_corr_threshold",
        "minimum_trading_days",
        "snapshot_date_ist",
        "optimizer_config_version",
        "risk_free_rate_annual",
    )
    if not previous_config or not current_config:
        return False
    return all(
        previous_config.get(key) == current_config.get(key)
        for key in analysis_keys
    )

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
with st.expander("🧾 Run diagnostics", expanded=False):
    st.caption(
        "The app records lifecycle, stage, timing, and error-type events only. "
        "It does not record holdings, credentials, or downloaded price data."
    )
    st.caption(f"Log file: `{APP_RUN_LOG_PATH}`")
    if APP_RUN_LOG_PATH.exists():
        log_bytes = APP_RUN_LOG_PATH.read_bytes()
        st.download_button(
            "Download run log",
            data=log_bytes,
            file_name="portfolio_rebalancer_run.log.jsonl",
            mime="application/x-ndjson",
            key="download_app_run_log",
        )
        st.caption(f"Recorded events: {len(log_bytes.splitlines()):,}")
    else:
        st.info("No run events have been recorded yet.")

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

checkpoint_restore_key = f"shortlist_checkpoint_restored::{CURRENT_USER}"
if not st.session_state.get(checkpoint_restore_key):
    with get_db_connection() as checkpoint_conn:
        restored_checkpoint = load_latest_checkpoint(checkpoint_conn, CURRENT_USER)
        saved_best_result = load_best_saved_result(checkpoint_conn, CURRENT_USER)
    if restored_checkpoint and restored_checkpoint.get("results"):
        restored_config = dict(restored_checkpoint.get("config") or {})
        # The visible results come from the latest checkpoint, so seed both warm-start
        # coordinates from that checkpoint's exact raw peak. Falling back to an older
        # saved peak is only appropriate when the restored result has no feasible row.
        restored_best = (
            best_feasible_result(restored_checkpoint["results"])
            or saved_best_result
        )
        restored_asset_caps = list(restored_config.get("exact_optimizer_asset_caps") or [])
        st.session_state.setdefault("advanced_shortlist_cap_search", True)
        st.session_state.setdefault("shortlist_cap_search_from", int(restored_config.get("minimum_cap") or restored_config.get("starting_cap") or DEFAULT_ADVANCED_CAP_FROM))
        st.session_state.setdefault("shortlist_cap_search_through", int(restored_config.get("maximum_cap") or DEFAULT_ADVANCED_CAP_THROUGH))
        st.session_state.setdefault("shortlist_cap_search_step", int(restored_config.get("step") or DEFAULT_ADVANCED_CAP_STEP))
        st.session_state["shortlist_cap_search_start_x"] = int(
            (restored_best or {}).get("Shortlist cap")
            or restored_config.get("starting_cap")
            or DEFAULT_ADVANCED_START_X
        )
        st.session_state["shortlist_cap_search_start_y"] = int(
            (restored_best or {}).get("Maximum assets")
            or restored_config.get("starting_asset_limit")
            or restored_config.get("exact_optimizer_asset_cap")
            or DEFAULT_ADVANCED_START_Y
        )
        st.session_state.setdefault(
            "shortlist_cap_search_spacing",
            "Adaptive hill-climb" if restored_config.get("adaptive") else "Fixed (complete 2D grid)",
        )
        if restored_asset_caps:
            st.session_state.setdefault("exact_asset_search_from", int(min(restored_asset_caps)))
            st.session_state.setdefault("exact_asset_search_through", int(max(restored_asset_caps)))
            if len(restored_asset_caps) > 1:
                restored_steps = [
                    right - left
                    for left, right in zip(sorted(restored_asset_caps)[:-1], sorted(restored_asset_caps)[1:])
                    if right > left
                ]
                st.session_state.setdefault("exact_asset_search_step", int(min(restored_steps) if restored_steps else 50))
        st.session_state.setdefault("exact_optimizer_asset_cap", int(restored_config.get("exact_optimizer_asset_cap") or 300))
        st.session_state.setdefault("shortlist_search_return_tolerance_pp", float(restored_config.get("return_equivalence_pp") or 0.10))
        st.session_state.setdefault("shortlist_search_skip_recent_best", True)
        st.session_state.setdefault("max_drawdown_input_pct", 100.0 * float(restored_config.get("max_dd") or 0.23))
        st.session_state.setdefault("manual_drop_bottom_pct_v8", float(restored_config.get("drop_bottom_pct") or 0.20))
        st.session_state.setdefault("redundancy_corr_threshold", float(restored_config.get("redundancy_corr_threshold") or 0.80))
        restored_target_volatility = restored_config.get("target_volatility")
        st.session_state.setdefault("use_target_volatility", restored_target_volatility is not None)
        if restored_target_volatility is not None:
            st.session_state.setdefault("target_volatility_input", float(restored_target_volatility))
        st.session_state.setdefault("auto_history_end", False)
        st.session_state.setdefault("history_buffer_days", int(restored_config.get("history_buffer_days") or 0))
        st.session_state["shortlist_cap_search_results"] = [dict(row) for row in restored_checkpoint["results"]]
        st.session_state["shortlist_cap_search_config"] = restored_config
        st.session_state["shortlist_cap_checkpoint_metadata"] = {
            key: restored_checkpoint.get(key)
            for key in ("status", "analysis_cutoff", "updated_at")
        }
        st.session_state["shortlist_cap_saved_best"] = dict(restored_best or {})
    st.session_state[checkpoint_restore_key] = True

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

pending_automatic_best_pair = st.session_state.pop(
    "pending_automatic_best_pair", None
)
if pending_automatic_best_pair:
    automatic_cap = int(pending_automatic_best_pair["cap"])
    automatic_assets = int(pending_automatic_best_pair["maximum_assets"])
    st.session_state["universal_preselection_cap"] = automatic_cap
    st.session_state["exact_optimizer_asset_cap"] = automatic_assets
    st.session_state["shortlist_cap_search_start_x"] = automatic_cap
    st.session_state["shortlist_cap_search_start_y"] = automatic_assets

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
    days_to_flip = st.number_input("Expected days to flip", min_value=1, value=1, step=1)
    security_forecast_horizon = st.number_input(
        "Security scenario horizon (trading sessions)",
        min_value=1,
        max_value=252,
        value=20,
        step=1,
        key="security_scenario_horizon_sessions",
        help=(
            "User-selected horizon for the historical scenario table only; it does not "
            "set an automatic rebalance date or promise a holding period."
        ),
    )
    security_forecast_cost_bps = st.number_input(
        "Scenario trading cost per side (basis points)",
        min_value=0.0,
        max_value=500.0,
        value=22.0,
        step=1.0,
        key="security_scenario_cost_bps",
        help=(
            "Applied once on entry and once on exit. This is a configurable proportional "
            "estimate, not a security- or broker-specific fee/tax calculation."
        ),
    )
    max_dd_pct = st.number_input(
        "Max drawdown input (%)",
        min_value=0.00,
        value=23.00,
        step=0.01,
        format="%.2f",
        key="max_drawdown_input_pct",
    )
    max_dd = (max_dd_pct / 100)
    st.caption(f"Internal max_dd used: {max_dd:.4f}")

    auto_history_end = st.checkbox(
        "Choose the history end date automatically",
        value=True,
        key="auto_history_end",
        help=(
            "Uses the day of week and the NSE close time: weekends roll back to Friday, "
            "and today is skipped only while its session is still open."
        ),
    )

    if auto_history_end:
        history_buffer_days, history_buffer_reason = auto_history_buffer_days()
    else:
        history_buffer_reason = ""
        history_buffer_days = int(
            st.number_input(
                "Exclude last N trading days of history",
                min_value=0,
                max_value=30,
                value=0,
                step=1,
                key="history_buffer_days",
                help=(
                    "Counted in trading days, not calendar days. 0 uses history up to and "
                    "including today."
                ),
            )
        )

    history_end_date = effective_history_end(None, history_buffer_days)[1]
    st.caption(
        f"History ends on or before: {history_end_date:%a, %d %b %Y} "
        "(exchange holidays may pull it earlier)"
    )
    if history_buffer_reason:
        st.caption(history_buffer_reason)

    drop_bottom_pct = float(
        st.number_input(
            "Drop bottom fraction of tickers by liquidity then history length",
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

    universal_preselection_cap = int(
        st.number_input(
            "Universal candidate shortlist cap",
            min_value=50,
            value=UNIVERSAL_PRESELECTION_CAP,
            step=50,
            key="universal_preselection_cap",
            help=(
                "Maximum Universal Portfolio candidates sent into full-history "
                "analysis before the existing liquidity, history, momentum and "
                "near-duplicate filters. Larger values increase runtime and memory use."
            ),
        )
    )

    exact_optimizer_asset_cap = int(
        st.number_input(
            "Maximum assets in exact optimization",
            min_value=100,
            value=DEFAULT_EXACT_OPTIMIZER_ASSET_CAP,
            step=50,
            key="exact_optimizer_asset_cap",
            help=(
                "The broad shortlist still discovers candidates. After all history, "
                "momentum and duplicate filters, only the strongest risk-adjusted "
                "candidates plus every owned holding enter the expensive constrained "
                "solver. Raise this only when you deliberately accept longer runs."
            ),
        )
    )

    redundancy_corr_threshold = float(
        st.number_input(
            "Merge assets correlated above",
            min_value=0.50,
            max_value=1.00,
            value=0.80,
            step=0.01,
            format="%.2f",
            key="redundancy_corr_threshold",
            help=(
                "Near-duplicate holdings such as eight gold ETFs are collapsed to the "
                "most liquid one, so the per-asset weight cap limits real exposure "
                "instead of one ticker. Correlation is measured on 5-day returns because "
                "stale daily closes hide duplicates. Set 1.00 to disable."
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

    use_target_vol = st.checkbox("Use target volatility", key="use_target_volatility")
    target_volatility = (
        st.number_input(
            "Target volatility",
            min_value=0.00696,
            value=0.01455,
            step=0.0001,
            format="%.5f",
            key="target_volatility_input",
        )
        if use_target_vol
        else None
    )

    advanced_cap_search = st.toggle(
        "Advanced shortlist-cap search",
        value=True,
        key="advanced_shortlist_cap_search",
        help=(
            "Searches shortlist capacity and the exact optimizer's Maximum-assets "
            "limit together. Fixed spacing evaluates the complete Cartesian grid; "
            "only feasible results compete on annual return."
        ),
    )
    if advanced_cap_search:
        cap_range_col1, cap_range_col2 = st.columns(2)
        with cap_range_col1:
            cap_search_from = int(
                st.number_input(
                    "Search cap from",
                    min_value=50,
                    value=DEFAULT_ADVANCED_CAP_FROM,
                    step=DEFAULT_ADVANCED_CAP_STEP,
                    key="shortlist_cap_search_from",
                )
            )
        with cap_range_col2:
            cap_search_through = int(
                st.number_input(
                    "Search cap through",
                    min_value=50,
                    value=max(DEFAULT_ADVANCED_CAP_THROUGH, cap_search_from),
                    step=DEFAULT_ADVANCED_CAP_STEP,
                    key="shortlist_cap_search_through",
                    help="No hidden maximum; the eligible universe remains the natural ceiling.",
                )
            )
        cap_search_through = max(cap_search_through, cap_search_from)
        cap_step_col, cap_spacing_col = st.columns(2)
        with cap_step_col:
            cap_search_step = int(
                st.number_input(
                    "Minimum search step",
                    min_value=50,
                    value=DEFAULT_ADVANCED_CAP_STEP,
                    step=DEFAULT_ADVANCED_CAP_STEP,
                    key="shortlist_cap_search_step",
                    help="Use 50 for the finest supported cap resolution.",
                )
            )
        with cap_spacing_col:
            cap_search_spacing = st.selectbox(
                "Search spacing",
                options=[
                    "Fixed (complete 2D grid)",
                    "Adaptive hill-climb",
                ],
                index=1,
                key="shortlist_cap_search_spacing",
                help=(
                    "Fixed evaluates every shortlist-cap step and every Maximum-assets "
                    "value. Adaptive uses broad anchors followed by local hill-climbing; "
                    "it is derivative-free because this discrete, noisy surface has no "
                    "reliable mathematical gradient."
                ),
            )
        asset_range_col1, asset_range_col2, asset_range_col3 = st.columns(3)
        with asset_range_col1:
            asset_search_from = int(
                st.number_input(
                    "Maximum assets from",
                    min_value=50,
                    value=DEFAULT_ADVANCED_ASSETS_FROM,
                    step=DEFAULT_ADVANCED_ASSETS_STEP,
                    key="exact_asset_search_from",
                )
            )
        with asset_range_col2:
            asset_search_through = int(
                st.number_input(
                    "Maximum assets through",
                    min_value=50,
                    value=max(DEFAULT_ADVANCED_ASSETS_THROUGH, asset_search_from),
                    step=DEFAULT_ADVANCED_ASSETS_STEP,
                    key="exact_asset_search_through",
                )
            )
        asset_search_through = max(asset_search_through, asset_search_from)
        with asset_range_col3:
            asset_search_step = int(
                st.number_input(
                    "Maximum-assets step",
                    min_value=50,
                    value=DEFAULT_ADVANCED_ASSETS_STEP,
                    step=DEFAULT_ADVANCED_ASSETS_STEP,
                    key="exact_asset_search_step",
                )
            )
        start_col1, start_col2 = st.columns(2)
        current_raw_peak = best_feasible_result(
            st.session_state.get("shortlist_cap_search_results") or []
        )
        if "shortlist_cap_search_start_x" not in st.session_state:
            st.session_state["shortlist_cap_search_start_x"] = int(
                (current_raw_peak or {}).get("Shortlist cap")
                or DEFAULT_ADVANCED_START_X
            )
        if "shortlist_cap_search_start_y" not in st.session_state:
            st.session_state["shortlist_cap_search_start_y"] = int(
                (current_raw_peak or {}).get("Maximum assets")
                or DEFAULT_ADVANCED_START_Y
            )
        current_start_x = int(st.session_state.get("shortlist_cap_search_start_x", DEFAULT_ADVANCED_START_X))
        current_start_y = int(st.session_state.get("shortlist_cap_search_start_y", DEFAULT_ADVANCED_START_Y))
        st.session_state["shortlist_cap_search_start_x"] = min(max(current_start_x, cap_search_from), cap_search_through)
        st.session_state["shortlist_cap_search_start_y"] = min(max(current_start_y, asset_search_from), asset_search_through)
        with start_col1:
            cap_search_start_x = int(st.number_input(
                "Start at shortlist cap (X)", min_value=int(cap_search_from),
                max_value=int(cap_search_through), step=int(cap_search_step),
                key="shortlist_cap_search_start_x",
                help="The first shortlist-cap coordinate. The configured range remains searchable on both sides.",
            ))
        with start_col2:
            cap_search_start_y = int(st.number_input(
                "Start at Maximum assets (Y)", min_value=int(asset_search_from),
                max_value=int(asset_search_through), step=int(asset_search_step),
                key="shortlist_cap_search_start_y",
                help="The first exact-optimizer coordinate. Use the previous best Y when warm-starting a new market snapshot.",
            ))
        return_equivalence_pp = float(
            st.number_input(
                "Near-equal return tolerance (percentage points)",
                min_value=0.0,
                value=0.10,
                step=0.05,
                format="%.2f",
                key="shortlist_search_return_tolerance_pp",
                help=(
                    "Pairs within this many annual-return percentage points of the "
                    "raw peak form a plateau. The robust choice then prefers lower "
                    "downside risk, volatility and complexity. This does not stop "
                    "the exhaustive search."
                ),
            )
        )
        skip_recent_best_search = st.checkbox(
            "Skip X/Y rediscovery for 24 hours after a full search",
            value=True,
            key="shortlist_search_skip_recent_best",
            help=(
                "On each rerun, re-evaluate the last fully searched best X/Y. If its "
                "annual return improves by more than the tolerance above, run the full "
                "adaptive search immediately. Otherwise skip X/Y rediscovery until "
                "24 hours have passed since the last full search. This is a "
                "time-saving heuristic, not proof of a global maximum."
            ),
        )
        validate_finalists = st.checkbox(
            "Require chronological finalist validation before auto-running a plan",
            value=True,
            key="shortlist_search_validate_finalists",
            help=(
            "After the adaptive search saturates, re-fit the top two feasible "
                "pairs on expanding historical windows, score later sessions after "
                "an estimated one-way-turnover cost, and auto-run only a consistent "
                "winner. Adds several optimizer solves and may take minutes."
            ),
        )
        if validate_finalists:
            finalist_transaction_cost_bps = float(st.number_input(
                "Estimated cost per one-way turnover (basis points)",
                min_value=0.0,
                max_value=500.0,
                value=22.0,
                step=1.0,
                key="shortlist_finalist_cost_bps",
                help=(
                    "A transparent sensitivity assumption, not a broker quote. "
                    "Set it to your estimated all-in execution cost; taxes, account "
                    "fees and market impact are not known by this app."
                ),
            ))
        else:
            finalist_transaction_cost_bps = 22.0
        st.caption(
            "On app open, X/Y are initialized once from the restored search's raw peak; "
            "manual edits then remain in control. Adaptive mode starts with a shared "
            "baseline and one neighboring probe per axis, so each coordinate line can "
            "fit a return slope from two points. It alternates X and Y and uses that slope "
            "to propose up to eight grid intervals per move, rather than walking one cell "
            "at a time. Each solver residual refits the line and adjusts the next move. "
            "Every completed pair is checkpointed for resume; fixed mode is exhaustive."
        )
    else:
        cap_search_from = int(universal_preselection_cap)
        cap_search_through = int(universal_preselection_cap)
        cap_search_step = 50
        cap_search_spacing = "Adaptive hill-climb"
        asset_search_from = int(exact_optimizer_asset_cap)
        asset_search_through = int(exact_optimizer_asset_cap)
        asset_search_step = 50
        return_equivalence_pp = 0.10
        skip_recent_best_search = False
        validate_finalists = False
        finalist_transaction_cost_bps = 22.0
        cap_search_start_x = int(cap_search_from)
        cap_search_start_y = int(asset_search_from)
    run_cap_search_btn = st.button(
        "Run / resume cap search",
        width="stretch",
        key="run_shortlist_cap_search_btn",
        disabled=not advanced_cap_search,
        help="This is a long-running research search and does not create trades.",
    )
    clear_cap_search_btn = st.button(
        "Clear cap-search results",
        width="stretch",
        key="clear_shortlist_cap_search_btn",
        disabled=not bool(st.session_state.get("shortlist_cap_search_results")),
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
            "Upload Groww/Indian or Tickertape US holdings",
            type=["xlsx", "xls", "csv"],
            key="broker_holdings_upload",
            help=(
                "Groww/Indian CSV or Excel statements are detected and matched by ISIN. "
                "Tickertape US reports are detected from explicit USD-labelled columns. "
                "Because Tickertape omits symbols, US names are matched to Yahoo only "
                "when the result is unambiguous; unresolved names are reported."
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
            value=10000.0,
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
            on_click=_start_optimization_plan_timer,
            args=(CURRENT_USER,),
        )
        optimization_timer_placeholder = st.empty()
        timer_summary = _timer_summary(
            st.session_state.get(OPTIMIZATION_TIMER_KEY)
        )
        if timer_summary:
            timer_state, timer_text = timer_summary
            if timer_state == "finished":
                optimization_timer_placeholder.success(timer_text)
            elif timer_state == "stopped":
                optimization_timer_placeholder.warning(timer_text)
            else:
                optimization_timer_placeholder.info(timer_text)

st.divider()
if clear_cap_search_btn:
    with get_db_connection() as checkpoint_conn:
        delete_checkpoints(checkpoint_conn, CURRENT_USER)
    st.session_state.pop("shortlist_cap_search_results", None)
    st.session_state.pop("shortlist_cap_search_config", None)
    st.session_state.pop("shortlist_cap_search_error", None)
    st.session_state.pop("shortlist_cap_checkpoint_metadata", None)
    st.session_state.pop("shortlist_cap_saved_best", None)
    st.session_state.pop("shortlist_cap_refinement_summary", None)
    st.session_state.pop("shortlist_cap_refinement_error", None)
    st.session_state.pop("shortlist_cap_walk_forward_validation", None)
    st.rerun()

exact_asset_search_values = inclusive_values(
    asset_search_from,
    asset_search_through,
    asset_search_step,
    minimum_step=50,
)
cap_search_values = inclusive_values(
    cap_search_from,
    cap_search_through,
    cap_search_step,
    minimum_step=50,
)
cap_search_config = {
    "minimum_cap": int(cap_search_from),
    "starting_cap": int(cap_search_start_x),
    "maximum_cap": int(cap_search_through),
    "step": int(cap_search_step),
    "adaptive": cap_search_spacing.startswith("Adaptive"),
    "asset_evaluation_order": "coarse_to_fine",
    "max_dd": float(max_dd),
    "target_volatility": (
        float(target_volatility) if target_volatility is not None else None
    ),
    "drop_bottom_pct": float(drop_bottom_pct),
    "history_buffer_days": int(history_buffer_days),
    "redundancy_corr_threshold": float(redundancy_corr_threshold),
    "exact_optimizer_asset_cap": int(exact_optimizer_asset_cap),
    "exact_optimizer_asset_caps": list(exact_asset_search_values),
    "starting_asset_limit": int(cap_search_start_y),
    "minimum_trading_days": 252,
    "return_equivalence_pp": float(return_equivalence_pp),
    "skip_recent_best_search": bool(skip_recent_best_search),
    "optimizer_config_version": OPTIMIZER_CONFIG_VERSION,
    "risk_free_rate_annual": float(RISK_FREE_RATE_ANNUAL),
    "snapshot_date_ist": datetime.now(
        timezone(timedelta(hours=5, minutes=30))
    ).date().isoformat(),
}

if advanced_cap_search:
    st.caption(
        f"Configured search space: **{len(cap_search_values):,} shortlist caps × "
        f"{len(exact_asset_search_values):,} Maximum-assets values = "
        f"{len(cap_search_values) * len(exact_asset_search_values):,} combinations**."
    )
    st.caption(
        f"Saved-best/warm-start pair: **X {cap_search_start_x:,} × "
        f"Y {cap_search_start_y:,}**. Adaptive search probes the lower-bound "
        "X/Y neighborhood first, then includes this pair as a seed."
    )
    checkpoint_metadata = st.session_state.get("shortlist_cap_checkpoint_metadata")
    if checkpoint_metadata:
        restored_rows = st.session_state.get("shortlist_cap_search_results") or []
        feasible_restored = [row for row in restored_rows if row.get("Status") == "Feasible"]
        restored_best = max(
            feasible_restored,
            key=lambda row: float(row.get("Annual Return") or -np.inf),
            default=None,
        )
        best_text = (
            f" Best saved pair: **{int(restored_best['Shortlist cap']):,} × "
            f"{int(restored_best['Maximum assets']):,}** at "
            f"**{float(restored_best['Annual Return']):.2%}**."
            if restored_best is not None else ""
        )
        st.info(
            f"Restored **{len(restored_rows):,} completed X/Y pairs** from disk"
            f" (saved {checkpoint_metadata.get('updated_at')}).{best_text} "
            "Run / resume continues only when today's configuration and snapshot match."
        )

automatic_prepared_analysis = None
if run_cap_search_btn:
    _app_run_log(
        "search_started",
        user=bool(CURRENT_USER),
        cap_from=int(cap_search_from),
        cap_through=int(cap_search_through),
        cap_step=int(cap_search_step),
        assets_from=int(asset_search_from),
        assets_through=int(asset_search_through),
        assets_step=int(asset_search_step),
        spacing=str(cap_search_spacing),
        skip_recent_best=bool(skip_recent_best_search),
    )
    previous_config = st.session_state.get("shortlist_cap_search_config") or {}
    existing_results = st.session_state.get("shortlist_cap_search_results") or []
    prior_results = (
        existing_results
        if (
            previous_config == cap_search_config
            or _search_results_are_compatible(previous_config, cap_search_config)
        )
        else []
    )
    st.session_state["shortlist_cap_search_config"] = dict(cap_search_config)
    st.session_state.pop("shortlist_cap_search_error", None)
    st.session_state.pop("shortlist_cap_refinement_summary", None)
    st.session_state.pop("shortlist_cap_refinement_error", None)
    st.session_state.pop("shortlist_cap_walk_forward_validation", None)

    st.subheader("Advanced shortlist-cap search")
    search_status = st.status(
        "Preparing the first shortlist-cap evaluation…",
        expanded=True,
    )
    search_progress = st.progress(0.0)
    search_detail = st.empty()
    live_running_pair = st.empty()
    st.markdown("**Live X/Y search map**")
    live_search_best = st.empty()
    live_search_chart = st.empty()
    live_chart_config = {
        "displaylogo": False,
        "scrollZoom": False,
        "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
    }

    def render_live_search_map(rows, proposal=None):
        current_best = best_feasible_result(rows)
        if current_best is None:
            live_search_best.info(
                "Waiting for the first feasible pair. Blank cells have not been evaluated."
            )
        else:
            live_search_best.success(
                "Highest return found so far: **"
                f"{int(current_best['Shortlist cap']):,} × "
                f"{int(current_best['Maximum assets']):,}** · "
                f"**{float(current_best['Annual Return']):.2%} annual return**."
            )
        live_figure = build_search_heatmap_figure(
            rows,
            shortlist_caps=cap_search_values,
            maximum_assets_values=exact_asset_search_values,
            best_so_far=current_best,
            next_proposal=proposal,
            live=True,
        )
        live_search_chart.plotly_chart(
            live_figure,
            width="stretch",
            height=520,
            theme=None,
            key=(
                f"optimizer_search_heatmap_live_{len(rows)}_"
                f"{proposal['x']}_{proposal['y']}"
                if proposal else f"optimizer_search_heatmap_live_{len(rows)}"
            ),
            config=live_chart_config,
        )

    render_live_search_map(prior_results)
    search_started_at = time.monotonic()
    validation_matrix_cache = {}

    def update_cap_search_stage(stage, fraction, details=None):
        details = details or {}
        cap_value = details.get("cap")
        cap_text = f"Cap {int(cap_value):,} · " if cap_value is not None else ""
        suffix = ""
        if details.get("batches"):
            current_batch = int(details.get("batch") or 0)
            total_batches = int(details["batches"])
            suffix = (
                f" · batch {current_batch:,}/{total_batches:,}"
                if current_batch
                else f" · {total_batches:,} batches planned"
            )
        elif details.get("shortlisted") is not None:
            suffix = f" · {int(details['shortlisted']):,} shortlisted"
        elif details.get("surviving_symbols") is not None:
            suffix = f" · {int(details['surviving_symbols']):,} symbols remain"
        elif details.get("aligned_assets") is not None:
            suffix = f" · {int(details['aligned_assets']):,} aligned assets"
        elif details.get("eligible_assets") is not None:
            suffix = f" · {int(details['eligible_assets']):,} optimizer-eligible"
        elif details.get("assets") is not None:
            suffix = f" · {int(details['assets']):,} assets"

        elapsed = float(details.get("elapsed_seconds") or 0.0)
        if elapsed <= 0:
            elapsed = time.monotonic() - search_started_at
        search_progress.progress(float(np.clip(fraction, 0.0, 1.0)))
        search_status.update(
            label=f"{cap_text}{stage}{suffix}",
            state="running",
            expanded=True,
        )
        next_probe = details.get("next_probe")
        running_pair = details.get("running_pair") or next_probe
        if running_pair:
            if next_probe:
                render_live_search_map(
                    st.session_state.get("shortlist_cap_search_results") or prior_results,
                    proposal=next_probe,
                )
            live_running_pair.markdown(
                "<style>"
                "@keyframes optimizer-pair-pulse {"
                "0%,100% {opacity:1; box-shadow:0 0 0 0 rgba(99,230,255,.48)}"
                "50% {opacity:.48; box-shadow:0 0 0 7px rgba(99,230,255,0)}"
                "}"
                ".optimizer-pair-live {animation:optimizer-pair-pulse 1.15s ease-in-out infinite;"
                "display:inline-block; padding:10px 16px; border-radius:8px;"
                "border:1px solid #63e6ff; color:#dffaff; background:#123044; font-weight:700;}"
                "</style>"
                f"<div class='optimizer-pair-live'>● RUNNING · "
                f"X {int(running_pair['x']):,} × Y {int(running_pair['y']):,}</div>",
                unsafe_allow_html=True,
            )
            if next_probe:
                search_detail.caption(
                    f"LR direction: {next_probe['axis']} · "
                    f"{next_probe['delta_steps']} grid interval(s) toward "
                    f"{int(next_probe['x']):,} × {int(next_probe['y']):,} · "
                    f"search elapsed {format_elapsed(elapsed)}"
                )
            else:
                search_detail.caption(
                    f"Running initial seed X {int(running_pair['x']):,} × "
                    f"Y {int(running_pair['y']):,} · "
                    f"search elapsed {format_elapsed(elapsed)}"
                )
        else:
            search_detail.caption(
                f"Current-cap progress {float(fraction):.0%} · "
                f"search elapsed {format_elapsed(elapsed)}"
            )

    def update_cap_search_progress(rows, latest):
        st.session_state["shortlist_cap_search_results"] = [dict(row) for row in rows]
        live_running_pair.empty()
        with get_db_connection() as checkpoint_conn:
            save_checkpoint(checkpoint_conn, CURRENT_USER, cap_search_config, rows, status="running")
        render_live_search_map(rows)
        current_cap = int(latest["Shortlist cap"])
        current_asset_limit = int(latest.get("Maximum assets") or 0)
        search_progress.progress(1.0)
        average_seconds = float(
            np.mean([float(row.get("Elapsed seconds") or 0.0) for row in rows])
        )
        search_detail.info(
            f"Completed pair **{current_cap:,} × {current_asset_limit:,}** · "
            f"{int(latest['Trading days']):,} trading sessions · "
            f"{int(latest['Assets']):,} assets · "
            f"{float(latest['Annual Return']):.2%} annual return · "
            f"latest **{format_elapsed(float(latest.get('Elapsed seconds') or 0.0))}** · "
            f"average solver result **{format_elapsed(average_seconds)}**"
            + (
                " · **Adaptive cap-spacing brake applied after the 15-minute cap budget**"
                if cap_search_spacing.startswith("Adaptive")
                and latest.get("Runtime brake") == "Applied"
                else ""
            )
        )
        search_status.update(
            label=(
                f"Evaluated {len(rows):,} X/Y pairs; "
                + (
                    f"stop reached at {current_cap:,}"
                    if latest.get("Status") != "Feasible"
                    else (
                        (
                            (
                                "solver-vs-linear-fit residual; adapting step and direction"
                                if "linear-fit" in str(latest.get("Search phase", "")).lower()
                                else "lower-bound seed points, then alternating adaptive X/Y refinement"
                                if "lower-bound seed" in str(latest.get("Search phase", "")).lower()
                                else "alternating X/Y gradient from the warm start"
                                if "gradient" in str(latest.get("Search phase", "")).lower()
                                else (
                                    f"next search jump +{int(latest.get('Next adaptive jump') or 50):,}"
                                    if cap_search_spacing.startswith("Adaptive")
                                    else "continuing complete grid"
                                )
                            )
                        )
                    )
                )
            ),
            state="running",
        )

    try:
        search_kwargs = {
            "max_dd": max_dd,
            "target_volatility": target_volatility,
            "drop_bottom_pct": drop_bottom_pct,
            "history_buffer_days": history_buffer_days,
            "redundancy_corr_threshold": redundancy_corr_threshold,
            "exact_optimizer_asset_cap": exact_optimizer_asset_cap,
            "exact_optimizer_asset_caps": exact_asset_search_values,
            "minimum_trading_days": 252,
            "step": cap_search_step,
            "maximum_cap": cap_search_through,
            "adaptive": cap_search_spacing.startswith("Adaptive"),
            "asset_evaluation_order": "coarse_to_fine",
            "gradient_return_scale": max(float(return_equivalence_pp) / 100.0, 1e-6),
            "prepared_matrix_store": validation_matrix_cache,
            "on_result": update_cap_search_progress,
            "on_progress": update_cap_search_stage,
        }
        skipper_reused_pair = False
        skipper_context = None
        today_session = str(cap_search_config["snapshot_date_ist"])
        if skip_recent_best_search and cap_search_spacing.startswith("Adaptive"):
            with get_db_connection() as checkpoint_conn:
                recent_checkpoint = load_latest_checkpoint_for_research_scope(
                    checkpoint_conn, CURRENT_USER, cap_search_config
                )
            saved_context = next(
                (
                    row.get("Search skipper context")
                    for row in reversed((recent_checkpoint or {}).get("results") or [])
                    if isinstance(row.get("Search skipper context"), dict)
                ),
                None,
            )
            decision = skipper_decision(
                saved_context,
                fallback_discovery_at=(recent_checkpoint or {}).get("updated_at"),
                maximum_age_hours=24,
            )
            saved_pair_row = next(
                (
                    row for row in (recent_checkpoint or {}).get("results") or []
                    if row.get("Status") == "Feasible"
                    and int(row.get("Shortlist cap") or -1) == int((decision or {}).get("best_x", -2))
                    and int(row.get("Maximum assets") or -1) == int((decision or {}).get("best_y", -2))
                ),
                None,
            )
            if decision and (
                int(cap_search_from) <= decision["best_x"] <= int(cap_search_through)
                and int(asset_search_from) <= decision["best_y"] <= int(asset_search_through)
            ) and saved_pair_row is not None:
                st.info(
                    f"The saved X/Y search is {decision['age_hours']:.1f} hours old. "
                    f"Skipping search and pair re-scoring; the normal rebalance run "
                    f"will optimize **{decision['best_x']:,} × {decision['best_y']:,}** "
                    "once using current data and refresh the 20-session analysis."
                )
                search_rows = [
                    dict(row)
                    for row in (recent_checkpoint or {}).get("results") or []
                ]
                saved_pair_row = next(
                    row for row in search_rows
                    if row.get("Status") == "Feasible"
                    and int(row.get("Shortlist cap") or -1) == decision["best_x"]
                    and int(row.get("Maximum assets") or -1) == decision["best_y"]
                )
                saved_pair_row["Search skipper status"] = (
                    "Recent full search (<24 hours): X/Y discovery and pair re-scoring "
                    "skipped; normal rebalance optimization ran on current data."
                )
                highest_return_pair = dict(saved_pair_row)
                skipper_context = dict(saved_context)
                skipper_reused_pair = True
                search_invalid_rows = []
            else:
                search_rows, search_invalid_rows = search_universal_shortlist_caps(
                    CURRENT_USER,
                    starting_cap=cap_search_start_x,
                    minimum_cap=cap_search_from,
                    starting_asset_limit=cap_search_start_y,
                    prior_results=prior_results,
                    **search_kwargs,
                )
        else:
            search_rows, search_invalid_rows = search_universal_shortlist_caps(
                CURRENT_USER,
                starting_cap=cap_search_start_x,
                minimum_cap=cap_search_from,
                starting_asset_limit=cap_search_start_y,
                prior_results=prior_results,
                **search_kwargs,
            )

        saturation_reached = bool(
            search_rows and search_rows[-1].get("Search saturated")
        )
        if not skipper_reused_pair:
            highest_return_pair = best_feasible_result(search_rows)
        if (
            not skipper_reused_pair
            and skip_recent_best_search
            and cap_search_spacing.startswith("Adaptive")
            and saturation_reached
            and highest_return_pair is not None
        ):
            for row in search_rows:
                row.pop("Search skipper context", None)
                row.pop("Search skipper status", None)
            latest_return = float(highest_return_pair["Annual Return"])
            highest_return_pair["Search skipper context"] = {
                "last_discovery_session": today_session,
                "last_discovery_at_utc": datetime.now(timezone.utc).isoformat(),
                "last_checked_session": today_session,
                "best_x": int(highest_return_pair["Shortlist cap"]),
                "best_y": int(highest_return_pair["Maximum assets"]),
                "last_checked_return": latest_return,
                "full_search_return": latest_return,
                "flat_reruns": 0,
            }
        st.session_state["shortlist_cap_search_results"] = search_rows
        with get_db_connection() as checkpoint_conn:
            save_checkpoint(checkpoint_conn, CURRENT_USER, cap_search_config, search_rows, status="complete")
        search_progress.progress(1.0)
        search_status.update(
            label=(
                "Recent full search reused; X/Y search skipped"
                if skipper_reused_pair
                else (
                    "Derivative-free adaptive scan completed"
                    if cap_search_spacing.startswith("Adaptive")
                    else "Complete shortlist-cap grid search completed"
                )
            ),
            state="complete",
            expanded=False,
        )
        if search_invalid_rows:
            st.warning(
                "Holdings skipped during the search: " + ", ".join(search_invalid_rows)
            )
        if (
            cap_search_spacing.startswith("Adaptive")
            and not skipper_reused_pair
            and saturation_reached
            and highest_return_pair is not None
        ):
            selected_pair = highest_return_pair
            validation_result = None
            if validate_finalists:
                validation_status = st.status(
                    "Checking leading pairs on later chronological periods…",
                    expanded=True,
                )
                validation_progress = st.progress(0.0)
                validation_detail = st.empty()

                def update_validation_progress(message, fraction):
                    validation_status.update(label=message, state="running")
                    validation_progress.progress(float(np.clip(fraction, 0.0, 1.0)))
                    validation_detail.caption(
                        "Reusing prepared optimizer matrices—no broad history re-download; "
                        "at most six fold solves with bounded solver attempts."
                    )

                try:
                    validation_result = validate_search_finalists_walk_forward(
                        CURRENT_USER,
                        search_rows,
                        max_dd=max_dd,
                        target_volatility=target_volatility,
                        drop_bottom_pct=drop_bottom_pct,
                        history_buffer_days=history_buffer_days,
                        redundancy_corr_threshold=redundancy_corr_threshold,
                        transaction_cost_bps=finalist_transaction_cost_bps,
                        finalist_count=2,
                        solver_timeout_seconds=60,
                        total_timeout_seconds=900,
                        prepared_matrices=validation_matrix_cache,
                        progress_callback=update_validation_progress,
                    )
                    st.session_state[
                        "shortlist_cap_walk_forward_validation"
                    ] = validation_result
                    validation_progress.progress(1.0)
                    if validation_result.get("winner"):
                        winner = validation_result["winner"]
                        winner_key = (
                            int(winner["Shortlist cap"]),
                            int(winner["Maximum assets"]),
                        )
                        selected_pair = next(
                            row for row in search_rows
                            if row.get("Status") == "Feasible"
                            and (int(row["Shortlist cap"]), int(row["Maximum assets"])) == winner_key
                        )
                        validation_status.update(
                            label="Chronological finalist check passed",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        selected_pair = None
                        validation_status.update(
                            label="No finalist passed the chronological consistency gate",
                            state="error",
                            expanded=True,
                        )
                except Exception as validation_exc:
                    selected_pair = None
                    st.session_state[
                        "shortlist_cap_walk_forward_validation"
                    ] = {
                        "status": "error",
                        "error": safe_stop_reason(validation_exc),
                        "finalists": [],
                        "winner": None,
                    }
                    validation_status.update(
                        label="Chronological finalist validation stopped",
                        state="error",
                        expanded=True,
                    )
                    st.error(
                        "No automatic plan was started because finalist validation "
                        f"could not complete: {safe_stop_reason(validation_exc)}"
                    )
            if selected_pair is not None:
                universal_preselection_cap = int(selected_pair["Shortlist cap"])
                exact_optimizer_asset_cap = int(selected_pair["Maximum assets"])
                automatic_prepared_analysis = validation_matrix_cache.get(
                    universal_preselection_cap
                )
                _start_optimization_plan_timer(CURRENT_USER)
                run_btn = True
                handoff_prefix = (
                    "Recent best pair rechecked with no significant gain; "
                    "X/Y rediscovery was skipped. Starting the final rebalancing "
                    "optimization with the "
                    if skipper_reused_pair
                    else "Adaptive search saturated. Starting the final rebalancing "
                    "optimization automatically with the "
                )
                st.success(
                    handoff_prefix +
                    f"{'chronologically validated' if validate_finalists else 'highest-return'} pair "
                    f"**{universal_preselection_cap:,} × "
                    f"{exact_optimizer_asset_cap:,}**. Cached histories and the "
                    "analytical solver are reused."
                )
            elif validate_finalists:
                st.warning(
                    "Search completed, but the final plan was not auto-started: "
                    "no finalist passed the chronological consistency gate. "
                    "Review the fold results below, or disable the gate to use "
                    "the raw full-history peak."
                )
        if skipper_reused_pair:
            universal_preselection_cap = int(skipper_context["best_x"])
            exact_optimizer_asset_cap = int(skipper_context["best_y"])
            _start_optimization_plan_timer(CURRENT_USER)
            run_btn = True
            st.success(
                "Skipped redundant X/Y discovery and pair re-scoring. Starting the "
                "normal current-data rebalance and 20-session analysis at the saved "
                f"pair **{universal_preselection_cap:,} × "
                f"{exact_optimizer_asset_cap:,}**."
            )
        _app_run_log(
            "search_finished",
            status="completed",
            result_count=len(st.session_state.get("shortlist_cap_search_results") or []),
            skipped=bool(skipper_reused_pair),
            checkpoint_count=len(st.session_state.get("shortlist_cap_search_results") or []),
            convergence_status=(
                "converged"
                if len(st.session_state.get("shortlist_cap_search_results") or [])
                else "no_results"
            ),
            selected_x=(
                int((st.session_state.get("shortlist_cap_saved_best") or {}).get("Shortlist cap"))
                if (st.session_state.get("shortlist_cap_saved_best") or {}).get("Shortlist cap") is not None
                else None
            ),
            selected_y=(
                int((st.session_state.get("shortlist_cap_saved_best") or {}).get("Maximum assets"))
                if (st.session_state.get("shortlist_cap_saved_best") or {}).get("Maximum assets") is not None
                else None
            ),
        )
    except Exception as exc:
        st.session_state["shortlist_cap_search_error"] = safe_stop_reason(exc)
        _app_run_log("search_finished", status="error", error_type=type(exc).__name__)
        search_status.update(
            label="Shortlist-cap search stopped",
            state="error",
            expanded=True,
        )

cap_search_error = st.session_state.get("shortlist_cap_search_error")
if cap_search_error:
    st.error(f"Cap search stopped: {cap_search_error}")

cap_search_results = st.session_state.get("shortlist_cap_search_results") or []
if cap_search_results:
    st.subheader("Shortlist-cap search results")
    skipper_note = next(
        (
            row.get("Search skipper status")
            for row in reversed(cap_search_results)
            if row.get("Search skipper status")
        ),
        None,
    )
    if skipper_note:
        st.info(
            skipper_note
            + " The pair was re-evaluated on the latest history; older search rows "
            "remain labeled as previous-run research. This heuristic does not prove "
            "that the saved pair is the global maximum."
        )
    walk_forward_result = st.session_state.get(
        "shortlist_cap_walk_forward_validation"
    )
    if walk_forward_result:
        st.markdown("**Chronological finalist check**")
        if walk_forward_result.get("error"):
            st.error(f"Validation stopped: {walk_forward_result['error']}")
        elif walk_forward_result.get("finalists"):
            st.caption(
                "Up to two finalists are scored across three expanding training windows. "
                "Each fold is scored on the immediately following sessions. Net annual "
                f"{float(walk_forward_result.get('transaction_cost_bps', 22.0)):.0f} bp "
                "per one-way turnover. This is a finalist stability check, not an "
                "unbiased point-in-time backtest: finalists and the exact optimizer "
                "screen were formed using full-history data, and the current candidate "
                "universe is reused."
            )
            if int(walk_forward_result.get("pairs_without_reusable_matrix", 0)):
                st.caption(
                    f"Skipped {int(walk_forward_result['pairs_without_reusable_matrix']):,} "
                    "older feasible pair(s) without a prepared matrix; validating them "
                    "would repeat the expensive broad-history scan."
                )
            validation_view = []
            for validation_row in walk_forward_result["finalists"]:
                fold_text = " · ".join(
                    f"{value:.1%}"
                    for value in validation_row.get("Fold net annual returns", [])
                )
                validation_view.append({
                    "Pair (X × Y)": (
                        f"{int(validation_row['Shortlist cap']):,} × "
                        f"{int(validation_row['Maximum assets']):,}"
                    ),
                    "Full-history return": (
                        f"{validation_row['Full-history annual return']:.2%}"
                        if pd.notna(validation_row.get("Full-history annual return")) else "—"
                    ),
                    "Fold net annual returns": fold_text or "—",
                    "Median net": (
                        f"{validation_row['Median net annual return']:.2%}"
                        if pd.notna(validation_row.get("Median net annual return")) else "—"
                    ),
                    "Worst fold": (
                        f"{validation_row['Worst fold net annual return']:.2%}"
                        if pd.notna(validation_row.get("Worst fold net annual return")) else "—"
                    ),
                    "Positive folds": validation_row.get("Positive folds", 0),
                    "Gate": validation_row.get("Status"),
                })
            st.dataframe(pd.DataFrame(validation_view), width="stretch", hide_index=True)
            if walk_forward_result.get("winner"):
                winner = walk_forward_result["winner"]
                st.success(
                    "Selected after chronological validation: **"
                    f"{int(winner['Shortlist cap']):,} × "
                    f"{int(winner['Maximum assets']):,}**, median net annual "
                    f"return **{float(winner['Median net annual return']):.2%}**."
                )
            elif walk_forward_result.get("status") == "no_finalist_passed":
                st.warning(
                    "No finalist met the gate (three usable folds and at least two "
                    "positive net-return folds). No automatic pair was selected."
                )
            elif walk_forward_result.get("status") == "no_reusable_matrices":
                st.warning(
                    "No matrix from the completed search was available for validation. "
                    "To avoid re-downloading the broad universe, validation did not "
                    "start. Run a fresh search evaluation to validate its finalist."
                )
    results_df = pd.DataFrame(cap_search_results)
    if "Maximum assets" not in results_df.columns:
        results_df["Maximum assets"] = int(exact_optimizer_asset_cap)
    if "Effective maximum assets" not in results_df.columns:
        results_df["Effective maximum assets"] = results_df.get(
            "Assets", int(exact_optimizer_asset_cap)
        )
    if "Cap elapsed seconds" not in results_df.columns:
        results_df["Cap elapsed seconds"] = results_df.get("Elapsed seconds", 0.0)
    feasible_df = results_df.loc[results_df["Status"].eq("Feasible")].copy()
    if feasible_df.empty:
        st.warning("No evaluated grid combination produced a feasible solution.")
    else:
        result_records = results_df.to_dict(orient="records")
        return_tolerance = max(float(return_equivalence_pp), 0.0) / 100.0
        best_record = best_feasible_result(result_records)
        best_row = pd.Series(best_record)
        robust_row = pd.Series(
            robust_feasible_result(
                result_records,
                return_tolerance=return_tolerance,
            )
        )
        convergence = search_convergence_summary(
            result_records,
            cap_search_values,
            exact_asset_search_values,
            return_tolerance=return_tolerance,
        )
        off_grid_completed = int(convergence.get("off_grid_completed") or 0)
        if off_grid_completed:
            st.caption(
                f"Preserved {off_grid_completed:,} completed pair(s) from an earlier "
                "grid. They remain visible and eligible as saved research, while "
                "convergence is measured only against the current step and bounds."
            )
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        metric_col1.metric("Raw-peak cap", f"{int(best_row['Shortlist cap']):,}")
        metric_col2.metric(
            "Raw-peak Maximum assets", f"{int(best_row['Maximum assets']):,}"
        )
        metric_col3.metric(
            "Raw annual return", f"{float(best_row['Annual Return']):.2%}"
        )
        metric_col4.metric("Trading days", f"{int(best_row['Trading days']):,}")
        metric_col5.metric("Assets solved", f"{int(best_row['Assets']):,}")
        robust_return_gap_pp = 100.0 * (
            float(best_row["Annual Return"]) - float(robust_row["Annual Return"])
        )
        st.success(
            f"Robust plateau choice: **{int(robust_row['Shortlist cap']):,} × "
            f"{int(robust_row['Maximum assets']):,}** at "
            f"**{float(robust_row['Annual Return']):.2%}** annual return. It gives up "
            f"**{robust_return_gap_pp:.2f} percentage points** versus the raw peak "
            f"within the configured {return_equivalence_pp:.2f}-point tolerance."
        )
        convergence_state = str(convergence["state"])
        if skipper_note:
            st.info(
                "X/Y rediscovery was intentionally skipped under the recent-search "
                "reuse rule. The displayed pair results are retained research, not "
                "a fresh full-grid convergence check; global optimality is unknown."
            )
        elif convergence_state == "global_verified":
            boundary_note = (
                " The winner is on the " + " and ".join(convergence["boundary_axes"])
                + " boundary, so it is global only inside the configured bounds."
                if convergence["boundary_axes"] else ""
            )
            st.success(
                "Bounded global maximum verified: every configured pair was evaluated."
                + boundary_note
            )
        elif convergence_state == "locally_converged":
            st.success(
                "Locally converged robust maximum: every minimum-step neighbour around "
                "the observed leader was evaluated. This is not an exhaustive global proof."
            )
        elif convergence_state == "incomplete":
            st.info(
                f"Search is not converged: {int(convergence['completed']):,} of "
                f"{int(convergence['total']):,} configured pairs have results."
            )
        elif convergence_state == "boundary_limited":
            st.warning(
                "The completed grid's raw peak lies on the "
                + " and ".join(convergence["boundary_axes"])
                + " boundary. It is the best observed point, but the maximum is not "
                "bounded on that side."
            )
        elif convergence_state == "no_feasible_result":
            st.info(
                "Saved results are preserved, but none of the completed pairs lies "
                "on the currently configured X/Y grid yet. Run/resume will evaluate "
                "the current grid without deleting the earlier research."
            )
        apply_col1, apply_col2 = st.columns(2)
        apply_col1.button(
            "Use robust plateau pair",
            type="primary",
            key="apply_robust_shortlist_cap_btn",
            on_click=_apply_best_search_pair,
            args=(
                int(robust_row["Shortlist cap"]),
                int(robust_row["Maximum assets"]),
            ),
        )
        apply_col2.button(
            "Use raw-return peak",
            key="apply_best_shortlist_cap_btn",
            on_click=_apply_best_search_pair,
            args=(
                int(best_row["Shortlist cap"]),
                int(best_row["Maximum assets"]),
            ),
        )

        saved_search_config = (
            st.session_state.get("shortlist_cap_search_config") or cap_search_config
        )
        refinement_minimum = int(saved_search_config.get("starting_cap", 50))
        refinement_maximum = int(
            saved_search_config.get("maximum_cap", refinement_minimum)
        )
        refinement_step = max(int(saved_search_config.get("step", 50)), 50)
        next_refinement_pair = local_refinement_cap_pair(
            cap_search_results,
            minimum_cap=refinement_minimum,
            maximum_cap=refinement_maximum,
            minimum_step=refinement_step,
        )
        st.caption(
            "Optional local refinement tests the untried interval on both sides of "
            "the best cap. It evaluates at most four additional caps, will not start "
            "another cap after 60 minutes, and stops after a pair unless annual return "
            "improves by at least 1 percentage point without 20-session expected "
            "shortfall worsening by more than 0.5 percentage point."
        )
        refine_cap_search_btn = st.button(
            "Refine around best cap",
            key="refine_best_shortlist_cap_btn",
            disabled=not bool(next_refinement_pair),
            help=(
                "Uses cached market history where available. Disabled when every "
                "neighboring interval is already resolved at the selected minimum step."
            ),
        )

        if refine_cap_search_btn:
            refinement_started = time.monotonic()
            refinement_budget_seconds = 60 * 60
            maximum_new_caps = 4
            combined_rows = [dict(row) for row in cap_search_results]
            for row in combined_rows:
                row.setdefault("Search phase", "Broad")
            baseline_row = dict(best_row)
            evaluated_rows = []
            stop_reason = "Neighboring intervals resolved at the minimum step"
            refinement_status = st.status(
                "Preparing bounded local refinement…",
                expanded=True,
            )
            refinement_progress = st.progress(0.0)
            refinement_detail = st.empty()

            try:
                while len(evaluated_rows) < maximum_new_caps:
                    pair = local_refinement_cap_pair(
                        combined_rows,
                        minimum_cap=refinement_minimum,
                        maximum_cap=refinement_maximum,
                        minimum_step=refinement_step,
                    )
                    remaining_slots = maximum_new_caps - len(evaluated_rows)
                    pair = pair[:remaining_slots]
                    if not pair:
                        break

                    pair_rows = []
                    for candidate_cap in pair:
                        elapsed = time.monotonic() - refinement_started
                        if elapsed >= refinement_budget_seconds:
                            stop_reason = "60-minute budget reached before another cap began"
                            break
                        evaluation_number = len(evaluated_rows) + 1

                        def update_refinement_stage(stage, fraction, details=None):
                            details = details or {}
                            total_fraction = (
                                len(evaluated_rows) + float(np.clip(fraction, 0.0, 1.0))
                            ) / maximum_new_caps
                            refinement_progress.progress(min(total_fraction, 1.0))
                            suffix = ""
                            if details.get("batches"):
                                current_batch = int(details.get("batch") or 0)
                                total_batches = int(details["batches"])
                                suffix = (
                                    f" · batch {current_batch:,}/{total_batches:,}"
                                    if current_batch
                                    else f" · {total_batches:,} batches planned"
                                )
                            refinement_status.update(
                                label=(
                                    f"Refinement {evaluation_number}/{maximum_new_caps} · "
                                    f"cap {candidate_cap:,} · {stage}{suffix}"
                                ),
                                state="running",
                                expanded=True,
                            )
                            refinement_detail.caption(
                                "Refinement elapsed "
                                + format_elapsed(time.monotonic() - refinement_started)
                            )

                        single_rows, _ = search_universal_shortlist_caps(
                            CURRENT_USER,
                            starting_cap=int(candidate_cap),
                            maximum_cap=int(candidate_cap),
                            adaptive=False,
                            step=refinement_step,
                            max_dd=float(saved_search_config["max_dd"]),
                            target_volatility=saved_search_config.get(
                                "target_volatility"
                            ),
                            drop_bottom_pct=float(
                                saved_search_config["drop_bottom_pct"]
                            ),
                            history_buffer_days=int(
                                saved_search_config["history_buffer_days"]
                            ),
                            redundancy_corr_threshold=float(
                                saved_search_config["redundancy_corr_threshold"]
                            ),
                            exact_optimizer_asset_cap=int(
                                saved_search_config["exact_optimizer_asset_cap"]
                            ),
                            exact_optimizer_asset_caps=tuple(
                                saved_search_config.get("exact_optimizer_asset_caps")
                                or [saved_search_config["exact_optimizer_asset_cap"]]
                            ),
                            minimum_trading_days=int(
                                saved_search_config.get("minimum_trading_days", 252)
                            ),
                            on_progress=update_refinement_stage,
                        )
                        candidate_row = dict(
                            best_feasible_result(single_rows) or single_rows[-1]
                        )
                        candidate_row["Search phase"] = "Refinement"
                        combined_rows = [
                            row
                            for row in combined_rows
                            if int(row["Shortlist cap"]) != int(candidate_cap)
                        ]
                        combined_rows.append(candidate_row)
                        combined_rows.sort(key=lambda row: int(row["Shortlist cap"]))
                        evaluated_rows.append(candidate_row)
                        pair_rows.append(candidate_row)
                        st.session_state["shortlist_cap_search_results"] = [
                            dict(row) for row in combined_rows
                        ]

                    if not pair_rows:
                        break
                    if not refinement_is_material(baseline_row, pair_rows):
                        stop_reason = (
                            "No material return improvement in the latest left/right pair"
                        )
                        break
                    feasible_combined = [
                        row for row in combined_rows if row.get("Status") == "Feasible"
                    ]
                    baseline_row = max(
                        feasible_combined,
                        key=lambda row: float(row.get("Annual Return", -np.inf)),
                    )
                    stop_reason = "Four-cap refinement limit reached"

                elapsed = time.monotonic() - refinement_started
                st.session_state["shortlist_cap_refinement_summary"] = {
                    "evaluated": len(evaluated_rows),
                    "elapsed_seconds": elapsed,
                    "stop_reason": stop_reason,
                }
                st.session_state.pop("shortlist_cap_refinement_error", None)
                refinement_progress.progress(1.0)
                refinement_status.update(
                    label="Bounded local refinement completed",
                    state="complete",
                    expanded=False,
                )
                st.rerun()
            except Exception as exc:
                st.session_state["shortlist_cap_refinement_error"] = safe_stop_reason(exc)
                refinement_status.update(
                    label="Local refinement stopped",
                    state="error",
                    expanded=True,
                )

        refinement_summary = st.session_state.get(
            "shortlist_cap_refinement_summary"
        )
        if refinement_summary:
            st.info(
                f"Local refinement evaluated "
                f"{int(refinement_summary.get('evaluated', 0))} additional cap(s) in "
                f"{format_elapsed(float(refinement_summary.get('elapsed_seconds', 0.0)))}. "
                f"Stopped because: {refinement_summary.get('stop_reason', 'completed')}."
            )
        refinement_error = st.session_state.get("shortlist_cap_refinement_error")
        if refinement_error:
            st.error(f"Local refinement stopped: {refinement_error}")

    percentage_columns = [
        "Annual Return",
        "Annual Volatility",
        "Historical ES 95% (1 Session)",
        "Block-Bootstrap ES 95% (20 Sessions)",
        "Linear fit predicted return",
        "Deviation from linear fit",
    ]
    feasible_chart = results_df.loc[results_df["Status"].eq("Feasible")].copy()
    if not feasible_chart.empty:
        chart_records = results_df.to_dict(orient="records")
        raw_marker = best_row.to_dict() if not feasible_df.empty else None
        robust_marker = robust_row.to_dict() if not feasible_df.empty else None
        evaluated_caps = sorted(
            feasible_chart["Shortlist cap"].astype(int).unique().tolist()
        )
        evaluated_asset_limits = sorted(
            feasible_chart["Maximum assets"].astype(int).unique().tolist()
        )
        if len(evaluated_caps) < 2:
            search_view = "Selected-cap curve"
            st.caption(
                f"Only shortlist cap {evaluated_caps[0]:,} has results, so a "
                "Maximum-assets curve is shown. The heatmap unlocks after a second "
                "shortlist cap completes."
            )
        elif len(evaluated_asset_limits) < 2:
            search_view = "Decision heatmap"
            st.caption(
                "Only one Maximum-assets value has results; the heatmap shows the "
                "observed return across shortlist caps."
            )
        else:
            search_view = st.segmented_control(
                "Search landscape view",
                options=["Decision heatmap", "Selected-cap curve", "3D landscape"],
                default="Decision heatmap",
                key="optimizer_search_landscape_view",
            )
        chart_config = {
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["toImage", "sendDataToCloud"],
        }
        if search_view == "3D landscape":
            chart_figure = build_search_surface_figure(
                chart_records,
                raw_peak=raw_marker,
                robust_choice=robust_marker,
            )
            chart_height = 650
            chart_key = "optimizer_search_surface_3d"
        elif search_view == "Selected-cap curve":
            preferred_cap = int(robust_row["Shortlist cap"])
            default_index = (
                evaluated_caps.index(preferred_cap)
                if preferred_cap in evaluated_caps else 0
            )
            if len(evaluated_caps) == 1:
                selected_slice_cap = evaluated_caps[0]
            else:
                selected_slice_cap = int(st.selectbox(
                    "Cross-section shortlist cap",
                    evaluated_caps,
                    index=default_index,
                    format_func=lambda value: f"{int(value):,}",
                    key="optimizer_search_slice_cap",
                ))
            chart_figure = build_search_slice_figure(
                chart_records, selected_slice_cap
            )
            chart_height = 430
            chart_key = "optimizer_search_slice"
        else:
            chart_figure = build_search_heatmap_figure(
                chart_records,
                raw_peak=raw_marker,
                robust_choice=robust_marker,
                shortlist_caps=cap_search_values,
                maximum_assets_values=exact_asset_search_values,
            )
            chart_height = 520
            chart_key = "optimizer_search_heatmap"
        if run_cap_search_btn and search_view == "Decision heatmap":
            live_search_chart.plotly_chart(
                chart_figure,
                width="stretch",
                height=chart_height,
                theme=None,
                key="optimizer_search_heatmap_live_final",
                config=chart_config,
            )
        else:
            st.plotly_chart(
                chart_figure,
                width="stretch",
                height=chart_height,
                theme=None,
                key=chart_key,
                config=chart_config,
            )
    display_columns = [
        "Search phase",
        "Shortlist cap",
        "Maximum assets",
        "Effective maximum assets",
        "Shortlisted",
        "Assets",
        "Trading days",
        *percentage_columns,
        "Sharpe Ratio",
        "Elapsed seconds",
        "Cap elapsed seconds",
        "Next adaptive jump",
        "Runtime brake",
        "Status",
    ]
    if "Search phase" not in results_df.columns:
        results_df["Search phase"] = "Previous run"
    else:
        results_df["Search phase"] = results_df["Search phase"].fillna("Previous run")
    for model_column in (
        "Linear fit predicted return",
        "Deviation from linear fit",
    ):
        if model_column not in results_df.columns:
            results_df[model_column] = np.nan
    st.dataframe(
        results_df[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column, format="percent")
            for column in percentage_columns
        }
        | {
            "Elapsed seconds": st.column_config.NumberColumn(
                "Solver time (seconds)", format="%.1f"
            ),
            "Cap elapsed seconds": st.column_config.NumberColumn(
                "Cap elapsed (seconds)", format="%.1f"
            ),
        },
    )
    st.caption(
        "Fixed mode evaluates the complete configured shortlist-cap × Maximum-assets "
        "grid. Each shortlist cap prepares history once; its exact-solver limits reuse "
        "that chronology-identical matrix. Only feasible solutions with at least 252 "
        "common trading sessions compete. The raw peak maximizes observed annual "
        "return; the robust choice treats near-equal returns as a plateau and uses "
        "downside risk, volatility, complexity and runtime as tie-breakers. Neither "
        "is a guaranteed future return. Nothing is published or traded automatically."
    )
    st.caption(
        "For adaptive refinements, a positive deviation means the solver beat the local "
        "linear-fit prediction; the next step may expand in that direction. A negative "
        "deviation means the fit was optimistic and the step is reduced."
    )

with st.expander("🌐 Universal Portfolio", expanded=False):
    st.caption(
        "One shared list of symbols visible and editable by everyone (quantity is always "
        "0, so it never counts as anyone's real holding). Run optimization automatically "
        "considers these as additional buy candidates alongside your own holdings."
    )
    universal_df = load_master_holdings(UNIVERSAL_OWNER)

    with st.container(border=True):
        st.markdown("**Composition at a glance**")
        if universal_df.empty:
            st.info("The universal portfolio is empty. Add symbols below.")
        else:
            universal_summary, universal_clusters = summarize_universal_portfolio(universal_df)
            cap_coverage = summarize_market_cap_coverage(
                universal_df,
                max_age_days=MARKET_CAP_MAX_AGE_DAYS,
            )
            universal_metric_cols = st.columns(4, gap="small")
            universal_metric_cols[0].metric(
                "Candidates", f'{universal_summary["total_symbols"]:,}', border=True
            )
            universal_metric_cols[1].metric(
                "India listed", f'{universal_summary["india_listed"]:,}', border=True
            )
            universal_metric_cols[2].metric(
                "Overseas listed", f'{universal_summary["overseas_listed"]:,}', border=True
            )
            universal_metric_cols[3].metric(
                "Listing clusters", f'{universal_summary["listing_clusters"]:,}', border=True
            )
            coverage_parts = []
            if universal_summary["exchanges"]:
                coverage_parts.append(
                    f'{len(universal_summary["exchanges"])} exchanges'
                )
            if universal_summary["currencies"]:
                coverage_parts.append(
                    "currencies " + ", ".join(universal_summary["currencies"])
                )
            if coverage_parts:
                st.caption("Coverage: " + " · ".join(coverage_parts))
            st.dataframe(
                universal_clusters,
                width="stretch",
                hide_index=True,
                column_config={
                    "Securities": st.column_config.NumberColumn(format="%d"),
                    "Share of universe": st.column_config.NumberColumn(format="percent"),
                },
            )
            st.caption(
                "A listing cluster is one exchange–currency group. It describes market "
                "coverage; it is not a sector classification or an allocation target."
            )
            cap_metrics = st.columns(4, gap="small")
            cap_metrics[0].metric(
                "Current market caps",
                f'{cap_coverage["current"]:,} / {cap_coverage["total"]:,}',
            )
            cap_metrics[1].metric("Stale / undated", f'{cap_coverage["stale"]:,}')
            cap_metrics[2].metric("Missing", f'{cap_coverage["missing"]:,}')
            cap_metrics[3].metric("Freshness window", f'{MARKET_CAP_MAX_AGE_DAYS} days')
            if cap_coverage["current"]:
                source_summary = ", ".join(cap_coverage["sources"]) or "source not recorded"
                st.caption(
                    f'Current snapshots: {cap_coverage["oldest_current_as_of"]} to '
                    f'{cap_coverage["newest_current_as_of"]} · source: {source_summary}.'
                )
            else:
                st.warning(
                    "No dated market-cap snapshot is currently usable. The market-cap "
                    "screen will retain candidates rather than claim a size exclusion."
                )

    with st.container(border=True):
        st.markdown("**Refresh market-cap snapshot**")
        st.caption(
            "Upload a CSV with a ticker column and marketCapitalizationMln (or an "
            "equivalent explicitly-in-millions column). This updates market-cap metadata "
            "only for exact Yahoo-ticker matches; quantities, prices, and the symbol list "
            "are untouched. You must enter the source and its as-of date—the import time "
            "is recorded separately. Snapshots older than 90 days are retained but not "
            "used by the optimizer's size screen."
        )
        market_cap_upload = st.file_uploader(
            "Market-cap snapshot CSV",
            type=["csv"],
            key="market_cap_snapshot_csv",
        )
        market_cap_source_col, market_cap_date_col = st.columns(2, gap="small")
        with market_cap_source_col:
            market_cap_source = st.text_input(
                "Data source / provider",
                value="Tickertape export",
                key="market_cap_snapshot_source",
            )
        with market_cap_date_col:
            market_cap_today_ist = pd.Timestamp.now(tz="Asia/Kolkata").date()
            market_cap_as_of = st.date_input(
                "Snapshot as-of date",
                value=market_cap_today_ist,
                max_value=market_cap_today_ist,
                key="market_cap_snapshot_as_of",
                help="Use the date stated by the data provider, not the upload date.",
            )
        refresh_market_cap_btn = st.button(
            "Refresh market-cap snapshot",
            key="refresh_market_cap_snapshot_btn",
            help="Fetch up to 1% of the remaining missing or stale market caps from Yahoo Finance and save dated metadata.",
        )
        refresh_until_stalled_btn = st.button(
            "Refresh repeatedly until no new market caps",
            key="refresh_market_cap_until_stalled_btn",
            help="Run bounded 1% batches until a batch adds no values or exceeds two-thirds failures (maximum 20 batches).",
        )
        if refresh_market_cap_btn:
            refresh_progress = st.progress(0.0, text="Preparing market-cap refresh…")
            refresh_detail = st.empty()

            def _market_cap_refresh_progress(done, total, ticker):
                fraction = done / total if total else 1.0
                refresh_progress.progress(
                    fraction,
                    text=f"Refreshing market caps: {done}/{total} · {ticker}",
                )
                refresh_detail.caption(f"Latest symbol: {ticker}")

            with st.spinner("Refreshing a bounded 1% sample of missing/stale market caps…"):
                with get_db_connection() as conn:
                    refresh_result = refresh_market_cap_sample(
                        conn,
                        universal_df,
                        owner=UNIVERSAL_OWNER,
                        sample_fraction=0.01,
                        max_age_days=MARKET_CAP_MAX_AGE_DAYS,
                        progress_callback=_market_cap_refresh_progress,
                    )
            refresh_progress.progress(1.0, text="Market-cap refresh finished")
            st.session_state["market_cap_refresh_result"] = refresh_result
            st.success(
                f"Market-cap refresh finished: {refresh_result.get('updated', 0):,} updated, "
                f"{refresh_result.get('failed', 0):,} failed out of "
                f"{refresh_result.get('attempted', 0):,} attempted."
            )
            if refresh_result.get("failures"):
                st.warning("Some symbols could not be refreshed: " + ", ".join(refresh_result["failures"]))
            st.rerun()
        if refresh_until_stalled_btn:
            refresh_progress = st.progress(0.0, text="Preparing repeated market-cap refresh…")
            refresh_detail = st.empty()

            def _market_cap_loop_progress(round_number, max_rounds, done, total, ticker):
                fraction = ((round_number - 1) + (done / total if total else 1.0)) / max_rounds
                refresh_progress.progress(
                    min(fraction, 1.0),
                    text=f"Batch {round_number}/{max_rounds}: {done}/{total} · {ticker}",
                )
                refresh_detail.caption(f"Latest symbol: {ticker}")

            with st.spinner("Refreshing 1% batches until no new values are available or failures exceed two-thirds…"):
                with get_db_connection() as conn:
                    refresh_result = refresh_market_cap_until_stalled(
                        conn,
                        universal_df,
                        owner=UNIVERSAL_OWNER,
                        sample_fraction=0.01,
                        max_failure_fraction=2 / 3,
                        max_age_days=MARKET_CAP_MAX_AGE_DAYS,
                        max_rounds=20,
                        progress_callback=_market_cap_loop_progress,
                    )
            refresh_progress.progress(1.0, text="Repeated market-cap refresh finished")
            st.session_state["market_cap_refresh_result"] = refresh_result
            st.success(
                f"Repeated refresh finished after {refresh_result.get('rounds', 0):,} batches: "
                f"{refresh_result.get('updated', 0):,} updated, "
                f"{refresh_result.get('failed', 0):,} failed "
                f"(stop: {refresh_result.get('stopped_reason', 'completed')})."
            )
            st.rerun()
        previous_refresh = st.session_state.get("market_cap_refresh_result")
        if previous_refresh:
            st.caption(
                f"Last refresh: {previous_refresh.get('as_of', 'unknown')} · "
                f"eligible {previous_refresh.get('eligible', 0):,} · "
                f"attempted {previous_refresh.get('attempted', 0):,} · "
                f"updated {previous_refresh.get('updated', 0):,} · "
                f"failed {previous_refresh.get('failed', 0):,}."
            )
        if market_cap_upload is not None:
            market_cap_preview = None
            market_cap_counts = {}
            market_cap_bytes = market_cap_upload.getvalue()
            market_cap_snapshot_key = hmac.new(
                b"market-cap-snapshot-preview",
                market_cap_bytes
                + str(market_cap_source).encode("utf-8")
                + market_cap_as_of.isoformat().encode("ascii"),
                digestmod="sha256",
            ).hexdigest()
            if st.session_state.get("market_cap_preview_key") != market_cap_snapshot_key:
                st.session_state["market_cap_preview_key"] = market_cap_snapshot_key
                st.session_state["confirm_market_cap_snapshot"] = False
            try:
                market_cap_preview, market_cap_counts = prepare_market_cap_snapshot(
                    market_cap_bytes,
                    universal_df,
                )
            except Exception as exc:
                st.error(f"Could not prepare this market-cap snapshot: {exc}")
            if market_cap_preview is not None:
                market_cap_kpis = st.columns(4, gap="small")
                market_cap_kpis[0].metric("CSV rows", f'{market_cap_counts.get("uploaded", 0):,}')
                market_cap_kpis[1].metric("Exact matches", f'{market_cap_counts.get("matched", 0):,}')
                market_cap_kpis[2].metric("Usable updates", f'{market_cap_counts.get("ready", 0):,}')
                market_cap_kpis[3].metric(
                    "Unmatched / ambiguous",
                    f'{market_cap_counts.get("unmatched", 0) + market_cap_counts.get("ambiguous", 0) + market_cap_counts.get("duplicate_source", 0):,}',
                )
                st.dataframe(
                    market_cap_preview.head(100),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Market Cap Millions": st.column_config.NumberColumn(
                            "Market cap (millions)", format="%.2f"
                        )
                    },
                )
                if len(market_cap_preview) > 100:
                    st.caption("Showing the first 100 rows; download the full match audit below.")
                st.download_button(
                    "Download market-cap match audit",
                    data=market_cap_preview.to_csv(index=False).encode("utf-8"),
                    file_name="market_cap_match_audit.csv",
                    mime="text/csv",
                    key="download_market_cap_match_audit",
                )
                st.checkbox(
                    "I checked the source and as-of date, and reviewed the exact-ticker matches",
                    key="confirm_market_cap_snapshot",
                )
                already_applied = (
                    st.session_state.get("market_cap_applied_snapshot_key")
                    == market_cap_snapshot_key
                )
                if st.button(
                    "Apply matched market-cap values only",
                    icon=":material/database_upload:",
                    type="primary",
                    width="stretch",
                    key="apply_market_cap_snapshot",
                    disabled=(
                        market_cap_counts.get("ready", 0) == 0
                        or not st.session_state.get("confirm_market_cap_snapshot", False)
                        or already_applied
                    ),
                ):
                    try:
                        with get_db_connection() as conn:
                            cap_result = apply_market_cap_snapshot(
                                conn,
                                market_cap_preview,
                                owner=UNIVERSAL_OWNER,
                                source=market_cap_source,
                                as_of=market_cap_as_of,
                            )
                        st.session_state["market_cap_applied_snapshot_key"] = market_cap_snapshot_key
                        st.session_state["holdings_flash_success"] = (
                            f'Updated market-cap metadata for {cap_result["updated"]:,} '
                            f'matched symbols from {cap_result["source"]} '
                            f'(as of {cap_result["as_of"]}). No holdings or quantities changed.'
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not apply the market-cap snapshot: {exc}")
                if already_applied:
                    st.success("This exact snapshot has already been applied in this session.")

    if not universal_df.empty:
        with st.container(border=True):
            st.markdown("**Shared symbols**")
            st.dataframe(
                universal_df[[
                    "Symbol", "Stock Name", "Yahoo Ticker", "Exchange", "Currency",
                    "Market Cap Millions", "Market Cap As Of", "Market Cap Source",
                    "Market Cap Imported At",
                ]],
                width="stretch",
                hide_index=True,
                column_config={
                    "Market Cap Millions": st.column_config.NumberColumn(
                        "Market cap (millions)", format="%.2f"
                    ),
                },
            )

    with st.container(border=True):
        st.markdown("**Replace overseas universe from Tickertape**")
        st.caption(
            "Upload a Tickertape U.S. screener CSV. The app validates and checkpoints "
            "the file before making any change. When you confirm replacement, all "
            "existing non-India Universal Portfolio rows are removed and the verified "
            "uploaded symbols are added in the same database transaction. India-listed "
            "rows are preserved."
        )
        overseas_source_upload = st.file_uploader(
            "Tickertape overseas screener CSV",
            type=["csv"],
            key="overseas_replacement_source_csv",
            help=(
                "The supplied Tickertape format with name and ticker columns is supported. "
                "Other screener columns are ignored."
            ),
        )
        prepare_overseas_replacement_btn = st.button(
            "Stage overseas replacement",
            icon=":material/inventory_2:",
            width="stretch",
            key="prepare_overseas_replacement",
            disabled=overseas_source_upload is None,
        )
        overseas_job = None
        if prepare_overseas_replacement_btn and overseas_source_upload is not None:
            try:
                with get_db_connection() as conn:
                    overseas_job_id, resumed_job, overseas_job = prepare_import_job(
                        conn,
                        overseas_source_upload.getvalue(),
                        overseas_source_upload.name,
                        UNIVERSAL_OWNER,
                        mode="replace_overseas",
                    )
                st.session_state["active_overseas_replacement_job"] = overseas_job_id
                if resumed_job:
                    st.info("This exact Tickertape file is already staged. Resuming it.")
                else:
                    st.success(
                        f"Staged {overseas_job['unique_symbols']:,} unique overseas "
                        "symbols. The current Universal Portfolio is unchanged."
                    )
            except Exception as exc:
                update_errors.append(f"Could not stage the overseas replacement: {exc}")

        active_overseas_job_id = st.session_state.get(
            "active_overseas_replacement_job"
        )
        try:
            with get_db_connection() as conn:
                if active_overseas_job_id:
                    overseas_job = get_import_job(conn, active_overseas_job_id)
                elif overseas_job is None:
                    overseas_job = latest_import_job(conn, mode="replace_overseas")
            if overseas_job is not None:
                st.session_state["active_overseas_replacement_job"] = overseas_job["job_id"]
        except Exception as exc:
            overseas_job = None
            update_errors.append(f"Could not load the overseas replacement job: {exc}")

        if overseas_job is not None:
            overseas_counts = overseas_job["counts"]
            overseas_pending = int(overseas_counts.get("pending", 0))
            overseas_verified = int(
                overseas_counts.get("verified", 0) + overseas_counts.get("added", 0)
            )
            validate_overseas_btn = st.button(
                "Validate / resume overseas symbols",
                icon=":material/fact_check:",
                width="stretch",
                key="validate_overseas_replacement",
                disabled=(overseas_pending == 0 or overseas_job["status"] == "applied"),
            )
            if validate_overseas_btn:
                try:
                    validate_universal_import_with_progress(overseas_job["job_id"])
                    st.rerun()
                except Exception as exc:
                    update_errors.append(
                        "Overseas validation stopped safely; the checkpoint can be resumed. "
                        f"Details: {exc}"
                    )

            overseas_metrics = st.columns(4, gap="small")
            overseas_metrics[0].metric(
                "Uploaded symbols", f"{overseas_job['unique_symbols']:,}"
            )
            overseas_metrics[1].metric("Verified", f"{overseas_verified:,}")
            overseas_metrics[2].metric("Pending", f"{overseas_pending:,}")
            overseas_metrics[3].metric(
                "Omitted",
                f"{overseas_counts.get('unresolved', 0) + overseas_counts.get('rejected', 0) + overseas_counts.get('conflict', 0):,}",
            )
            st.progress(
                float(overseas_job["progress"]),
                text=(
                    f"Processed {overseas_job['processed']:,} of "
                    f"{overseas_job['unique_symbols']:,} · {overseas_job['status']}"
                ),
            )

            replacement_ready = (
                overseas_pending == 0
                and int(overseas_counts.get("verified", 0)) > 0
                and overseas_job["status"] != "applied"
            )
            if replacement_ready:
                st.warning(
                    "The next action replaces the complete overseas candidate universe. "
                    "Your personal holdings are not changed, and every India-listed "
                    "Universal Portfolio row is retained."
                )
                st.checkbox(
                    "I reviewed the validation result and want to replace the overseas universe",
                    key="confirm_overseas_replacement",
                )
                overseas_confirmation = st.text_input(
                    "Type `REPLACE OVERSEAS` to confirm",
                    key="overseas_replacement_phrase",
                )
                apply_overseas_replacement_btn = st.button(
                    "Replace overseas universe",
                    icon=":material/sync_alt:",
                    type="primary",
                    width="stretch",
                    key="apply_overseas_replacement",
                    disabled=(
                        not st.session_state.get("confirm_overseas_replacement", False)
                        or overseas_confirmation.strip() != "REPLACE OVERSEAS"
                    ),
                )
                if apply_overseas_replacement_btn:
                    try:
                        with get_db_connection() as conn:
                            replacement_result = apply_verified_overseas_replacement(
                                conn, overseas_job["job_id"], UNIVERSAL_OWNER
                            )
                        st.session_state["holdings_flash_success"] = (
                            f"Replaced the overseas universe: removed "
                            f"{replacement_result['removed_overseas']:,} prior overseas "
                            f"symbols, added {replacement_result['added_overseas']:,} "
                            f"verified Tickertape symbols, and retained "
                            f"{replacement_result['retained_india']:,} India-listed symbols."
                        )
                        st.rerun()
                    except Exception as exc:
                        update_errors.append(f"Could not replace the overseas universe: {exc}")

            with get_db_connection() as conn:
                overseas_job = get_import_job(conn, overseas_job["job_id"])
                overseas_items = import_items_frame(conn, overseas_job["job_id"])
            st.download_button(
                "Download overseas validation report",
                data=safe_report_csv(overseas_items),
                file_name=f"overseas_replacement_{overseas_job['job_id']}.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
                key="download_overseas_replacement_report",
            )

    with st.container(border=True):
        st.markdown("**Percentile cleaner**")
        st.caption(
            "Stage a conservative optimizer-exclusion proposal using recent price availability, the "
            "strict bearish sequence Price < 25-DMA < 50-DMA < 200-DMA, and—where "
            "available—recent management net selling as a secondary signal. Your owned "
            "holdings are protected. Applying a result never deletes a Universal Portfolio "
            "row; it only omits the selected candidates from future optimization runs."
        )
        cleaner_cluster_options = available_clusters(universal_df)
        cleaner_clusters = st.multiselect(
            "Listing clusters to inspect",
            options=cleaner_cluster_options,
            default=[],
            placeholder="All listing clusters",
            help=(
                "Leave empty to inspect the full Universal Portfolio, or select one or "
                "more exchange–currency clusters."
            ),
            key="universal_cleaner_clusters",
        )
        cleaner_limit = st.slider(
            "Maximum removal percentile",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            format="%d%%",
            key="universal_cleaner_percentile",
            help=(
                "This is a hard cap, not a removal target. At 100%, every qualifying "
                "unowned symbol in the selected scope can be excluded from optimization; "
                "the Universal Portfolio rows are still not deleted."
            ),
        )
        cleaner_criterion_labels = {
            "Potentially obsolete / delisted": "unavailable",
            "Strict bearish DMA stack": "bearish",
            "Recent management net selling": "management",
        }
        cleaner_criteria_labels = st.multiselect(
            "Evidence to consider",
            options=list(cleaner_criterion_labels),
            default=list(cleaner_criterion_labels),
            key="universal_cleaner_criteria",
            help=(
                "Potentially obsolete/delisted means no usable recent Yahoo history and "
                "still requires manual review. Management selling is used only for a "
                "bounded U.S. shortlist; missing or ambiguous filing data is neutral."
            ),
        )
        cleaner_criteria = [
            cleaner_criterion_labels[label] for label in cleaner_criteria_labels
        ]
        prepare_cleaner_btn = st.button(
            "Prepare cleaner",
            icon=":material/filter_alt:",
            width="stretch",
            key="prepare_universal_cleaner",
            disabled=(universal_df.empty or not cleaner_criteria),
        )

        cleaner_job = None
        if prepare_cleaner_btn:
            try:
                with get_db_connection() as conn:
                    cleaner_job = prepare_cleaner_job(
                        conn,
                        UNIVERSAL_OWNER,
                        cleaner_clusters,
                        cleaner_limit,
                        "management" in cleaner_criteria,
                        cleaner_criteria,
                    )
                st.session_state["active_universal_cleaner_job"] = cleaner_job["job_id"]
                st.success(
                    f"Prepared a reversible snapshot of {cleaner_job['total_symbols']:,} "
                    "symbols. Run the scan to build the proposal."
                )
            except Exception as exc:
                update_errors.append(f"Could not prepare the Universal Portfolio cleaner: {exc}")

        active_cleaner_job_id = st.session_state.get("active_universal_cleaner_job")
        try:
            with get_db_connection() as conn:
                if active_cleaner_job_id:
                    cleaner_job = get_cleaner_job(conn, active_cleaner_job_id)
                elif cleaner_job is None:
                    cleaner_job = latest_cleaner_job(conn)
            if cleaner_job is not None:
                st.session_state["active_universal_cleaner_job"] = cleaner_job["job_id"]
        except Exception as exc:
            cleaner_job = None
            update_errors.append(f"Could not load the saved cleaner job: {exc}")

        if cleaner_job is not None:
            cleaner_counts = cleaner_job["counts"]
            cleaner_incomplete = bool(
                cleaner_counts["history_pending"] or cleaner_counts["insider_pending"]
            )
            cleaner_actions = st.container(horizontal=True)
            with cleaner_actions:
                run_cleaner_btn = st.button(
                    "Scan / resume",
                    icon=":material/manage_search:",
                    key="run_universal_cleaner",
                    disabled=(
                        cleaner_job["status"] in {"ready", "blocked", "applied"}
                        and not cleaner_incomplete
                    ),
                )
                apply_cleaner_btn = st.button(
                    "Apply optimizer exclusions",
                    icon=":material/filter_alt:",
                    type="primary",
                    key="apply_universal_cleaner",
                    disabled=(
                        cleaner_job["status"] != "ready"
                        or cleaner_counts["proposed"] == 0
                    ),
                )

            if run_cleaner_btn:
                try:
                    scan_universal_cleaner_with_progress(cleaner_job["job_id"])
                    st.rerun()
                except Exception as exc:
                    update_errors.append(
                        "Cleaner scan stopped safely; its last checkpoint can be resumed. "
                        f"Details: {exc}"
                    )

            if apply_cleaner_btn:
                try:
                    with get_db_connection() as conn:
                        cleaner_result = apply_cleaner_exclusions(
                            conn, cleaner_job["job_id"], UNIVERSAL_OWNER
                        )
                    protected_suffix = (
                        f" {len(cleaner_result['protected_at_apply']):,} newly owned symbols "
                        "were protected."
                        if cleaner_result["protected_at_apply"]
                        else ""
                    )
                    st.session_state["holdings_flash_success"] = (
                        f"Excluded {len(cleaner_result['excluded']):,} reviewed candidates "
                        f"from optimization without deleting them.{protected_suffix}"
                    )
                    st.rerun()
                except Exception as exc:
                    update_errors.append(f"Could not apply the cleaner proposal: {exc}")

            with get_db_connection() as conn:
                cleaner_job = get_cleaner_job(conn, cleaner_job["job_id"])
                cleaner_preview = cleaner_preview_frame(conn, cleaner_job["job_id"])
                cleaner_audit = cleaner_audit_frame(conn, cleaner_job["job_id"])
            cleaner_counts = cleaner_job["counts"]
            cleaner_metrics = st.columns(4, gap="small")
            cleaner_metrics[0].metric("In scope", f"{cleaner_job['total_symbols']:,}")
            cleaner_metrics[1].metric("Awaiting scan", f"{cleaner_counts['history_pending']:,}")
            cleaner_metrics[2].metric("Owned and protected", f"{cleaner_counts['protected']:,}")
            cleaner_metrics[3].metric("Proposed exclusions", f"{cleaner_counts['proposed']:,}")
            if cleaner_job.get("note"):
                if cleaner_job["status"] == "blocked":
                    st.error(cleaner_job["note"])
                else:
                    st.caption(cleaner_job["note"])
            if cleaner_job["clusters"]:
                st.caption("Scope: " + ", ".join(cleaner_job["clusters"]))
            else:
                st.caption("Scope: all listing clusters")
            if not cleaner_preview.empty:
                st.markdown("**Review proposed optimizer exclusions**")
                st.dataframe(
                    cleaner_preview,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="%.2f"),
                        "25-DMA": st.column_config.NumberColumn(format="%.2f"),
                        "50-DMA": st.column_config.NumberColumn(format="%.2f"),
                        "200-DMA": st.column_config.NumberColumn(format="%.2f"),
                        "Management net-sale ratio": st.column_config.NumberColumn(format="percent"),
                        "Exclusion score": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
            st.download_button(
                "Download cleaner audit",
                data=safe_report_csv(cleaner_audit),
                file_name=f"universal_cleaner_{cleaner_job['job_id']}.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
                key="download_universal_cleaner_audit",
            )
            st.caption(
                "No usable recent price is a review flag—not proof that a symbol is "
                "delisted. A broad data outage blocks the proposal. Management filings "
                "can reflect grants, exercises or planned sales, so they never act alone."
            )

        with get_db_connection() as conn:
            active_exclusions = optimizer_exclusions_frame(conn)
        if not active_exclusions.empty:
            exclusion_row = st.container(horizontal=True, vertical_alignment="center")
            with exclusion_row:
                st.metric(
                    "Currently omitted from optimization",
                    f"{len(active_exclusions):,}",
                    help="These rows remain in the Universal Portfolio and can be restored.",
                )
                restore_exclusions = st.button(
                    "Restore all candidates",
                    icon=":material/undo:",
                    key="restore_optimizer_exclusions",
                )
            with st.expander("Active optimizer exclusions", expanded=False):
                st.dataframe(active_exclusions, hide_index=True, width="stretch")
            if restore_exclusions:
                with get_db_connection() as conn:
                    restored = clear_optimizer_exclusions(conn)
                st.session_state["holdings_flash_success"] = (
                    f"Restored {restored:,} candidates to optimizer eligibility."
                )
                st.rerun()

    with st.container(border=True):
        st.markdown("**Remove an unsupported listing cluster**")
        st.caption(
            "Use this only when your investing platform cannot provide a listing cluster. "
            "Prepare a frozen sample, check those tickers on Tickertape, then explicitly "
            "confirm the cluster removal. Personal holdings remain protected."
        )
        cluster_to_review = st.selectbox(
            "Listing cluster",
            options=available_clusters(universal_df),
            index=None,
            placeholder="Choose one cluster to verify",
            key="cluster_delete_selection",
        )
        prepare_cluster_review = st.button(
            "Prepare verification sample",
            icon=":material/preview:",
            width="stretch",
            key="prepare_cluster_delete_review",
            disabled=cluster_to_review is None,
        )
        if prepare_cluster_review and cluster_to_review is not None:
            try:
                with get_db_connection() as conn:
                    cluster_members = cluster_members_frame(
                        conn, UNIVERSAL_OWNER, cluster_to_review
                    )
                st.session_state["cluster_delete_snapshot"] = {
                    "cluster": cluster_to_review,
                    "members": cluster_members.to_dict(orient="records"),
                }
                st.session_state["cluster_delete_checked"] = False
                st.session_state["cluster_delete_phrase"] = ""
            except Exception as exc:
                update_errors.append(f"Could not prepare the cluster sample: {exc}")

        cluster_snapshot = st.session_state.get("cluster_delete_snapshot")
        if (
            cluster_snapshot
            and cluster_to_review == cluster_snapshot.get("cluster")
        ):
            reviewed_members = pd.DataFrame(cluster_snapshot.get("members", []))
            sample = representative_cluster_sample(
                reviewed_members, cluster_to_review, sample_size=20
            )
            st.warning(
                f"This cluster contains {len(reviewed_members):,} symbols. Check the "
                f"{len(sample):,} representative tickers below on Tickertape before "
                "deciding. The sample is stable and distributed independently of table order."
            )
            st.dataframe(sample, hide_index=True, width="stretch")
            if not sample.empty:
                st.code(", ".join(sample["Yahoo ticker"].astype(str)), language=None)
            st.download_button(
                "Download complete cluster snapshot",
                data=safe_report_csv(reviewed_members),
                file_name=(
                    "universal_cluster_"
                    + re.sub(r"[^A-Za-z0-9]+", "_", cluster_to_review).strip("_").lower()
                    + ".csv"
                ),
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
                key="download_cluster_delete_snapshot",
            )
            st.checkbox(
                "I checked the sample and this cluster is not usable on my platform",
                key="cluster_delete_checked",
            )
            required_phrase = f"DELETE {cluster_to_review}"
            entered_phrase = st.text_input(
                f"Type `{required_phrase}` to confirm",
                key="cluster_delete_phrase",
            )
            delete_cluster_btn = st.button(
                "Delete reviewed cluster",
                icon=":material/delete_forever:",
                type="primary",
                width="stretch",
                key="delete_reviewed_cluster",
                disabled=(
                    not st.session_state.get("cluster_delete_checked", False)
                    or entered_phrase.strip() != required_phrase
                    or reviewed_members.empty
                ),
            )
            if delete_cluster_btn:
                try:
                    with get_db_connection() as conn:
                        cluster_delete_result = delete_cluster_snapshot(
                            conn,
                            UNIVERSAL_OWNER,
                            cluster_to_review,
                            reviewed_members["Symbol"].astype(str).tolist(),
                        )
                    protected_note = (
                        f" {len(cluster_delete_result['protected']):,} personally owned "
                        "symbols remained protected."
                        if cluster_delete_result["protected"]
                        else ""
                    )
                    st.session_state["holdings_flash_success"] = (
                        f"Removed {len(cluster_delete_result['removed']):,} reviewed "
                        f"symbols from {cluster_to_review}.{protected_note}"
                    )
                    st.session_state.pop("cluster_delete_snapshot", None)
                    st.rerun()
                except Exception as exc:
                    update_errors.append(f"Could not delete the reviewed cluster: {exc}")
        elif cluster_snapshot and cluster_to_review is not None:
            st.info("Prepare a new verification sample for the selected cluster.")

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
render_saved_analysis(
    saved_analysis_placeholder,
    CURRENT_USER,
    instance_key="page_load",
)

st.subheader("Master Holdings")
master_df = load_master_holdings(CURRENT_USER)

with st.expander("Manage holdings", expanded=True):
    st.markdown("**Add a holding**")
    st.caption(
        "Enter the bought quantity now. Average buy price is in the security's quoted "
        "currency; leave it at 0 only to use the latest Yahoo price as a temporary placeholder."
    )
    with st.form("manual_holding_add"):
        manual_symbol = st.text_input(
            "Ticker or Yahoo symbol",
            key="manual_holding_symbol",
            placeholder="RELIANCE.NS, AXTI or VT",
        )
        manual_quantity = st.number_input(
            "Bought quantity",
            min_value=0.000001,
            value=1.0,
            step=1.0,
            format="%.6f",
            key="manual_holding_quantity",
        )
        manual_average_price = st.number_input(
            "Average buy price (0 = use latest Yahoo price)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key="manual_holding_average_price",
        )
        manual_add = st.form_submit_button("Add holding", type="primary")
    if manual_add:
        try:
            get_latest_price_map.clear()
            _latest_fx_rate_to_inr.clear()
            added = add_holding_with_quantity(
                manual_symbol,
                manual_quantity,
                CURRENT_USER,
                average_price=(manual_average_price or None),
            )
            st.session_state["holdings_editor_version"] += 1
            clear_drop_bottom_coverage_preview()
            st.session_state.pop("drop_bottom_auto_result", None)
            st.session_state.pop("drop_bottom_auto_error", None)
            st.session_state["holdings_flash_success"] = (
                f"Added {added['symbol']} with quantity {added['quantity']:g} and "
                f"average price {added['average_price']:,.2f} "
                f"({added['price_source']})."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not add holding: {exc}")

    if not master_df.empty:
        st.divider()
        st.markdown("**Delete holdings**")
        holdings_to_delete = st.multiselect(
            "Select holdings to delete",
            options=master_df["Symbol"].tolist(),
            key="manual_holdings_to_delete",
            help="Deletion removes the selected rows from your private holdings only.",
        )
        if st.button(
            "Delete selected holdings",
            disabled=not holdings_to_delete,
            key="delete_selected_holdings",
        ):
            try:
                removed, missing = remove_symbols_from_master(
                    holdings_to_delete, CURRENT_USER)
                if not removed:
                    raise ValueError("No selected holdings were found.")
                st.session_state["holdings_editor_version"] += 1
                clear_drop_bottom_coverage_preview()
                st.session_state.pop("drop_bottom_auto_result", None)
                st.session_state.pop("drop_bottom_auto_error", None)
                message = "Deleted: " + ", ".join(removed)
                if missing:
                    message += ". Already absent: " + ", ".join(missing)
                st.session_state["holdings_flash_success"] = message
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete holdings: {exc}")

if st.button("Refresh FX to INR", key="refresh_holdings_fx"):
    _latest_fx_rate_to_inr.clear()
master_df = load_master_holdings(CURRENT_USER)

if master_df.empty:
    st.info("The master holdings table is empty. Add symbols from the sidebar.")
else:
    display_master = master_df.copy()
    try:
        fx = get_fx_to_inr_map(master_df["Currency"])
        display_master["FX to INR"] = master_df["Currency"].map(_normalize_currency_code).map(fx)
        display_master["Average Price INR"] = pd.to_numeric(master_df["Average Price"], errors="coerce") * display_master["FX to INR"]
        display_master["Value at entered cost INR"] = pd.to_numeric(master_df["Quantity"], errors="coerce") * display_master["Average Price INR"]
        st.caption("Average Price stays in the quoted currency. INR columns use Yahoo reference FX, refreshed at most every five minutes or with Refresh FX. This is current-FX conversion of entered cost, not historical INR purchase cost or live market value.")
    except Exception:
        st.warning("FX unavailable: INR equivalents are not shown. No 1:1 conversion is assumed; your native holdings are unchanged.")
    st.dataframe(display_master, width="stretch", hide_index=True, column_config={
        "FX to INR": st.column_config.NumberColumn(format="%.4f"),
        "Average Price INR": st.column_config.NumberColumn(format="₹%.2f"),
        "Value at entered cost INR": st.column_config.NumberColumn(format="₹%.2f"),
    })

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
    _app_run_log("optimization_started", user=bool(CURRENT_USER))
    try:
        _display_optimization_stage(
            optimization_timer_placeholder, "Loading holdings", 0.03
        )
        if master_df.empty:
            stopped_timer = _abort_optimization_plan_timer("empty holdings")
            optimization_timer_placeholder.warning(_timer_summary(stopped_timer)[1])
            st.error("Add at least one symbol before running the analysis.")
            st.stop()

        with st.spinner("Loading holdings from the database..."):
            portfolio_df, invalid_holding_rows = build_current_allocation_from_db(CURRENT_USER)

        if portfolio_df.empty:
            stopped_timer = _abort_optimization_plan_timer("no usable holdings")
            optimization_timer_placeholder.warning(_timer_summary(stopped_timer)[1])
            st.error(
                "No usable holdings were found. Add valid Average Price and Quantity values in the master table."
            )
            st.stop()

        if invalid_holding_rows:
            st.warning(
                "Skipped holdings with missing or invalid quantity/price: "
                + ", ".join(invalid_holding_rows)
            )

        _display_optimization_stage(
            optimization_timer_placeholder,
            "Shortlisting the broad candidate universe",
            0.08,
        )

        def report_shortlist_progress(fraction, details=None):
            details = details or {}
            primary_batch = int(details.get("primary_batch") or 0)
            primary_batches = int(details.get("primary_batches") or 0)
            recovered = int(details.get("recovered") or 0)
            requests = int(details.get("requests") or 0)
            if details.get("validating_integrity"):
                stage = f"Validating {recovered:,} recovered shortlist histories"
            elif details.get("retry"):
                stage = (
                    f"Retrying missing shortlist data after batch "
                    f"{primary_batch:,}/{primary_batches:,}"
                )
            elif primary_batches:
                stage = (
                    f"Downloading shortlist market data · batch "
                    f"{primary_batch:,}/{primary_batches:,} · "
                    f"{recovered:,} recovered · {requests:,} requests"
                )
            else:
                stage = "Preparing shortlist market-data batches"
            overall_progress = 0.08 + 0.25 * float(np.clip(fraction, 0.0, 1.0))
            _display_optimization_stage(
                optimization_timer_placeholder,
                stage,
                overall_progress,
            )

        if automatic_prepared_analysis is not None:
            cached_allocation, cached_returns, cached_meta = automatic_prepared_analysis
            if (
                int(cached_meta.get("universal_preselection", {}).get("cap") or universal_preselection_cap)
                != int(universal_preselection_cap)
                or cached_returns is None
                or cached_returns.empty
            ):
                automatic_prepared_analysis = None
        if automatic_prepared_analysis is not None:
            portfolio_df = cached_allocation.copy()
            universal_preselection = dict(
                (cached_meta or {}).get("universal_preselection") or {}
            )
            universal_candidate_symbols = (
                portfolio_df.loc[
                    pd.to_numeric(portfolio_df.get("Quantity", 0), errors="coerce")
                    .fillna(0.0)
                    .le(0.0),
                    "Yahoo Ticker",
                ]
                .dropna()
                .astype(str)
                .tolist()
            )
            st.info(
                "Reusing the selected finalist’s prepared shortlist and market history "
                "for the automatic final solve."
            )
        else:
            with st.spinner("Adding Universal Portfolio symbols as buy candidates..."):
                (
                    portfolio_df,
                    universal_candidate_symbols,
                    universal_preselection,
                ) = extend_allocation_with_universal_candidates(
                    portfolio_df,
                    maximum_candidates=universal_preselection_cap,
                    owner=CURRENT_USER,
                    progress_callback=report_shortlist_progress,
                )
        previous_elapsed, previous_workload = _previous_run_estimate(CURRENT_USER)
        refined_estimate = estimate_run_seconds(
            previous_elapsed,
            previous_workload,
            universal_preselection.get("eligible"),
        )
        _display_optimization_stage(
            optimization_timer_placeholder,
            "Candidate shortlist ready",
            0.35,
            estimated_total_seconds=refined_estimate,
        )
        if universal_candidate_symbols:
            eligible_universe_count = int(
                universal_preselection.get("eligible")
                or universal_preselection.get("eligible_symbols")
                or universal_preselection.get("shortlisted")
                or len(universal_candidate_symbols)
            )
            market_cap_filter = universal_preselection.get("market_cap_filter") or {}
            fresh_caps = int(market_cap_filter.get("fresh_known", market_cap_filter.get("known", 0)))
            stale_caps = int(market_cap_filter.get("stale_retained", 0))
            missing_caps = int(
                market_cap_filter.get(
                    "missing",
                    market_cap_filter.get("unknown_retained", eligible_universe_count),
                )
            )
            excluded_caps = int(market_cap_filter.get("excluded", 0))
            risk_appetite = (
                (universal_preselection.get("market_data") or {}).get("risk_appetite")
                or {}
            )
            market_read = interpret_risk_appetite_regime(risk_appetite)
            st.info(
                f"Universal candidate shortlist: {len(universal_candidate_symbols):,} of "
                f"{eligible_universe_count:,} eligible symbols."
            )
            if fresh_caps:
                st.caption(
                    f"Market-cap screen active: excluded {excluded_caps:,} symbols in "
                    "sufficiently populated listing clusters using a fresh dated snapshot; "
                    f"{stale_caps:,} stale and {missing_caps:,} missing caps were retained."
                )
            else:
                st.warning(
                    "Market-cap screen inactive: no fresh dated snapshot was available, "
                    f"so it excluded 0 symbols. {stale_caps:,} stale and {missing_caps:,} "
                    "missing market caps were retained. Refresh the dated cap data before "
                    "relying on market size as a shortlist safeguard."
                )
            _render_market_read_panel(market_read, title="Market backdrop")
        quarterly_excluded = int(universal_preselection.get("quarterly_excluded", 0))
        if quarterly_excluded:
            st.caption(
                f"{quarterly_excluded:,} paused securities are excluded "
                "from the universal candidate pool until their 90-day pause expires."
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
            stopped_timer = _abort_optimization_plan_timer("no resolved Yahoo tickers")
            optimization_timer_placeholder.warning(_timer_summary(stopped_timer)[1])
            st.error("No valid Yahoo tickers resolved.")
            st.stop()

        _display_optimization_stage(
            optimization_timer_placeholder,
            "Downloading full history and optimizing risk/return",
            0.42,
        )

        def report_analysis_progress(stage, fraction, details=None):
            details = details or {}
            suffix = ""
            if details.get("batches"):
                current_batch = int(details.get("batch") or 0)
                total_batches = int(details["batches"])
                suffix = (
                    f" · batch {current_batch:,}/{total_batches:,}"
                    if current_batch
                    else f" · {total_batches:,} batches planned"
                )
            elif details.get("surviving_symbols") is not None:
                suffix = f" · {int(details['surviving_symbols']):,} symbols"
            elif details.get("aligned_assets") is not None:
                suffix = f" · {int(details['aligned_assets']):,} assets"
            elif details.get("eligible_assets") is not None:
                suffix = f" · {int(details['eligible_assets']):,} eligible"
            elif details.get("assets") is not None:
                suffix = f" · {int(details['assets']):,} assets"
            overall_progress = 0.42 + 0.43 * float(np.clip(fraction, 0.0, 1.0))
            _display_optimization_stage(
                optimization_timer_placeholder,
                f"{stage}{suffix}",
                overall_progress,
                estimated_total_seconds=refined_estimate,
            )

        if automatic_prepared_analysis is not None:
            _display_optimization_stage(
                optimization_timer_placeholder,
                "Solving the selected search finalist from prepared history",
                0.80,
                estimated_total_seconds=refined_estimate,
            )
            optimal_weights, log_returns, current_stats, optimal_stats, meta = (
                solve_prepared_search_finalist(
                    automatic_prepared_analysis,
                    portfolio_df.copy(),
                    maximum_assets=exact_optimizer_asset_cap,
                    max_dd=max_dd,
                    target_volatility=target_volatility,
                )
            )
        else:
            with st.spinner("Running optimization..."):
                optimal_weights, log_returns, current_stats, optimal_stats, meta = run_portfolio_analysis_multi(
                    yahoo_tickers,
                    portfolio_df.copy(),
                    max_dd=max_dd,
                    target_volatility=target_volatility,
                    drop_bottom_pct=drop_bottom_pct,
                    buffer_days=history_buffer_days,
                    redundancy_corr_threshold=redundancy_corr_threshold,
                    exact_optimizer_asset_cap=exact_optimizer_asset_cap,
                    progress_callback=report_analysis_progress,
                )

        _display_optimization_stage(
            optimization_timer_placeholder,
            "Optimization complete; preparing execution plan",
            0.88,
        )

        if optimal_weights is None:
            stopped_timer = _abort_optimization_plan_timer("no usable allocation")
            optimization_timer_placeholder.warning(_timer_summary(stopped_timer)[1])
            st.error("Portfolio optimization did not return a usable allocation.")
            st.stop()
        solver_screen = meta.get("exact_optimizer_screen") or {}
        history_cache = meta.get("market_history_cache") or {}
        pipeline_timings = meta.get("pipeline_timings_seconds") or {}
        if history_cache:
            timing_parts = [
                ("Yahoo", (history_cache.get("timings_seconds") or {}).get("yahoo_download")),
                ("cache", (history_cache.get("timings_seconds") or {}).get("cache_read")),
                ("filters", pipeline_timings.get("quality_and_momentum_filters")),
                ("alignment", pipeline_timings.get("common_period_alignment")),
                ("deduplication", pipeline_timings.get("near_duplicate_filter")),
                ("solver", pipeline_timings.get("optimizer_solver")),
            ]
            timing_text = " · ".join(
                f"{label} {float(seconds):.1f}s"
                for label, seconds in timing_parts
                if seconds is not None
            )
            st.caption(
                "History cache · "
                f"{int(history_cache.get('symbols_reused') or 0):,} reused · "
                f"{int(history_cache.get('symbols_refreshed') or 0):,} refreshed · "
                f"{int(history_cache.get('rows_written') or 0):,} rows written · "
                f"{int(history_cache.get('symbols_unavailable') or 0):,} unavailable"
                + (f"  |  {timing_text}" if timing_text else "")
            )
        if solver_screen:
            st.info(
                "Exact optimization used "
                f"**{int(solver_screen.get('selected_assets') or log_returns.shape[1]):,}** "
                "of "
                f"**{int(solver_screen.get('eligible_assets') or log_returns.shape[1]):,}** "
                "post-filter assets; "
                f"**{int(solver_screen.get('owned_assets_protected') or 0):,}** owned "
                "holdings were protected."
            )
        price_integrity_report = meta.get("price_integrity_report")
        if isinstance(price_integrity_report, pd.DataFrame) and not price_integrity_report.empty:
            repaired_count = int(
                price_integrity_report["Status"].eq("REPAIRED_CORPORATE_ACTION").sum()
            )
            quarantined_count = int(
                price_integrity_report["Status"].astype(str).str.startswith("QUARANTINED_").sum()
            )
            if repaired_count:
                st.info(
                    f"Price-integrity gate reconciled **{repaired_count:,}** "
                    "corporate-action histories before calculating returns."
                )
            if quarantined_count:
                st.warning(
                    f"Price-integrity gate quarantined **{quarantined_count:,}** "
                    "unresolved histories. New candidates were excluded; owned "
                    "holdings were frozen at their current weights."
                )
            with st.expander("Price-integrity audit", expanded=False):
                st.dataframe(price_integrity_report, width="stretch", hide_index=True)
        missing_owned_returns = meta.get("missing_owned_return_tickers", [])
        if missing_owned_returns:
            st.warning(
                "Current-portfolio comparison excluded holdings without usable final "
                "return history: " + ", ".join(missing_owned_returns) + ". The optimized "
                "portfolio remains valid for the securities in its return matrix."
            )
        # Show optimal portfolio weights table
        st.subheader("Optimal Portfolio (Weights)")
        optimal_portfolio_df = pd.DataFrame({
            "Ticker": log_returns.columns,
            "Optimal Weight": optimal_weights,
        })
        optimal_portfolio_df = optimal_portfolio_df[optimal_portfolio_df["Optimal Weight"] > 0].sort_values(
            "Optimal Weight", ascending=False
        ).reset_index(drop=True)
        
        st.dataframe(
            optimal_portfolio_df.style.format({"Optimal Weight": "{:.2%}"}),
            width="stretch",
            hide_index=True,
        )
        
        st.download_button(
            "Download optimal portfolio CSV",
            data=optimal_portfolio_df.to_csv(index=False).encode("utf-8"),
            file_name="optimal_portfolio_weights.csv",
            mime="text/csv",
            width="stretch",
            key="download_optimal_portfolio_csv",
        )

        if not meta["dropped_df"].empty:
            st.subheader("Dropped Tickers")
            st.dataframe(meta["dropped_df"], width="stretch")

        momentum_dropped_df = meta.get("momentum_dropped_df")
        if momentum_dropped_df is not None and not momentum_dropped_df.empty:
            st.subheader("Excluded by Robust Momentum Gate")
            st.caption(
                "At most 20% of zero-quantity Universal candidates can be excluded. "
                "Owned holdings are always preserved for the optimizer to reduce or sell correctly."
            )
            st.dataframe(
                momentum_dropped_df.style.format({
                    "Stable Momentum Score": "{:.1%}",
                    "3-1 INR Return": "{:.2%}",
                    "6-1 INR Return": "{:.2%}",
                    "12-1 INR Return": "{:.2%}",
                }),
                width="stretch",
                hide_index=True,
            )

        redundant_df = meta.get("redundant_df")
        if redundant_df is not None and not redundant_df.empty:
            st.subheader("Merged Near-Duplicate Assets")
            st.caption(
                f"Correlated at or above {redundancy_corr_threshold:.2f}; the most liquid "
                "candidate of each group was kept so the weight cap limits real exposure. "
                "Existing owned holdings are protected from removal by this filter."
            )
            st.dataframe(
                redundant_df.style.format({"Correlation": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )

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

        _display_optimization_stage(
            optimization_timer_placeholder,
            "Fetching final prices and generating trades",
            0.94,
        )
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
        with get_db_connection() as conn:
            active_deferrals = active_trade_deferrals(conn, CURRENT_USER)
        rebal_df, deferred_rebal_df = split_deferred_trades(
            rebal_df,
            active_deferrals,
        )
        deferred_trade_expiry = (
            dict(
                zip(
                    active_deferrals["yahoo_ticker"].astype(str).str.upper(),
                    active_deferrals["expires_at"].astype(str),
                )
            )
            if not active_deferrals.empty
            else {}
        )
        finished_timer = _finish_optimization_plan_timer()
        optimization_timer_placeholder.success(_timer_summary(finished_timer)[1])
        scenario_ranking = pd.DataFrame()
        scenario_ranking_error = None
        try:
            scenario_ranking = forecast_executable_buys(
                rebal_df,
                log_returns,
                horizon_sessions=int(security_forecast_horizon),
                one_way_cost_bps=float(security_forecast_cost_bps),
            )
        except Exception as exc:
            # An optional research view must never invalidate the actionable
            # rebalance plan or its saved analysis.
            scenario_ranking_error = str(exc)

        if missing_prices:
            st.warning(f"Skipped symbols with missing latest price: {', '.join(missing_prices)}")
        if missing_alloc:
            st.warning(f"Skipped symbols missing in allocation: {', '.join(missing_alloc)}")

        holdings_action_df = build_holdings_action_summary(
            portfolio_df,
            rebal_df,
            dict(zip(log_returns.columns, optimal_weights)),
            price_map,
            meta,
            invalid_holding_rows=invalid_holding_rows,
            missing_price_tickers=missing_prices,
            deferred_trade_expiry=deferred_trade_expiry,
        )

        if not holdings_action_df.empty:
            st.subheader("Action for Every Holding You Own")
            st.caption(
                "The rebalancing plan below lists executable trades only. This table also "
                "covers holdings that produced no trade, and says why."
            )
            st.dataframe(
                style_holdings_action_summary(holdings_action_df),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download holdings action summary CSV",
                data=holdings_action_df.to_csv(index=False).encode("utf-8"),
                file_name="holdings_action_summary.csv",
                mime="text/csv",
                width="stretch",
                key="download_holdings_action_summary",
            )

        lumpsum_plan_payload = None
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
            execution_sheet_html = lumpsum_execution_sheet_html(
                lumpsum_df,
                lumpsum_inr,
                unallocated_cash,
            )
            lumpsum_plan_payload = {
                "amount_inr": float(lumpsum_inr),
                "unallocated_cash": float(unallocated_cash),
                "orders": lumpsum_df.to_dict(orient="records"),
            }
            trigger_download(
                execution_sheet_html.encode("utf-8"),
                "lumpsum_buy_orders.html",
                "text/html",
            )

        st.subheader("Rebalancing Plan")

        st.markdown("**Single-buy scenario ranking**")
        st.caption(
            f"Empirical-Bayes-shrunk historical block-resampling scenarios over {int(security_forecast_horizon)} "
            f"trading sessions; estimated {float(security_forecast_cost_bps):.0f} bp cost "
            "per side is deducted on entry and exit. Ranking score = expected net scenario "
            "return × historical positive-path frequency; optimized weight is shown only "
            "for context and does not affect the single-buy ranking. This product is a "
            "heuristic, not expected value, because the mean return already includes "
            "negative paths. These frequencies "
            "are not calibrated future probabilities; taxes, security-specific fees, "
            "available-cash feasibility, and market impact are not modeled. This ranks "
            "buys already present in the executable plan and does not change any order."
        )
        if scenario_ranking_error:
            st.warning(f"Scenario ranking unavailable; the rebalance plan is unaffected: {scenario_ranking_error}")
        elif scenario_ranking.empty:
            st.info("No executable buy candidates with usable return history were available to rank.")
        else:
            leading_buy = scenario_ranking.iloc[0]
            leading_score = float(leading_buy["Return × positive-probability score (%)"])
            if leading_score > 0:
                st.success(
                    f"Top single-buy candidate (research-only): "
                    f"{leading_buy['Security']} · expected net scenario return "
                    f"{float(leading_buy['Expected net scenario return (%)']):+.2f}% · "
                    f"return × positive-path score {leading_score:+.2f}% · "
                    f"historical-bootstrap positive-path frequency "
                    f"{float(leading_buy['Probability of positive net return (%)']):.1f}%."
                )
            else:
                st.info(
                    "No executable buy has a positive return × positive-path score under the "
                    "selected horizon and cost assumption."
                )
            scenario_display = scenario_ranking.drop(columns=["Scenario method"])
            st.dataframe(
                scenario_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Optimized Weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Expected net scenario return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Median net return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    "Probability of positive net return (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Return × positive-probability score (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )

        if rebal_df.empty:
            st.success("No trades to execute after filtering Executable Quantity = 0")
        else:
            st.caption(
                "Buys and zero-target exits are shown. Partial sells are omitted when "
                "the security remains in the optimal portfolio, reducing avoidable turnover and costs."
            )
            ordered_rebal_df = _sort_rebalance_df_for_priority(rebal_df)
            st.dataframe(style_rebalance_df(ordered_rebal_df), width="stretch")

            csv = ordered_rebal_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download rebalancing plan CSV",
                data=csv,
                file_name="rebalancing_plan.csv",
                mime="text/csv",
                width="stretch",
                key="download_rebalancing_plan_csv",
            )

        publication_run_id = f"RUN-{uuid.uuid4().hex.upper()}"
        approved_constituents = round_weights_to_whole_percent([
            {"ticker": str(ticker), "target_weight": float(weight)}
            for ticker, weight in zip(log_returns.columns, optimal_weights)
        ])
        portfolio_log_return_history = np.asarray(log_returns @ optimal_weights, dtype=float)
        leverage_assessment = None
        leverage_assessment_error = None
        try:
            with st.spinner("Calculating portfolio-derived leverage scenarios…"):
                leverage_assessment = build_portfolio_leverage_assessment(
                    portfolio_log_return_history,
                    invested_base=float(total_current_value),
                )
            leverage_assessment["computed_at"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
        except (TypeError, ValueError, FloatingPointError) as exc:
            # Leverage is a supplementary diagnostic and must not invalidate a
            # completed rebalancing plan when its statistical input is unusable.
            leverage_assessment_error = str(exc)
        prior_saved_analysis = load_latest_analysis(CURRENT_USER) or {}
        analysis_payload = {
            "review_analysis_context": {
                "schema_version": 1,
                "optimizer_config_version": OPTIMIZER_CONFIG_VERSION,
                "risk_free_rate_annual": RISK_FREE_RATE_ANNUAL,
                "trading_days_per_year": TRADING_DAYS_PER_YEAR,
                "max_weight_per_asset": MAX_WEIGHT_PER_ASSET,
                "price_history_start": str(meta["valid_start"]),
                "price_history_end": str(meta["valid_end"]),
                "return_observation_start": str(log_returns.index.min()),
                "return_observation_end": str(log_returns.index.max()),
                "return_observations": int(len(log_returns)),
                "return_type": "log",
                "currency": "INR",
                "tickers": list(log_returns.columns),
                "purpose": "Optimizer provenance, not a predicted target-XIRR date. Review must independently validate current common history and exit costs.",
            },
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_to_rebalancing_plan_timing": serializable_timer(
                st.session_state.get(OPTIMIZATION_TIMER_KEY)
            ),
            "run_id": publication_run_id,
            "holdings_analyzed": int((portfolio_df["Quantity"] > 0).sum()),
            "total_invested": float(total_invested),
            "current_portfolio_value": float(total_current_value),
            "leverage_return_history": {
                "schema_version": 1,
                "dates": [str(value) for value in log_returns.index],
                "portfolio_log_returns": portfolio_log_return_history.tolist(),
                "weights_source": "saved optimal portfolio weights applied to prepared full-history log returns",
            },
            "leverage_assessment": leverage_assessment,
            "leverage_assessment_error": leverage_assessment_error,
            "leverage_check_inputs": dict(
                prior_saved_analysis.get("leverage_check_inputs") or {}
            ),
            "executable_trade_count": int(len(rebal_df)),
            "settings": {
                "optimizer_config": dict(OPTIMIZER_CONFIG),
                "days_to_flip": int(days_to_flip),
                "security_scenario_forecast": {
                    "method": "eb-shrunk-stationary-block-bootstrap-v1",
                    "horizon_sessions": int(security_forecast_horizon),
                    "one_way_cost_bps": float(security_forecast_cost_bps),
                    "paths": 4000,
                    "status": "RESEARCH_SCENARIOS_NOT_CALIBRATED_FORECASTS",
                },
                "max_drawdown_input_pct": float(max_dd_pct),
                "internal_max_dd": float(max_dd),
                "drop_bottom_fraction": float(drop_bottom_pct),
                "momentum_filter": dict(MOMENTUM_FILTER_CONFIG),
                "universe_preselection": dict(universal_preselection),
                "sell_trade_policy": "omit-partial-sells-v1",
                "history_buffer_days": int(history_buffer_days),
                "redundancy_corr_threshold": float(redundancy_corr_threshold),
                "exact_optimizer_asset_cap": int(exact_optimizer_asset_cap),
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
                "exact_optimizer_screen": dict(
                    meta.get("exact_optimizer_screen") or {}
                ),
                "price_integrity_version": str(
                    meta.get("price_integrity_version") or PRICE_INTEGRITY_VERSION
                ),
                "price_integrity_report": (
                    meta["price_integrity_report"].to_dict(orient="records")
                    if isinstance(meta.get("price_integrity_report"), pd.DataFrame)
                    else []
                ),
                "price_integrity_frozen_owned_weights": dict(
                    meta.get("price_integrity_frozen_owned_weights") or {}
                ),
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
                "merged_redundant_assets": (
                    redundant_df.to_dict(orient="records")
                    if redundant_df is not None and not redundant_df.empty
                    else []
                ),
                "momentum_exclusions": (
                    momentum_dropped_df.to_dict(orient="records")
                    if momentum_dropped_df is not None and not momentum_dropped_df.empty
                    else []
                ),
                "momentum_diagnostics": (
                    meta["momentum_report_df"].to_dict(orient="records")
                    if meta.get("momentum_report_df") is not None
                    and not meta["momentum_report_df"].empty
                    else []
                ),
            },
            "current_stats": current_stats or {},
            "optimal_stats": optimal_stats or {},
            "optimal_allocation": approved_constituents,
            "publication_candidate": {
                "basket_id": "PUBLIC-01",
                "run_id": publication_run_id,
                "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "calculation_version": "private-rebalancer-v1",
                "strategy_version": "portfolio-rebalancer-v1",
                "constituents": approved_constituents,
                "cash_weight": 0.0,
            },
            "top_correlations": (
                top_corrs.head(5).to_dict(orient="records")
                if not top_corrs.empty
                else []
            ),
            "current_allocation": portfolio_df.to_dict(orient="records"),
            "holdings_action_summary": holdings_action_df.to_dict(orient="records"),
            "rebalancing_plan": rebal_df.to_dict(orient="records"),
            "security_scenario_ranking": (
                scenario_ranking.to_dict(orient="records")
                if not scenario_ranking.empty
                else []
            ),
            "security_scenario_ranking_error": scenario_ranking_error,
            "deferred_rebalancing_trades": deferred_rebal_df.to_dict(orient="records"),
            "lumpsum_plan": lumpsum_plan_payload,
            "warnings": {
                "invalid_holding_rows": invalid_holding_rows,
                "unresolved_yahoo_tickers": unresolved,
                "missing_history_tickers": meta.get("missing_history_tickers", []),
                "missing_owned_return_tickers": meta.get(
                    "missing_owned_return_tickers", []
                ),
                "missing_latest_prices": missing_prices,
                "missing_allocation": missing_alloc,
            },
        }

        save_latest_analysis(analysis_payload, CURRENT_USER)
        render_saved_analysis(
            saved_analysis_placeholder,
            CURRENT_USER,
            instance_key="after_optimization",
        )
        st.success(
            "Analysis completed and saved. You can now download the complete "
            "analysis backup as JSON."
        )
        market_cap_filter = ((analysis_payload.get("settings") or {}).get("universe_preselection") or {}).get("market_cap_filter") or {}
        leading_buy = scenario_ranking.iloc[0] if not scenario_ranking.empty else None
        try:
            reserve_log = calculate_portfolio_reserve_target(
                analysis_payload.get("optimal_stats") or {},
                invested_base=float(analysis_payload.get("current_portfolio_value") or analysis_payload.get("total_invested") or 0.0),
            )
            reserve_fraction_log = float(reserve_log.get("reserve_fraction"))
        except (TypeError, ValueError, FloatingPointError):
            reserve_fraction_log = None
        _app_run_log(
            "optimization_finished",
            status="completed",
            trading_days=int(len(log_returns)),
            holdings_count=int((portfolio_df["Quantity"] > 0).sum()) if "Quantity" in portfolio_df.columns else int(len(portfolio_df)),
            executable_trades=int(len(rebal_df)),
            market_cap_fresh_known=int(market_cap_filter.get("fresh_known", market_cap_filter.get("known", 0)) or 0),
            market_cap_stale_retained=int(market_cap_filter.get("stale_retained", 0) or 0),
            market_cap_missing=int(market_cap_filter.get("unknown_retained", market_cap_filter.get("missing", 0)) or 0),
            market_cap_refresh_attempted=int((market_cap_filter.get("auto_refresh") or {}).get("attempted", 0) or 0),
            market_cap_refresh_updated=int((market_cap_filter.get("auto_refresh") or {}).get("updated", 0) or 0),
            market_cap_refresh_failed=int((market_cap_filter.get("auto_refresh") or {}).get("failed", 0) or 0),
            reserve_fraction=reserve_fraction_log,
            leverage_debt_ratio=(
                float((leverage_assessment or {}).get("recommended_debt_ratio_of_portfolio_base"))
                if (leverage_assessment or {}).get("recommended_debt_ratio_of_portfolio_base") is not None
                else None
            ),
            top_buy_symbol=(str(leading_buy.get("Security")) if leading_buy is not None and leading_buy.get("Security") else None),
            top_buy_score=(float(leading_buy.get("Return × positive-probability score (%)")) if leading_buy is not None and leading_buy.get("Return × positive-probability score (%)") is not None else None),
            warning_count=len((analysis_payload.get("warnings") or {})),
        )

        st.session_state["approved_publication_candidate"] = analysis_payload["publication_candidate"]

    except Exception as e:
        _app_run_log("optimization_finished", status="error", error_type=type(e).__name__)
        stopped_timer = _abort_optimization_plan_timer(str(e))
        timer_summary = _timer_summary(stopped_timer)
        if timer_summary:
            optimization_timer_placeholder.warning(timer_summary[1])
        st.error(f"Error: {e}")


candidate = st.session_state.get("approved_publication_candidate")
if candidate:
    st.subheader("Approve public publication")
    st.caption(
        "This publishes the already calculated allocation. It does not run the optimizer again. "
        "Repeating the same run is idempotent."
    )
    try:
        configured_publisher_token = str(st.secrets["public_basket"]["publisher_token"]).strip()
    except Exception:
        configured_publisher_token = ""
    if len(configured_publisher_token) < 32:
        st.warning("Configure public_basket.publisher_token (at least 32 characters) to enable publication.")
    else:
        with st.form(f"publish_approved_{candidate['run_id']}"):
            supplied_token = st.text_input("Publisher token", type="password")
            approval_phrase = st.text_input("Type PUBLISH PUBLIC-01 to approve")
            approval_checked = st.checkbox("I reviewed the final whole-percent allocation")
            publish_now = st.form_submit_button("Publish approved portfolio", type="primary")
        if publish_now:
            if not hmac.compare_digest(supplied_token, configured_publisher_token):
                st.error("Publisher token is invalid.")
            elif approval_phrase.strip().upper() != "PUBLISH PUBLIC-01" or not approval_checked:
                st.error("Complete both approval confirmations before publishing.")
            else:
                try:
                    with connect_public_basket_db(get_public_basket_database_url()) as conn:
                        published = publish_approved_portfolio(
                            conn,
                            basket_id=candidate["basket_id"], run_id=candidate["run_id"],
                            as_of=datetime.fromisoformat(candidate["as_of"]),
                            calculation_version=candidate["calculation_version"],
                            strategy_version=candidate["strategy_version"],
                            constituents=candidate["constituents"], cash_weight=candidate["cash_weight"],
                        )
                    st.success(
                        f"Published immutable portfolio version {published['portfolio_version']} "
                        f"as {published['publication_id']}."
                    )
                    st.json(published)
                except Exception as exc:
                    st.error(f"Publication failed without changing the approved allocation: {exc}")
