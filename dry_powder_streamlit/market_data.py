"""Market data access and uploaded-data parsing."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf


NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports-indices-historical-index-data",
}


def download_nse_index_close(index_name: str, start: date | str, end: date | str) -> pd.Series:
    """Download daily index closes from NSE's public historical-index endpoint.

    NSE may throttle or block automated cloud traffic. Callers should expose a CSV
    fallback and present a clear error rather than silently fabricating data.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts >= end_ts:
        raise ValueError("Start date must be before end date.")

    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    origin = "https://www.nseindia.com/reports-indices-historical-index-data"
    try:
        session.get(origin, timeout=20)
    except requests.RequestException:
        pass

    chunks = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=364), end_ts)
        url = (
            "https://www.nseindia.com/api/historical/indicesHistory"
            f"?indexType={quote(index_name.upper())}"
            f"&from={cursor.strftime('%d-%m-%Y')}"
            f"&to={chunk_end.strftime('%d-%m-%Y')}"
        )
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            payload = response.json()
            chunks.append(_nse_payload_to_close(payload, index_name))
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise ValueError(f"NSE historical-data request failed for {index_name}: {exc}") from exc
        cursor = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise ValueError(f"NSE returned no history for {index_name}.")
    close = pd.concat(chunks).sort_index()
    close = close[~close.index.duplicated(keep="last")].dropna()
    if close.empty:
        raise ValueError(f"NSE returned no usable close values for {index_name}.")
    close.name = index_name
    return close


def _nse_payload_to_close(payload: dict, index_name: str) -> pd.Series:
    data = payload.get("data", payload)
    records = []
    if isinstance(data, dict):
        for key in ("indexCloseOnlineRecords", "indexCloseRecords", "data"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
    elif isinstance(data, list):
        records = data
    if not records:
        raise ValueError("response contained no close records")

    df = pd.DataFrame(records)
    date_col = next((c for c in ("EOD_TIMESTAMP", "TIMESTAMP", "HistoricalDate", "Date", "date") if c in df.columns), None)
    close_col = next((c for c in ("EOD_CLOSE_INDEX_VAL", "CLOSE_INDEX_VAL", "CLOSE", "Close", "close") if c in df.columns), None)
    if not date_col or not close_col:
        raise ValueError(f"unexpected NSE response fields: {', '.join(map(str, df.columns[:12]))}")

    dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    values = pd.to_numeric(df[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=index_name).dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


def download_benchmark_close(
    nse_name: str,
    yahoo_symbol: str,
    start: date | str,
    end: date | str,
    provider: str = "Automatic",
) -> tuple[pd.Series, str]:
    if provider == "NSE official":
        return download_nse_index_close(nse_name, start, end), "NSE official historical data"
    if provider == "Yahoo/yfinance":
        return download_close(yahoo_symbol, start=start, end=end), "Yahoo Finance via yfinance"

    nse_error = None
    try:
        return download_nse_index_close(nse_name, start, end), "NSE official historical data"
    except Exception as exc:
        nse_error = exc
    try:
        return download_close(yahoo_symbol, start=start, end=end), "Yahoo Finance via yfinance (NSE fallback failed)"
    except Exception as yahoo_exc:
        raise ValueError(f"Both providers failed. NSE: {nse_error}. Yahoo: {yahoo_exc}") from yahoo_exc


def _extract_close(raw: pd.DataFrame, symbol: str | None = None) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance can return (Price, Ticker) or (Ticker, Price) columns.
        level0 = set(map(str, raw.columns.get_level_values(0)))
        level1 = set(map(str, raw.columns.get_level_values(1)))
        if "Close" in level0:
            close = raw["Close"]
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1)
        else:
            return pd.Series(dtype=float)
        if isinstance(close, pd.DataFrame):
            if symbol and symbol in close.columns:
                close = close[symbol]
            else:
                close = close.iloc[:, 0]
    elif "Close" in raw.columns:
        close = raw["Close"]
    elif "Adj Close" in raw.columns:
        close = raw["Adj Close"]
    else:
        return pd.Series(dtype=float)

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = symbol or "Close"
    return close.sort_index()


def download_close(
    symbol: str,
    start: date | str | None = None,
    end: date | str | None = None,
    period: str | None = None,
) -> pd.Series:
    kwargs = {
        "tickers": symbol,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
        "timeout": 20,
    }
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    raw = yf.download(**kwargs)
    close = _extract_close(raw, symbol)
    if close.empty:
        raise ValueError(
            f"No price history was returned for {symbol}. Try another symbol or upload a Date/Close CSV."
        )
    return close


def download_many(symbols: Iterable[str], start: date | str, end: date | str) -> pd.DataFrame:
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
        timeout=30,
    )
    if raw.empty:
        return pd.DataFrame()

    output = {}
    for symbol in symbols:
        try:
            series = _extract_symbol_from_multi(raw, symbol, len(symbols))
            if not series.empty:
                output[symbol] = series
        except (KeyError, IndexError, TypeError):
            continue
    return pd.DataFrame(output).sort_index()


def _extract_symbol_from_multi(raw: pd.DataFrame, symbol: str, symbol_count: int) -> pd.Series:
    if symbol_count == 1:
        return _extract_close(raw, symbol)
    if not isinstance(raw.columns, pd.MultiIndex):
        return pd.Series(dtype=float)

    level0 = set(map(str, raw.columns.get_level_values(0)))
    if "Close" in level0:
        close_df = raw["Close"]
        if symbol not in close_df.columns:
            return pd.Series(dtype=float)
        series = close_df[symbol]
    else:
        # Some yfinance versions may return ticker as the first level.
        if symbol not in level0:
            return pd.Series(dtype=float)
        sub = raw[symbol]
        if "Close" not in sub.columns:
            return pd.Series(dtype=float)
        series = sub["Close"]

    series = pd.to_numeric(series, errors="coerce").dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series.name = symbol
    return series


def parse_benchmark_csv(uploaded_file) -> pd.Series:
    df = pd.read_csv(uploaded_file)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    if "date" not in normalized:
        raise ValueError("Benchmark CSV must contain a Date column.")
    close_key = next((key for key in ("close", "adj close", "adjusted close", "price") if key in normalized), None)
    if not close_key:
        raise ValueError("Benchmark CSV must contain Close, Adj Close, Adjusted Close or Price.")

    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[normalized["date"]], errors="coerce"),
            "Close": pd.to_numeric(df[normalized[close_key]], errors="coerce"),
        }
    ).dropna()
    out = out.drop_duplicates("Date", keep="last").sort_values("Date")
    if len(out) < 20:
        raise ValueError("Benchmark CSV needs at least 20 valid daily observations.")
    return out.set_index("Date")["Close"]


def parse_holdings_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    ticker_key = next((k for k in ("ticker", "symbol", "nse symbol") if k in normalized), None)
    if not ticker_key:
        raise ValueError("Holdings CSV must contain a ticker or symbol column.")

    value_key = next((k for k in ("value", "market value", "current value", "amount") if k in normalized), None)
    weight_key = next((k for k in ("weight", "weight %", "allocation", "allocation %") if k in normalized), None)
    if not value_key and not weight_key:
        raise ValueError("Holdings CSV must contain either value/amount or weight/allocation.")

    out = pd.DataFrame()
    out["ticker"] = df[normalized[ticker_key]].astype(str).str.strip().str.upper()
    out = out[out["ticker"].ne("") & out["ticker"].ne("NAN")]
    out["ticker"] = out["ticker"].map(_normalize_indian_ticker)

    if value_key:
        out["raw_weight"] = pd.to_numeric(df.loc[out.index, normalized[value_key]], errors="coerce")
    else:
        out["raw_weight"] = pd.to_numeric(df.loc[out.index, normalized[weight_key]], errors="coerce")

    out = out.dropna(subset=["raw_weight"])
    out = out[out["raw_weight"] > 0]
    out = out.groupby("ticker", as_index=False)["raw_weight"].sum()
    if out.empty:
        raise ValueError("No positive holdings were found in the CSV.")
    out["weight"] = out["raw_weight"] / out["raw_weight"].sum()
    return out[["ticker", "weight"]].sort_values("weight", ascending=False)


def _normalize_indian_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith("^") or ticker.endswith((".NS", ".BO")):
        return ticker
    return f"{ticker}.NS"
