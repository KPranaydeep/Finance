"""INR valuation for INR and USD quotes. Never convert NSE quotes twice."""
import numpy as np
import pandas as pd
import yfinance as yf


def to_inr(closes, currencies, usd_inr=None):
    result = closes.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index)).tz_localize(None).normalize()
    if result.index.duplicated().any():
        raise ValueError("DUPLICATE_PRICE_DATES")
    fx = None
    if usd_inr is not None:
        fx = pd.to_numeric(usd_inr, errors="coerce").copy()
        fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).tz_localize(None).normalize()
        if fx.index.duplicated().any():
            raise ValueError("DUPLICATE_FX_DATES")
        fx = fx.where(np.isfinite(fx) & (fx > 0))
    for ticker in result:
        currency = currencies.get(ticker)
        values = pd.to_numeric(result[ticker], errors="coerce")
        values = values.where(np.isfinite(values) & (values > 0))
        if currency == "USD":
            if fx is None:
                raise ValueError("USD_INR_HISTORY_REQUIRED")
            values = values * fx.reindex(result.index)
        elif currency != "INR":
            raise ValueError("UNSUPPORTED_QUOTE_CURRENCY")
        result[ticker] = values
    return result


def download_inr(tickers, *, start=None, end=None, period=None, auto_adjust=False):
    currencies = {}
    for ticker in tickers:
        currency = "INR" if ticker.endswith(".NS") else yf.Ticker(ticker).get_history_metadata().get("currency")
        if currency not in {"INR", "USD"}:
            raise ValueError("UNSUPPORTED_QUOTE_CURRENCY")
        currencies[ticker] = currency
    requested = list(tickers) + (["INR=X"] if "USD" in currencies.values() else [])
    options = {"period": period} if period else {"start": start, "end": end}
    data = yf.download(requested, interval="1d", auto_adjust=auto_adjust,
                       progress=False, threads=False, group_by="column", **options)
    if data.empty:
        raise ValueError("PRICE_HISTORY_UNAVAILABLE")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested[0])
    return to_inr(close.reindex(columns=list(tickers)), currencies,
                  close.get("INR=X")), currencies
