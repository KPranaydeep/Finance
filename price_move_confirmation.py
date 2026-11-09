"""Bounded, optional second-provider evidence for extreme US listing moves."""

import io
import re

import pandas as pd
import requests


def download_confirmation(ticker, start_date, end_date, *, fetch=None):
    """Read Stooq daily closes for supported US symbols; unavailable stays unknown.

    No retries, subscription dependency, credential storage or fallback to Yahoo.
    Unsupported listings and provider access failures never certify a move.
    """
    ticker = str(ticker).strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,9}", ticker):
        return None
    try:
        response = (fetch or requests.get)(
            "https://stooq.com/q/d/l/",
            params={"s": ticker.lower() + ".us", "i": "d",
                    "d1": pd.Timestamp(start_date).strftime("%Y%m%d"),
                    "d2": pd.Timestamp(end_date).strftime("%Y%m%d")},
            timeout=(3, 5),
        )
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if not {"Date", "Close"}.issubset(frame.columns):
            return None
        close = pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                          index=pd.to_datetime(frame["Date"], errors="coerce"))
        close = close.loc[close.index.notna()].sort_index()
        close = close.loc[~close.index.duplicated(keep="last")]
        if len(close.dropna()) < 2 or (close.dropna() <= 0).any():
            return None
        return {"confirmation_close": close, "confirmation_source": "Stooq daily US listing"}
    except (requests.RequestException, ValueError, TypeError, KeyError, pd.errors.ParserError):
        return None
