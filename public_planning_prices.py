"""Read-only planning quotes with a bounded retry for partial provider failures."""

import logging

import numpy as np
import pandas as pd

from public_price_currency import download_inr


LOGGER = logging.getLogger(__name__)


def latest_planning_prices(tickers):
    tickers = tuple(dict.fromkeys(tickers))
    if not tickers:
        return {}
    # Same supported INR / direct-US universe as the public valuation path.
    currencies = {ticker: "INR" if ticker.endswith(".NS") else "USD"
                  for ticker in tickers}
    result = {}

    def fetch(requested, period):
        try:
            close, resolved = download_inr(
                list(requested), period=period, auto_adjust=False,
                currencies=currencies, timeout=10,
            )
            for ticker in requested:
                if close is None or ticker not in close:
                    continue
                series = pd.to_numeric(close[ticker], errors="coerce")
                series = series.where(np.isfinite(series) & (series > 0)).dropna()
                if not series.empty:
                    series = series.sort_index()
                    result[ticker] = {
                        "price": float(series.iloc[-1]), "currency": "INR",
                        "source_currency": resolved[ticker],
                        "price_as_of": pd.Timestamp(series.index[-1]).date().isoformat(),
                    }
        except Exception:
            LOGGER.warning("Planning quote batch unavailable (%s securities)",
                           len(requested), exc_info=True)

    fetch(tickers, "7d")
    missing = tuple(ticker for ticker in tickers if ticker not in result)
    # Recover a small partial failure, without retrying a whole provider outage.
    if 0 < len(missing) <= 5:
        fetch(missing, "1mo")
    return result
