"""Build an event-driven public-basket input snapshot from a holdings CSV."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import yfinance as yf

FX_TICKER_TEMPLATE = "{ccy}INR=X"  # e.g. USDINR=X, EURINR=X
DEFAULT_BATCH_SIZE = 25
DEFAULT_RETRIES = 3


def validate_holdings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"Symbol", "Yahoo Ticker", "Currency", "Quantity", "Average Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Average Price"] = pd.to_numeric(df["Average Price"], errors="coerce")

    # Drop exited positions / bad rows. Adjust this filter if your backup
    # encodes "closed" differently (e.g. a Status column).
    df = df.dropna(subset=["Quantity", "Average Price"])
    df = df[df["Quantity"] > 0]

    if df.empty:
        raise ValueError("No open positions found in the CSV after filtering.")

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Yahoo Ticker"] = df["Yahoo Ticker"].astype(str).str.strip().str.upper()
    df["Currency"] = df["Currency"].astype(str).str.strip().str.upper()
    invalid = df[
        df["Symbol"].eq("")
        | df["Yahoo Ticker"].eq("")
        | df["Currency"].eq("")
        | df["Average Price"].le(0)
    ]
    if not invalid.empty:
        raise ValueError("Open-position rows contain blank identifiers or invalid prices.")
    if df["Yahoo Ticker"].duplicated().any():
        duplicates = sorted(df.loc[df["Yahoo Ticker"].duplicated(False), "Yahoo Ticker"].unique())
        raise ValueError("Duplicate Yahoo tickers: " + ", ".join(duplicates))
    return df.reset_index(drop=True)


def load_holdings(csv_path: str) -> pd.DataFrame:
    return validate_holdings(pd.read_csv(csv_path))


def fetch_fx_rates(currencies: set[str]) -> dict[str, float]:
    rates = {"INR": 1.0}
    for ccy in currencies:
        if ccy == "INR" or ccy in rates:
            continue
        ticker = FX_TICKER_TEMPLATE.format(ccy=ccy)
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            raise RuntimeError(f"Could not fetch FX rate for {ccy} via {ticker}")
        rates[ccy] = float(hist["Close"].iloc[-1])
    return rates


def _download_with_retries(tickers: list[str], retries: int) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return yf.download(
                tickers,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
                timeout=30,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def _close_frame(data: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            frame = data["Close"]
        elif "Close" in data.columns.get_level_values(1):
            frame = data.xs("Close", axis=1, level=1)
        else:
            return pd.DataFrame()
    elif "Close" in data.columns and len(requested) == 1:
        frame = data[["Close"]].rename(columns={"Close": requested[0]})
    else:
        return pd.DataFrame()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=requested[0])
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame.apply(pd.to_numeric, errors="coerce")


def fetch_latest_prices_with_missing(
    tickers: list[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retries: int = DEFAULT_RETRIES,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Fetch in bounded batches and report unresolved tickers without mislabelling all."""
    unique = [
        ticker
        for ticker in dict.fromkeys(str(value).strip().upper() for value in tickers)
        if ticker
    ]
    prices: dict[str, float] = {}
    total_batches = max((len(unique) + batch_size - 1) // batch_size, 1)

    for batch_number, start in enumerate(range(0, len(unique), batch_size), start=1):
        batch = unique[start : start + batch_size]
        try:
            frame = _close_frame(_download_with_retries(batch, retries), batch)
        except Exception:
            frame = pd.DataFrame()

        for ticker in batch:
            if ticker not in frame.columns:
                continue
            valid = frame[ticker].dropna()
            if valid.empty:
                continue
            price = float(valid.iloc[-1])
            if math.isfinite(price) and price > 0:
                prices[ticker] = price
        if progress:
            progress(batch_number, total_batches)

    missing = [ticker for ticker in unique if ticker not in prices]
    return prices, missing


def fetch_latest_prices(tickers: list[str]) -> dict[str, float]:
    prices, missing = fetch_latest_prices_with_missing(tickers)
    if missing:
        sample = ", ".join(missing[:25])
        suffix = f" (+{len(missing) - 25} more)" if len(missing) > 25 else ""
        raise RuntimeError(
            f"Could not fetch valid prices for {len(missing)} tickers: {sample}{suffix}"
        )
    return prices


def build_payload(
    df: pd.DataFrame,
    *,
    prices: dict[str, float] | None = None,
    fx_rates: dict[str, float] | None = None,
) -> dict:
    currencies = set(df["Currency"].astype(str).str.upper())
    fx_rates = fx_rates or fetch_fx_rates(currencies)

    tickers = df["Yahoo Ticker"].astype(str).str.strip().str.upper().tolist()
    prices = prices or fetch_latest_prices(tickers)

    market_values = []
    for _, row in df.iterrows():
        ticker = str(row["Yahoo Ticker"]).strip().upper()
        ccy = str(row["Currency"]).upper()
        price = prices[ticker]
        fx = fx_rates[ccy]
        market_values.append(row["Quantity"] * price * fx)

    total_value = sum(market_values)
    if total_value <= 0:
        raise RuntimeError("Total portfolio market value is zero or negative.")

    portfolio_rows = []
    for (_, row), value in zip(df.iterrows(), market_values):
        ticker = str(row["Yahoo Ticker"]).strip().upper()
        ccy = str(row["Currency"]).upper()
        portfolio_rows.append(
            {
                "Symbol": str(row["Symbol"]).strip(),
                "Yahoo Ticker": ticker,
                "Currency": ccy,
                "Quantity": float(row["Quantity"]),
                "Average Price": float(row["Average Price"]),
                "FX to INR": fx_rates[ccy],
                "Weight": value / total_value,
            }
        )

    now = datetime.now(timezone.utc).isoformat()
    return {
        "market_data_cutoff": now,
        "input_created_at": now,
        "portfolio": portfolio_rows,
        # "settings" intentionally omitted -> adapter applies DEFAULT_SETTINGS.
        # Override here if you want different optimizer parameters.
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="universal_portfolio_backup.csv")
    parser.add_argument("--out", default="today_input.json")
    args = parser.parse_args()

    df = load_holdings(args.csv)
    payload = build_payload(df)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out}: {len(payload['portfolio'])} positions")


if __name__ == "__main__":
    main()
