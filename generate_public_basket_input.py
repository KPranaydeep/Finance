"""
Generate today's frozen input JSON for public_basket_optimizer_adapter.py
from universal_portfolio_backup.csv.

This replaces the placeholder "write today's frozen input" step in the
weekly GitHub Actions workflow with something that actually reads live
holdings and live prices.

Assumptions (check against your actual CSV before trusting this):
- One row per currently-held position (if the backup also contains fully
  exited/zero-quantity positions, filter those out below).
- "Average Price" is cost basis, NOT current price -- current price is
  fetched live via yfinance to compute weights and the FX rate.
- Non-INR rows get their FX rate from Yahoo Finance (e.g. "USDINR=X").

Usage:
    python generate_public_basket_input.py \
        --csv universal_portfolio_backup.csv \
        --scheduled-session-date 2026-09-07 \
        --out today_input.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

FX_TICKER_TEMPLATE = "{ccy}INR=X"  # e.g. USDINR=X, EURINR=X


def load_holdings(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
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

    return df.reset_index(drop=True)


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


def fetch_latest_prices(tickers: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    missing = []
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            missing.append(ticker)
            continue
        prices[ticker] = float(hist["Close"].iloc[-1])
    if missing:
        raise RuntimeError(f"Could not fetch live prices for: {', '.join(missing)}")
    return prices


def build_payload(df: pd.DataFrame, scheduled_session_date: str) -> dict:
    currencies = set(df["Currency"].astype(str).str.upper())
    fx_rates = fetch_fx_rates(currencies)

    tickers = df["Yahoo Ticker"].astype(str).str.strip().tolist()
    prices = fetch_latest_prices(tickers)

    market_values = []
    for _, row in df.iterrows():
        ticker = str(row["Yahoo Ticker"]).strip()
        ccy = str(row["Currency"]).upper()
        price = prices[ticker]
        fx = fx_rates[ccy]
        market_values.append(row["Quantity"] * price * fx)

    total_value = sum(market_values)
    if total_value <= 0:
        raise RuntimeError("Total portfolio market value is zero or negative.")

    portfolio_rows = []
    for (_, row), value in zip(df.iterrows(), market_values):
        ticker = str(row["Yahoo Ticker"]).strip()
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
        "scheduled_session_date": scheduled_session_date,
        "market_data_cutoff": now,
        "input_created_at": now,
        "portfolio": portfolio_rows,
        # "settings" intentionally omitted -> adapter applies DEFAULT_SETTINGS.
        # Override here if you want different optimizer parameters.
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="universal_portfolio_backup.csv")
    parser.add_argument("--scheduled-session-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="today_input.json")
    args = parser.parse_args()

    df = load_holdings(args.csv)
    payload = build_payload(df, args.scheduled_session_date)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out}: {len(payload['portfolio'])} positions, "
          f"scheduled_session_date={args.scheduled_session_date}")


if __name__ == "__main__":
    main()
