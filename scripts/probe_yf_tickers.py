# scripts/probe_yf_tickers.py
import time
import argparse
import csv

import pandas as pd
import yfinance as yf

def probe_one(ticker, timeout=30):
    try:
        df = yf.download(ticker, period="10d", threads=False, progress=False, timeout=timeout, auto_adjust=True)
        if df is None or df.empty:
            return False
        numeric = df.apply(pd.to_numeric, errors="coerce")
        return bool(numeric.notna().any().any())
    except Exception:
        return False

def main(input_path, out_good="good_tickers.csv", out_bad="bad_tickers.csv", delay=0.5):
    # Input can be a text file (one ticker per line) or a CSV with column 'Ticker'
    tickers = []
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path, dtype=str)
        # try common column names
        if "Ticker" in df.columns:
            tickers = df["Ticker"].dropna().astype(str).str.strip().tolist()
        elif "Yahoo Ticker" in df.columns:
            tickers = df["Yahoo Ticker"].dropna().astype(str).str.strip().tolist()
        else:
            # use first column
            tickers = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    else:
        with open(input_path, "r", encoding="utf-8") as fh:
            tickers = [line.strip() for line in fh if line.strip()]

    tickers = list(dict.fromkeys(tickers))  # dedupe while preserving order

    good = []
    bad = []
    for t in tickers:
        ok = probe_one(t)
        (good if ok else bad).append(t)
        print(f"{t}: {'OK' if ok else 'NO'}")
        time.sleep(delay)

    # write outputs
    with open(out_good, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Ticker"])
        for t in good:
            writer.writerow([t])

    with open(out_bad, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Ticker"])
        for t in bad:
            writer.writerow([t])

    print(f"Done. Good: {len(good)}, Bad: {len(bad)}")
    print(f"Wrote: {out_good}, {out_bad}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe tickers with yfinance (10d).")
    parser.add_argument("input", help="Path to tickers file (txt) or CSV (first column or 'Ticker'/'Yahoo Ticker').")
    parser.add_argument("--good", default="good_tickers.csv")
    parser.add_argument("--bad", default="bad_tickers.csv")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between probes (reduce rate-limit).")
    args = parser.parse_args()
    main(args.input, out_good=args.good, out_bad=args.bad, delay=args.delay)
