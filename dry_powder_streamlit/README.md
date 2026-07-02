# Dry Powder Planner — Indian Equity Portfolios

A deployable Streamlit application that helps an Indian retail investor:

1. choose an NSE/NIFTY benchmark manually or statistically match one to uploaded holdings;
2. estimate a transparent minimum dry-powder allocation;
3. simulate the previous completed quarter with and without dry powder;
4. deploy the reserve in equal tranches at preset drawdown levels; and
5. export the quarter-level simulation.

## Important distinction

- **Emergency fund:** personal living-expense reserve; keep it outside this app.
- **Dry powder:** liquid investable capital reserved for portfolio rebalancing.
- **Tail hedge:** derivatives such as put options; this app does not model them.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and add all files from this folder.
2. In Streamlit Community Cloud, choose **Create app**.
3. Select the repository, branch and `app.py`.
4. Deploy. `requirements.txt` supplies the Python dependencies.

No API key is required. Automatic mode first tries NSE’s public historical-index endpoint and then falls back to `yfinance`. Either provider can throttle or block cloud traffic, so the app also accepts a Date/Close CSV. Market data is end-of-day/delayed, not an exchange-grade real-time feed.

## Holdings CSV format

Use either market values:

```csv
ticker,value
RELIANCE,350000
HDFCBANK,250000
INFY,200000
```

or weights:

```csv
ticker,weight
RELIANCE,35
HDFCBANK,25
INFY,20
```

Tickers without an exchange suffix are converted to `.NS`.

## Recommendation formula

The minimum cash estimate is:

```text
minimum cash weight = max(0, 1 - tolerable drawdown / stress drawdown)
```

It assumes cash remains stable while the selected benchmark experiences the stress drawdown. The chosen policy floor is then applied, and the result is rounded upward to the next 2.5 percentage points.

Example: if the benchmark stress drawdown is 35% and the investor can tolerate 25%, the formula minimum is approximately 28.6%.

## Backtest assumptions

- The “fully invested” portfolio starts 100% in the selected benchmark.
- The dry-powder strategy starts with the recommended reserve and the rest in the benchmark.
- Reserve capital accrues the user-entered annual cash yield.
- Equal tranches are deployed when the benchmark falls from its running quarter peak through selected thresholds.
- Taxes, brokerage, bid/ask spreads, fund expenses, tracking error and slippage are excluded.
- ETF proxies may differ from their underlying indices.

## Suggested next production upgrades

- store user portfolios and policies in a database;
- authenticate users;
- use a licensed/official market-data feed;
- support asset-class and sector decomposition;
- include debt/liquid-fund NAVs, taxation and exit loads;
- add rolling-quarter and full-cycle backtests;
- alert users when a deployment threshold is reached.

## Disclaimer

This project is an educational decision-support tool, not investment advice or a promise of loss prevention.
