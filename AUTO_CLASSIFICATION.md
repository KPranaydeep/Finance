# Automatic review classifications

The page and model-review workflow classify every constituent in memory, with no PostgreSQL or policy-file writes. `public_review_policy.json` contains owner decisions only; it no longer contains or accumulates ticker names.

Every run resolves the active publications automatically, so no manual ticker entry or generated policy artifact is required. Classification remains fail-closed: an unavailable metadata source or unknown instrument stops the review rather than applying equity tax rules by guesswork.

Source: NSE equity master and ETF list:
- https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
- https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv

ETF Underlying categories map to the existing model: EQUITY -> equity_etf; GLOBAL INDICES and COMMODITY -> listed_non_equity_etf; DEBT -> specified_debt_etf. Ordinary company shares require the equity master, an INE ISIN and an eligible share series. No symbol-name guesses. Hybrid/unknown instruments, changed feed schemas and source outages remain explicit blockers. Source metadata is cached in-process for up to one hour.

No production database is accessed during classification tests. No secrets changes are required.
