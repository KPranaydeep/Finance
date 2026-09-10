# Automatic review classifications — first integration only

The page resolves missing classifications in memory, with no PostgreSQL or policy-file writes. The enabled model-review workflow automatically fills missing instrument_kinds in its checked-out public_review_policy.json before the monitor runs. Its completed JSON is downloadable as the resolved-public-review-policy Actions artifact.

GitHub runner files are temporary: this does not commit generated changes to main. Every run resolves missing entries again, so no manual ticker entry or artifact upload is required for normal operation. Existing entries, thresholds, approval and tax settings are preserved.

Source: NSE equity master and ETF list:
- https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
- https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv

ETF Underlying categories map to the existing model: EQUITY -> equity_etf; GLOBAL INDICES and COMMODITY -> listed_non_equity_etf; DEBT -> specified_debt_etf. Ordinary company shares require the equity master, an INE ISIN and an eligible share series. No symbol-name guesses. Hybrid/unknown instruments, changed feed schemas and source outages remain explicit blockers. Source metadata is cached for up to one hour; existing policy entries are not reclassified automatically.

Verified all 15 supplied P006 tickers against live NSE metadata. 70 local review tests passed, including category mapping, unknown rejection, override preservation and atomic/idempotent JSON updates. No production database accessed, no workflow run triggered and no push/deployment performed.

Install: upload the included files preserving folders, commit on main, then start a NEW Public portfolio model review run. No secrets changes. This release intentionally does not alter common-period/missing-session processing, which remains the next task.
