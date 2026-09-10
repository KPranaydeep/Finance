# Step 2 — gap-safe review history and INR prices

Upload all included files to the matching Finance/main paths and commit together. Then run NEW daily-trust and model-review workflows on main. Refresh Streamlit after deployment/cache expiry.

Changes:
- Review history uses only complete adjacent-session returns; no forward fill or multi-day moves mislabeled as daily returns. Bootstrap blocks restart at gaps, and walk-forward test windows spanning gaps are excluded.
- Provisional review can use the prior completed session when today's common prices are incomplete, with the actual date disclosed. Frozen-model valuations still require complete entry/current prices.
- Direct USD quotes are converted with same-date USD/INR for the price table, allocation inputs and newly calculated NAV. Unknown currencies fail rather than assuming INR. NSE-listed ETFs remain INR and are not converted twice.
- NAV calculation version is now 7, retaining existing snapshots. Re-run the daily trust workflow to calculate new INR-based snapshots and dependent forecasts. Conversion is an end-of-day convention, not synchronized intraday valuation.
- Actual P006 data check completed: 640 usable daily returns through 2026-09-08, rather than silently using incomplete September 9 prices. Runtime dates are not hardcoded.

IMPORTANT FOR P007: it contains direct overseas listings. The review fee/tax engine remains NSE-only. It now reports FOREIGN_REVIEW_COST_MODEL_REQUIRED instead of asking you to misclassify US stocks as NSE shares. Overseas brokerage, FX/remittance spreads, withholding/ADR treatment and investor-specific taxes are not modeled. Existing allocation-cost and NAV-drag assumptions remain estimates, not verified foreign-account net costs. This release does not claim to deliver a valid foreign post-tax target-crossing date.

78 review tests and 3 benchmark tests passed; syntax checks passed. Production DB/workflows were not executed. No policy reset or actual trades. This package retains step-1 automatic NSE classifications.

The shared period is calculated from available public-review prices, not copied from a private optimizer export. Importing private coverage metadata remains separate.

Streamlit skill guidance informed native currency/coverage captions. No custom styling changes.
