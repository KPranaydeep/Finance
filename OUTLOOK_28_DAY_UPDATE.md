# 28-day outlook update

Upload the included files to the Finance repository, preserving the tests/ and .github/workflows/ paths. Commit and run Public portfolio daily trust update using Run workflow on the new commit.

Keep PUBLIC_MODEL_BACKFILL_TRADING_DAYS at your chosen development value (for example 756). This setting controls input history; it does not set the forecast horizon.

The new method is historical-calendar-block-28d-v1 and the immutable calculation version is forecast-v3-28calendar. New records use 28 calendar days. Existing 14-day records remain in the ledger and retain their original evaluation rules.

The new outlook needs at least 126 NAV observations and 20 complete four-week scenarios. Until its first record exists, the page shows a waiting message. Refresh after the five-minute cache interval.

Historical blocks overlap. Quantiles and frequencies describe historical scenarios, with unknown future allocations and market regimes. The methodology file documents the calculation and expiry conventions.

Verification: four standalone unit tests passed with the bundled Python runtime; changed Python files passed compilation. The full repository pytest suite and live PostgreSQL workflow were not run locally.
