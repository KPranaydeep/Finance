# Complete NAV history fix

Reviewed against Finance main commit 5c0f0f22ba5ec606de81f2e84e9d6cf096b79da9.

Problem: readers selected the newest calculation independently for each date. A shorter research window could retain older dates, and the daily_nav conflict rule could preserve stale values from a previous run. This could create artificial returns/drawdowns and affect forecast inputs.

Each successful NAV calculation now appends its full result to public_model_nav_snapshots. Performance, forecast generation, forecast realization and the smoke test all read the latest complete snapshot for the basket. Dates are never filled from previous snapshots. A SHA-256 check detects changes to the stored payload. Old daily_nav rows are retained for legacy inspection.

Upload all included Python files to the repository root and the test file under tests/. Commit and run Public portfolio daily trust update from the new commit. The NAV job creates the new table and first snapshot automatically with the configured database role. No manual SQL or ledger reset is required. Until that first run, readers return no model history rather than display mixed history.

Keep your chosen PUBLIC_MODEL_BACKFILL_TRADING_DAYS setting. Forecast records include the NAV snapshot hash and a calculation identifier containing that hash, so changed input history creates a new record even on the same day. Refresh the page after its five-minute cache expires.

The fix does not change price data, allocation formulas, cost assumptions or the separate evidence-export rejection. It does not prove any particular return or drawdown is correct; it removes one confirmed source of inconsistent history.

Verification: four unit tests passed (shorter rerun, basket isolation/missing history, hash tampering, invalid history); all changed Python files compile. Live PostgreSQL and the full repository test suite were not run locally.
