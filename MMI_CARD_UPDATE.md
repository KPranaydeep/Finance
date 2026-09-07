# Tickertape MMI card

Upload public_market_mood.py and public_portfolio_performance.py to the repository root. Upload tests/test_market_mood.py under tests/. Commit and allow Streamlit to redeploy.

No workflow, secret, database migration or new package is needed. Based on main commit 796fdf3c75257a7901234540b8139e7a29ce9d73.

The page fetches Tickertape's public MMI page on load/rerun with a 15-minute shared cache, reads its structured nowData indicator and source timestamp, and displays the score, zone, source link and Asia/Kolkata timestamp. An idle browser does not poll; the next page load/rerun refreshes an expired cache.

The official scale remains 0–100. Values outside the user's expected 10–90 interval are not clipped. Readings over 24 hours old are labelled Older reading. Readings over seven days old, invalid scores, missing timestamps and failed requests show unavailable. Network requests have a six-second timeout and do not bypass access restrictions.

MMI does not enter optimization, rebalancing, execution prompts, forecasts or database records. It is a display-only card.

Validation: five unit tests passed, Python files compiled, and the live public page fetch succeeded locally. Streamlit Cloud access and rendering still require confirmation after upload; upstream HTML structure or access policies may change.
