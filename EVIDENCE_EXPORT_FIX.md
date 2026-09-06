# Simulation evidence export

Based on Finance main commit d50bf1e91ba6bd8fd8acabb69712e6b6e1f6c432.

Upload public_portfolio_performance.py and public_release_checks.py to the repository root, and tests/test_evidence_export.py under tests/. Commit and refresh Streamlit after deployment.

The page offers Download simulation evidence when loaded NAV records explicitly contain is_backfill=true. The JSON includes research classification and retains original forecast provenance, records and hashes. Only the exact DEVELOPMENT_BACKFILL value in known forecast history_source fields is exempted from the non-production-marker check for this export. Credentials, database URLs, private paths and unrelated test markers continue to block export.

No additional secrets, database writes or workflow changes are required. The production smoke test and its configured backfill gate are unchanged.

Five unit tests passed, plus Python syntax checks. Streamlit rendering and the live database were not exercised locally. This fix does not validate the reported returns, forecasts, cost assumptions or future performance.
