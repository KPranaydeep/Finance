# Public portfolio production release

## Environment isolation

- Protect GitHub environments named `TEST` and `PRODUCTION`.
- Store a different `PUBLIC_BASKET_DATABASE_URL` in each environment.
- Never run destructive or fixture tests against `PRODUCTION`.
- Keep `publisher_token` only in private Streamlit secrets.

## Release gate

1. Run the complete trust-layer test suite: `python -m pytest -q tests/test_public_portfolio_trust.py`.
2. Run the private rebalancer and download its complete analysis JSON.
3. Review the `publication_candidate` allocation.
4. Upload that JSON to Public Portfolio Publisher and review the fingerprinted preview.
5. Publish once. Confirm a new `Pxxx` version appears on the public landing page.
6. Confirm NAV, cash-flow conventions, historical metrics, and basket-scoped audit status.
7. Run `update_public_forecasts.py` against TEST, then enable the production workflow.
8. Inspect the evidence bundle for private identifiers, credentials, local paths, debug data, and test records.
9. Perform a fresh public-page smoke test in a logged-out browser.

## Explicit production configuration

Set these variables on scheduled jobs: `PUBLIC_PORTFOLIO_ENV=PRODUCTION`,
`PUBLIC_BASKET_ID=PUBLIC-01`, `PUBLIC_PORTFOLIO_TIMEZONE=Asia/Kolkata`,
`PUBLIC_PERFORMANCE_VERSION=performance-v1`, `PUBLIC_FORECAST_VERSION=forecast-v1`,
`PUBLIC_FORECAST_METHOD=historical-bootstrap-14d-v1`, `PUBLIC_CACHE_TTL_SECONDS=300`,
and `PUBLIC_REFRESH_POLICY=weekdays-after-market-close`. Store only the database URL and
publisher token as secrets. TEST must use a separate database or schema and synthetic fixtures.

Run the non-destructive production gate with the same production variables:

`python production_smoke_test.py`

It must report `PASS` before release. It does not create, update, or delete portfolio records.

## Database migration

`init_trust_schema()` is repeatable. It creates the four public trust entities, constraints,
immutability triggers, and basket-scoped audit table. If the earlier v3 publication and position
tables exist, their rows are copied with `ON CONFLICT DO NOTHING`; the old tables are left intact
for rollback. Take a provider snapshot before first production execution.

## Required cleanup

After the production smoke test, remove these temporary Streamlit pages:

- `pages/Filter_Universal_By_GoodTickers.py`
- `pages/Probe_YF_Tickers.py`
- `pages/YY_Generate_and_Run_From_CSV.py`
- `pages/ZZ_Manual_Run_Weekly_Signal.py`
- `pages/ZZ_Rename_Public_Basket_ID.py`
- `pages/ZZ_Run_From_CSV_AvgPrice.py`

Do not delete legitimate production publications or ledger history.

## Correcting a mistaken publication

Never delete or update an immutable publication. Temporarily deploy
`pages/ZZ_Correct_Public_Portfolio_Version.py`, authenticate with the publisher token, and append a
`VOIDED_DUPLICATE` correction referencing the authoritative version. Corrected versions remain in
History with their reason but are excluded from the current portfolio, new forecasts, and NAV
calculation version 4. Run the daily trust workflow, verify the evidence bundle, then remove the
temporary correction page. New exact-allocation replicas are rejected by allocation fingerprint,
even when their run IDs or timestamps differ.
