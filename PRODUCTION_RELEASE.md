# Public portfolio production release

## Environment isolation

- Protect GitHub environments named `TEST` and `PRODUCTION`.
- Store a different `PUBLIC_BASKET_DATABASE_URL` in each environment.
- Never run destructive or fixture tests against `PRODUCTION`.
- Keep `publisher_token` only in private Streamlit secrets.

## Release gate

1. Run the complete test suite against TEST.
2. Run the private rebalancer and download its complete analysis JSON.
3. Review the `publication_candidate` allocation.
4. Upload that JSON to Public Portfolio Publisher and review the fingerprinted preview.
5. Publish once. Confirm a new `Pxxx` version appears on the public landing page.
6. Confirm NAV, cash-flow conventions, historical metrics, and basket-scoped audit status.
7. Run `update_public_forecasts.py` against TEST, then enable the production workflow.
8. Inspect the evidence bundle for private identifiers, credentials, local paths, debug data, and test records.
9. Perform a fresh public-page smoke test in a logged-out browser.

## Required cleanup

After the production smoke test, remove these temporary Streamlit pages:

- `pages/Filter_Universal_By_GoodTickers.py`
- `pages/Probe_YF_Tickers.py`
- `pages/YY_Generate_and_Run_From_CSV.py`
- `pages/ZZ_Manual_Run_Weekly_Signal.py`
- `pages/ZZ_Rename_Public_Basket_ID.py`
- `pages/ZZ_Run_From_CSV_AvgPrice.py`

Do not delete legitimate production publications or ledger history.
