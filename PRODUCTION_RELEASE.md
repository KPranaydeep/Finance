# Public portfolio production release

This checklist is the canonical transition from private development to the public
portfolio. It does not authorize deletion of legitimate production history.

## Environment isolation

- Protect GitHub environments named `TEST` and `PRODUCTION`.
- Use different PostgreSQL databases or schemas for tests and production.
- Keep `PUBLIC_BASKET_DATABASE_URL`, the Streamlit `publisher_token`, and optional
  notification credentials in their respective secret stores.
- Never commit `.streamlit/secrets.toml`, broker reports, private holdings, database
  exports or private optimizer analysis JSON.
- Never run fixture, reset or destructive tests against `PRODUCTION`.

## Current production contract

The scheduled daily workflow uses these values:

| Variable | Production value |
|---|---|
| `PUBLIC_PORTFOLIO_ENV` | `PRODUCTION` |
| `PUBLIC_BASKET_ID` | `PUBLIC-01` |
| `PUBLIC_PORTFOLIO_TIMEZONE` | `Asia/Kolkata` |
| `PUBLIC_PERFORMANCE_VERSION` | `performance-v1` |
| `PUBLIC_FORECAST_VERSION` | `forecast-v3-28calendar` |
| `PUBLIC_FORECAST_METHOD` | `historical-calendar-block-28d-v1` |
| `PUBLIC_CACHE_TTL_SECONDS` | `300` |
| `PUBLIC_REFRESH_POLICY` | `weekdays-after-market-close` |

`PUBLIC_MODEL_BACKFILL_TRADING_DAYS` is a development-only GitHub environment
variable. It defaults to `0` and must be `0` for the public release.

## Release gate

1. Freeze optimizer, publication and policy changes for the release candidate.
2. Create and verify a restorable PostgreSQL provider snapshot.
3. Run `python -m pytest -q` from a clean checkout.
4. Run the private rebalancer and download its complete analysis JSON.
5. Review the exact `publication_candidate`, its prices, currencies, weights and
   calculation versions.
6. Upload that JSON to Public Basket Publisher and verify the fingerprinted,
   read-only preview before publishing once.
7. Confirm that the new `Pxxx` version, allocation and publication changes appear
   correctly on the public pages.
8. Run `update_public_nav.py` and `update_public_forecasts.py` with the production
   configuration.
9. Run `python production_smoke_test.py`; require `PASS`.
10. Inspect the evidence export for credentials, private identifiers, local paths,
    test records and development markers.
11. Verify the allocation card, track-record page and review panel in a logged-out
    browser at desktop and mobile widths.
12. Confirm the scheduled daily and model-review workflows complete successfully.

The smoke test is read-only. It verifies configuration, current publication,
weights, NAV, forecast availability, audit integrity and public-data hygiene.

## Development-ledger reset

The repository deliberately contains no automatic production reset. Immediately
before the inaugural public release:

1. keep the verified provider snapshot;
2. set `PUBLIC_MODEL_BACKFILL_TRADING_DAYS=0`;
3. use the separately reviewed, explicitly targeted reset procedure;
4. publish the clean inaugural portfolio;
5. regenerate NAV and outlook records; and
6. rerun the production smoke test.

Do not reset merely because a publication is inconvenient. During normal
operation, corrections are append-only.

## Operator surface verification

The publisher is the only operator page shipped with the public application. If it
remains deployed, verify that no publication action is possible without the private
token.

## Correcting a mistaken publication

Never update or delete an immutable publication. Append a correction referencing
the affected publication and retain the reason in history. Corrected publications
remain auditable but are excluded from current allocation, new forecasts and active
NAV calculations. Exact-allocation replicas are rejected by fingerprint even when
run IDs or timestamps differ.

No correction page is permanently shipped in the current repository. Any temporary
operator correction surface must be authenticated, reviewed separately and removed
immediately after use.

## Rollback

- Disable `PUBLIC_REVIEW_ENABLED` to stop review-monitor writes and alerts.
- Disable scheduled workflows before investigating a suspected calculation defect.
- Revert application code normally; do not roll back by editing immutable database
  records.
- Restore a database snapshot only for a genuine disaster-recovery event with the
  exact target and consequences reviewed in advance.
