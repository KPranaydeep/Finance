# Public trust release manifest

This manifest identifies the production trust boundary. Deploy the repository as a
coherent commit; do not copy isolated files between versions.

## Private calculation and publication

- `portfolio_rebalancer_database.py` — private analysis and publication-candidate
  export.
- `portfolio_optimizer_config.py` and `portfolio_optimizer_config.json` — validated
  optimizer configuration.
- `pages/02_Public_Basket_Publisher.py` — authenticated preview and append-only
  publication fallback; never optimizes.
- `public_basket_optimizer_adapter.py` and
  `public_basket_rebalance_service.py` — conversion and event-driven publication
  services.

## Durable ledger and trust calculations

- `public_basket_postgres.py` — durable PostgreSQL event ledger and audit chain.
- `public_portfolio_publications.py` — immutable publications, forecasts,
  realizations and corrections.
- `public_portfolio_trust.py` — version-aware NAV and performance calculations.
- `public_nav_snapshots.py`, `public_outlook.py`, `public_world_benchmark.py` —
  display-ready public evidence.
- `public_portfolio_config.py` — validated environment contract.
- `public_release_checks.py` and `production_smoke_test.py` — fail-closed production
  inspection.

## Model review

- `public_review/` — mixed-market entry, instrument classification, conservative
  costs and taxes, forecasts, audit events, notifications and read-only UI.
- `public_review_policy.json` — owner-approved policy inputs; no credentials or
  per-publication ticker list.
- `update_public_review.py` — scheduled review operation and acknowledgement entry
  point.

## Public application and evidence

- `public_portfolio_performance.py` — principal public dashboard.
- `pages/01_Public_Basket_Status.py` — read-only ledger inspection.
- `pages/03_Public_Track_Record.py`, `public_card_feed.py` and
  `public_track_record.py` — publication-linked portfolio and security evidence.
- `public_allocation_card.py` — downloadable portrait allocation image.
- `update_public_nav.py` and `update_public_forecasts.py` — scheduled daily trust
  calculations.

## Automation and documentation

- `.github/workflows/tests.yml`
- `.github/workflows/public_portfolio_daily.yml`
- `.github/workflows/public_portfolio_review.yml`
- `README.md`, `CONTRIBUTING.md`, `PRODUCTION_RELEASE.md`,
  `PUBLIC_PORTFOLIO_METHODOLOGY.md` and `REVIEW_DELIVERY.md`

## Excluded material

Never deploy or commit secrets, `.streamlit/secrets.toml`, Python caches, local
databases, broker reports, private holdings, database backups, private analysis
JSON, temporary migration/correction pages, test artifacts or generated evidence
containing private data.

## Release order

Provider snapshot → clean-checkout tests → private analysis → operator review →
fingerprinted preview → publish once → NAV and outlook → production smoke test →
evidence inspection → public-navigation cleanup → logged-out verification.

See [PRODUCTION_RELEASE.md](PRODUCTION_RELEASE.md) for the complete gate.
