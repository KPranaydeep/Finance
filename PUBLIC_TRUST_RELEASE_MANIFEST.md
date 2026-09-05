# Public trust release manifest

Upload these paths to the Finance repository with the same names:

- `portfolio_rebalancer_database.py` — private calculation plus explicit approval publisher.
- `pages/02_Public_Basket_Publisher.py` — operator fallback publisher; never optimizes.
- `public_portfolio_publications.py` — repeatable schema, immutable publications, forecasts, realizations, and audit.
- `public_portfolio_trust.py` — performance, XIRR, bootstrap, calibration, and model-NAV calculations.
- `public_portfolio_config.py` — validated environment/configuration contract.
- `public_release_checks.py` — automated public-data inspection.
- `public_portfolio_performance.py` — public trust dashboard and evidence export.
- `update_public_nav.py` — scheduled version-aware model NAV update.
- `update_public_forecasts.py` — scheduled forecast publication and realization.
- `production_smoke_test.py` — read-only production gate.
- `tests/test_public_portfolio_trust.py`, `pytest.ini` — deterministic trust-layer suite.
- `.github/workflows/public_portfolio_daily.yml` — explicitly configured production refresh.
- `PRODUCTION_RELEASE.md`, `PUBLIC_PORTFOLIO_METHODOLOGY.md` — operator and methodology documentation.

Do not upload `.test-venv`, caches, local databases, `.streamlit/secrets.toml`, analysis JSON containing private holdings, or CSV backups to the public deployment.

Release order: provider backup → upload/commit → focused tests → generate and review private result → publish once → generate NAV and forecast → run production smoke test → inspect evidence → remove the six temporary pages listed in `PRODUCTION_RELEASE.md` → verify the logged-out public site.
