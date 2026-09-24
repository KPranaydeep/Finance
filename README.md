# Finance public portfolio

This repository contains a private portfolio optimizer and a public, read-only
evidence layer for the versioned model portfolio published as `PUBLIC-01`.

The optimizer remains private to the operator. The public application displays
immutable target allocations, estimated-net model performance, a 28-calendar-day
statistical outlook, publication-linked security track records, and a monitored
review date. It does not place orders or connect to an investor's broker account.

## Architecture

```text
private holdings + market history
                ↓
private optimizer and operator review
                ↓
fingerprinted publication candidate
                ↓
immutable PostgreSQL publication ledger
                ↓
NAV, outlook and model-review workflows
                ↓
read-only Streamlit pages and evidence exports
```

The public site never runs the optimizer. Publishing, model review and scheduled
performance calculations are separate operations with separate audit records.

## Public surfaces

- `public_portfolio_performance.py` — current allocation, review state,
  estimated-net history, VT comparison, 28-day outlook and private execution-plan
  helpers.
- `pages/01_Public_Basket_Status.py` — durable-ledger health and record inspection.
- `pages/03_Public_Track_Record.py` — publication-linked portfolio and security
  evidence cards, including current and exited holdings.

## Operator surfaces

- `portfolio_rebalancer_database.py` — private optimizer and analysis export.
- `pages/02_Public_Basket_Publisher.py` — authenticated publication preview and
  append-only publication action; it does not optimize.
- `pages/Filter_Universal_By_GoodTickers.py` and
  `pages/Probe_YF_Tickers.py` — development utilities that must not remain in the
  public navigation at release.

## Scheduled jobs

- `.github/workflows/public_portfolio_daily.yml` updates model NAV, creates or
  evaluates the 28-calendar-day outlook, and runs the production smoke test.
- `.github/workflows/public_portfolio_review.yml` performs mixed-market entry,
  monitoring and planning checks without submitting trades. Its manual inputs can
  append an acknowledgement that the owner completed a review.
- `.github/workflows/tests.yml` runs the deterministic test suite on pushes and
  pull requests.

## Development and release

Install dependencies and run the suite from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Development backfill is explicitly labelled simulation. Before public release,
`PUBLIC_MODEL_BACKFILL_TRADING_DAYS` must be set to `0` and the development ledger
must be reset according to [PRODUCTION_RELEASE.md](PRODUCTION_RELEASE.md).

The canonical methodology is in
[PUBLIC_PORTFOLIO_METHODOLOGY.md](PUBLIC_PORTFOLIO_METHODOLOGY.md). The release
checklist is in [PRODUCTION_RELEASE.md](PRODUCTION_RELEASE.md), and review-monitor
operations are in [REVIEW_DELIVERY.md](REVIEW_DELIVERY.md).

## Contributor FYI

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing calculations, schemas,
workflows or public wording. In particular:

- never mutate an immutable publication or audit event;
- never add optimizer or trade execution controls to a public page;
- fail closed when classifications, prices, FX, costs or calendars are uncertain;
- never commit secrets, private holdings, broker reports or analysis exports;
- update tests and documentation in the same pull request as a behavior change.

Historical performance and statistical scenarios are model evidence, not
investment advice or guaranteed future results.
