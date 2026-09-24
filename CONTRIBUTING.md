# Contributor FYI

Thank you for improving the Finance public-portfolio project. This repository
publishes financial-model evidence, so correctness, provenance and restrained public
claims matter more than feature volume.

## Trust invariants

Every contribution must preserve these rules:

1. **Publications are immutable.** Never update or delete a published allocation to
   make history look cleaner. Corrections are append-only and reasoned.
2. **The public site is read-only.** It may calculate private execution helpers in
   memory, but it must not optimize, publish, acknowledge reviews, place orders or
   write ledger records during page rendering.
3. **Unknown means unavailable.** Missing prices, FX, calendars, classifications,
   costs or validation evidence must produce a clear unavailable state—not a guessed
   value or permissive fallback.
4. **Model evidence is not broker evidence.** Do not describe modeled prices,
   charges, NAV or returns as actual fills or investor-account performance.
5. **Horizons remain distinct.** The allocation-review date and 28-calendar-day
   statistical outlook answer different questions and must not be blended.
6. **No automatic trade advice.** Review triggers request reassessment. They are not
   mandatory rebalance dates or buy/sell instructions.

## Repository map

- Private optimizer: `portfolio_rebalancer_database.py` and `portfolio_optimizer_*`
- Publication pipeline: `public_basket_*`, `public_portfolio_publications.py`
- Trust calculations: `public_portfolio_trust.py`, `public_nav_snapshots.py`,
  `public_outlook.py`
- Review monitor: `public_review/`, `update_public_review.py`
- Public UI: `public_portfolio_performance.py`, `pages/01_*`, `pages/03_*`
- Operator UI: `pages/02_Public_Basket_Publisher.py` and development utilities
- Tests: `tests/`
- Production operations: `.github/workflows/`, `PRODUCTION_RELEASE.md`

## Before changing behavior

- Identify the authoritative source for the value or rule. Configuration belongs in
  validated configuration, not duplicated literals across UI and workflows.
- Check whether the change affects a canonical payload, fingerprint, audit hash,
  database constraint or immutable event. Treat these as migration-sensitive.
- Preserve basket scoping in every query and audit operation.
- Keep timestamps timezone-aware. Store UTC; convert to `Asia/Kolkata` only for
  display or an explicitly local policy rule.
- Keep currencies explicit. Native prices, INR-normalized prices and FX evidence
  must not be silently interchanged.

## Tests required

Run the complete suite from a clean checkout:

```powershell
python -m pytest -q
```

Add focused tests for every behavior change. At minimum, cover the success path and
the fail-closed path. Financial calculations should also test boundaries, rounding,
zero values and timezone/calendar edges.

For Streamlit changes, verify the relevant page at desktop and mobile widths. For
downloadable images, assert exact pixel dimensions and inspect a representative
maximum-density portfolio.

Never point tests at the production database. Use fixtures, an isolated schema or an
in-memory database appropriate to the component.

## Pull requests

- Start from current `main` on a focused branch.
- Keep unrelated generated files, caches and local artifacts out of the commit.
- Explain the user-visible change, trust impact and verification performed.
- Update methodology, release instructions and public wording in the same pull
  request when behavior changes.
- Do not merge while required checks are failing.

## Secrets and private data

Never commit or paste into issues, logs, screenshots or fixtures:

- PostgreSQL URLs or credentials;
- Streamlit/GitHub tokens, SMTP credentials or Telegram secrets;
- PAN, demat/account numbers, email, phone, address or other identifiers;
- broker statements, private holdings or optimizer analysis exports; or
- provider backups and production evidence bundles.

Use synthetic values in tests. Keep production secrets in protected environment or
Streamlit secret stores, according to their consumer.

## Database and destructive operations

Schema initialization must remain repeatable. Additive migrations should be safe to
retry. Never hide a destructive migration inside application startup or a public
page. A reset or deletion requires a separate, explicitly targeted operator
procedure and a verified restorable backup.

## Public language

Use precise labels such as “estimated net model return,” “planning review,” and
“statistical scenario.” Avoid “guaranteed,” “optimal date,” “actual execution,” or
other wording not supported by the stored evidence.

When in doubt, preserve the evidence, expose the uncertainty and fail closed.
