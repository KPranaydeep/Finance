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

## Planning quote recovery

The public allocation makes one bounded retry for up to five missing planning
quotes, retaining successful quotes and requiring same-date USD/INR conversion.
The retry uses a one-month history window and displays the actual quote date;
it never substitutes an entry price or an invented FX value. A widespread
provider outage does not trigger per-security retry loops. “Refresh planning
prices” clears only the planning-quote cache and performs no optimization,
publication or ledger write. Unresolved prices and listings stay unavailable.

## Private execution modes

“Build your private execution plan” defaults to Global. INR only selects native
INR-quoted securities, including India-listed overseas ETFs; Overseas only selects
foreign-currency listings. Filtered target weights are rescaled to 100% for private
planning. Buy plans, model withdrawals, entry estimates, images, prompts and JSON
exports use the same selection. Exported prompts leave outside-scope holdings
untouched and exclude their value from trade funding. The immutable global
publication, its fingerprint and performance history are unchanged. All planning
amounts remain INR-normalized, including in Overseas only mode.

## Evidence navigation

The public track-record page uses a searchable ticker picker instead of anonymous
page numbers. Options show allocation and tracking date; exited
securities show last allocation and exit date. Named previous/next controls
and the portfolio overview remain available. Each view retains its selection by
ticker within the session, and a removed selection resets to the overview. The
picker reuses the cached bulk history used by the overview to sort by highest
entry-to-latest/exit INR return. Returns are shown in the options; unavailable
or non-finite evidence appears last, never as zero. Lifecycle dates must match
before an outcome is attached to a security. Holding periods differ. No return
calculation or immutable publication is changed.

History ranking loads in a shared background job so a slow data provider cannot
block the security picker. A loading view explicitly shows unavailable returns
and alphabetical order; “Refresh return ranking” checks the existing job without
starting a duplicate download. Jobs are keyed by publication, lifecycle inputs
and the 15-minute refresh window, with at most two active downloads and eight
retained results. Workers perform data calculation only, with no Streamlit UI or
ledger writes. A new lifecycle never inherits the previous lifecycle's return.
Public Yahoo batch calls also share a process lock: the installed client's shared
batch dictionaries must not be reset by a concurrent page download. Individual
batch threading is preserved; financial inputs and conversion rules are unchanged.

The evidence picker displays “₹100 → ₹…” using the same value calculation as the
card, caption and interactive metric. A card's computed result is retained in the
session, keyed by publication, ticker, entry date and exit date, and reused in the
picker even when the bulk history refresh is pending. A newly available card
causes one cached rerun to update the selection and ranking; identical results do
not loop. Missing outcomes stay missing, rather than becoming ₹100 or zero.

All holdings are now populated automatically in eight-security batches, with
current holdings before historical exits. Each finished batch becomes available
immediately. A three-second fragment watches only job progress and refreshes the
picker when values change; it performs no duplicate downloads and stops polling
when the job finishes. Previously completed values for the same publication and
lifecycle set remain visible during refresh. A card no longer needs to be opened
to make its outcome available in the picker. Truly unavailable market history
remains blank, including new entries without a completed trading session.

### Fractional private rebalancing quantities

The private rebalance planner preserves fractional trade quantities for an existing
non-INR fractional position, using that position's recorded decimal precision
(up to eight places). This is a planning inference, not a verified broker lot size.
Integer positions and INR listings keep whole-unit sizing; fractional INR holdings
can arise from corporate actions and do not establish fractional order support.
Buys round down after the days-to-flip split. Zero-allocation exits use the held
quantity without the previous extra one-share deduction, capped at the holding.
Partial sells for non-zero target allocations follow the portfolio Sharpe check below.
Plans, holding summaries and CSV values preserve decimals. A missing order is no
longer described as necessarily less than one share from target. Existing saved
plans are historical outputs; the corrected sizing applies when generating a new
plan, without modifying publications or placing orders.

### Sharpe-qualified partial sells

Partial sells are eligible when the executable rounded portfolio improves the
overall historical Sharpe ratio after estimated trading costs. The gate also
compares with the same proposal without partial trims when that alternative has
valid unlevered exposure, so an unnecessary trim cannot simply piggyback on other
helpful trades. Trims are assessed together, including covariance effects.
The calculation uses the configured annual risk-free rate and trading-day count,
common log-return history for every owned and traded security (at least 30
observations), and the actual proposed quantities. Paused orders are removed
before the check; missing owned-security prices/history or undefined volatility
withhold partial sells. Residual cash is assumed to earn zero. One-time costs use
the configured per-side scenario assumption, amortized over 30 sessions as a
return hurdle under the existing annualized log-return Sharpe convention.
This is a historical eligibility comparison, not a confidence claim or permission
to skip the existing funded-order and uncertainty review. Buys and zero-target
exits retain their existing rules. The before/after Sharpe, cost assumption and
decision are saved with the new plan and displayed again after restart.

### Calendar-bounded review forecasts

The configured 40-session review horizon is a maximum. When fewer future exchange
sessions are verified, the observed assessment and provisional previews can use a
shorter window with at least five sessions (or the configured observation minimum,
whichever is larger). The five-session floor follows the validation engine's
supported minimum. Validation's test horizon is capped to the same available
window; its policy hash must match the effective forecast policy before a date can
pass the validation gate. Approved policy settings are retained unchanged, with
the effective policy and calendar coverage stored alongside new evidence.
The UI distinguishes a shorter forecast from an unavailable forecast. Missing,
duplicated or out-of-order dates never fill the configured horizon; no dates beyond
the calendar's verified-through boundary are generated. Optional follow-up dates
do not become a requirement for any of these forecast paths. Forecast method v8
and captured-security method v3 make changed results append-only under new keys.

On 4 November 2026, the retained NSE verification boundary remains 31 December
2026. The official NSE holiday page did not supply a verified 2027 equity schedule
in this review, so the boundary was not extended. NYSE publishes its 2027 schedule
separately; that does not establish NSE dates for a mixed-market portfolio.
Sources: https://www.nseindia.com/resources/exchange-communication-holidays and
https://www.nyse.com/markets/hours-calendars.

### Current trigger versus next timing check

While a new basket is collecting its required post-entry observations, the review
panel shows the next scheduled timing update separately from an active review
flag. It reads the workflow cron entries and applies the existing US DST guard;
it does not assume data-ready time is when the app will publish new evidence.
For the current four-session window, data is ready on 7 November at 03:00 IST and
the next scheduled weekend update begins at 18:15 IST. Scheduling delays and job
completion still govern actual publication. An elapsed observation window does
not automatically pass validation. Previous operational windows remain in the
evidence expander while this next-check explanation is active.
Calendar notices explain how many future dates remain unverified and distinguish
them from missing observed prices. The public refresh button clears only the
basket's cached static record and rereads published evidence; it does not run the
optimizer, wake the database or publish an assessment.
