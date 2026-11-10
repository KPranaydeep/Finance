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

In the private saved analysis, **Update rebalance plan** sizes a new execution
plan from current master holdings, current prices/FX and the saved optimal
weights. It does not run candidate screening, X/Y discovery or the optimizer.
Only owned securities and saved target securities need history refreshes. The
same action refreshes 30-session research scores and the executable rebalance
assessment. Pauses, fractional quantities and the partial-sell Sharpe check still
apply. Missing quotes or concurrent holdings/target edits retain the previous
plan. Saved weights and publication candidates are unchanged; rerun optimization
when you want to reconsider the allocation itself.

Election reserve combines components with `max()`: the portfolio buffer,
India's selected seat-exposure policy, active India historical stress and active
US whole-portfolio stress. The seat policy uses the next **20 calendar days**
(today through today+19): sum actual verified contests separately for Assembly,
Lok Sabha and Rajya Sabha, divide by their dated house totals, take the maximum
percentage, then scale it by the current optimal INR allocation. It is a liquidity
policy, not a statistical loss forecast. Term ends and geographic LS/RS allocations
are excluded; missing schedule/count coverage is explicit. Official duplicate
notices and multi-phase polling do not multiply seat counts.

US regular federal dates follow the statutory calendar beyond 2026. After voting,
the panel shows a post-election watch that expires after the 20-calendar-day
aftermath window; it does not imply certification or track all special/runoff
elections. Historical stress still uses complete 30-session price episodes and
current optimal weights. Outdated prices/FX cannot qualify merely because an old
snapshot's own expected date matches. Election refreshes do not optimize or trade.
Schedule discovery runs monthly, daily near a known counting date, or on demand;
ECI and English PIB checks share a bounded request budget. The dashboard exposes
incomplete national schedule/count coverage rather than calling missing data zero
risk. Snapshot replacement retries brief Windows locks and retains the old file
if the update cannot finish.

## Scheduled jobs

- `.github/workflows/public_portfolio_daily.yml` updates model NAV, creates or
  evaluates the 28-calendar-day outlook, and runs the production smoke test.
  For a newly published version with sufficient existing NAV history, its manual
  `refresh_scope=outlook_only` mode creates the version-linked outlook, runs the
  same acceptance checks and republishes the verified snapshot, while skipping
  price downloads and NAV rebuilding. It does not publish a new allocation.
- `.github/workflows/public_portfolio_review.yml` performs mixed-market entry,
  monitoring and planning checks without submitting trades. Its manual inputs can
  append an acknowledgement that the owner completed a review.
- `.github/workflows/tests.yml` runs the deterministic test suite on pushes and
  pull requests.
- `.github/workflows/public_snapshot_recovery.yml` manually republishes the
  existing verified record after an outage, without running NAV updates or model
  review. It reads one repeatable-read, read-only PostgreSQL transaction and
  publishes `public_record_snapshot.json` to the `public-snapshot` branch. The
  production database must be available; missing or invalid evidence stops the
  export and leaves any previously published snapshot intact.

Public readers use that verified static snapshot instead of querying PostgreSQL.
After deployment, check that the snapshot branch exists and the public page
loads. Do not enable database fallback to work around a missing snapshot: public
traffic would consume the database quota again.

The Streamlit 1.59 dependency range is paired with `starlette<1.8`, tested with
1.7.0. The Cloud combination of Streamlit 1.59.2 and Starlette 1.10.0 can send
media response bodies before their headers. Run the HTTP middleware regression
tests before widening this compatibility constraint.

## Development and release

Public and private pages share native Streamlit headings and summary cards in
`portfolio_ui.py`, with light/dark colors and typography in
`.streamlit/config.toml`. Cards stack on narrow screens so full currency values,
dates and unavailable-state notes stay readable. Diagnostics remain available in
a collapsed panel. No external fonts, React runtime or additional UI package is
required; calculations, publication evidence and pause persistence are unchanged.

In the private rebalancer, 90-day security pauses can be applied directly from
optimal weights (including saved weights after restart), even when no trade is
executable. These use the same owner-scoped registry as trade-row pauses, block
trades and future solver inclusion until expiry, and leave saved analysis weights
and holdings records intact.
Pause submission refreshes only its own controls, preserving the analysis on
screen. Saved optimal weights open without another optimizer run, and the
independent **Security pause status · 90 days** panel lists all active pauses
and expiry dates even when no saved analysis is available.

The public model-review selector defaults to the latest publication even while
its entry evidence is pending. Earlier frozen model investments remain selectable;
their entry dates, capital and performance are never reused for the new version.

Private allocation integrity checks validate solver fallbacks and final weights
against the configured position, historical drawdown and daily-volatility limits.
Sub-1% weight cleanup is skipped when it would break a constraint; a small universe
does not authorize relaxing the configured position cap. Drawdown validation
includes the initial invested value and replays daily-rebalanced simple returns.
These historical checks are not future loss limits.

After rounding, pauses and the partial-sell review, buys are rounded down to fit
recorded cash plus estimated net sale proceeds, including modeled per-side costs.
Unrecorded cash/credit is assumed zero. The final marked holdings (including paused
holdings), remaining cash, quantity precision and available risk history are checked
again. Sale proceeds must be available before buys; settlement and live broker fills
are not verified. A failed check withholds executable orders while preserving the
saved analysis and a clearly labelled proposal. **Update plan** performs this check
against current holdings without rerunning the optimizer. Older saved plans have
no pass status until refreshed. The configured limits remain explicit policy,
not statistically proven safe thresholds.

The near-duplicate filter removes strongly positive correlations, preserving
opposite-moving exposures. Extreme price moves are reconciled against corporate
actions first. A date-matched second-provider return can verify a genuine move,
whose complete loss/gain remains in risk history. Optional, cached Stooq daily
closes support bare US listing symbols; unsupported listings, unavailable access,
or conflicting evidence retain quarantine. No missing witness is treated as a
confirmation, and a second Yahoo response alone cannot certify the move.

The v6 private allocator adds a sparse SciPy/HiGHS alternative to the existing
solver, using 1,024 deterministic joint moving-block, 30-session buy-and-hold
scenarios. Shared block draws preserve cross-security dependence. Its objective
maximizes shrunk scenario mean return after trading costs, with invested capital
plus fees equal to the initial capital; no credit or additional cash is assumed.
The loss CVaR95 budget is the active current mix's CVaR under the same scenarios.
Per-side modeled costs are stressed at 2× inside the objective and tail constraint.
Three expanding-window finalist checks refit the same allocator on each training
prefix; their held-out cost calculation includes both buy and sell legs.

Explicit starting policy in `portfolio_optimizer_config.json`: a 2% portfolio
loss budget per security under at least a 50% single-security shock (a 4% weight
ceiling, tightened for larger observed daily losses); positively correlated groups
linked at 0.90 have a 25% combined ceiling. These are policy assumptions, not
optimal/safe thresholds learned from prices. Means are shrunk 50% toward their
cross-security median. An opposite-moving security is not part of a positive group.
The LP's default 10-second budget is enforced by HiGHS. It uses existing histories
and installed SciPy, with no new data-provider/library dependency.

Both the incumbent and challenger must pass the same policy and P0 checks before
comparison. The incumbent may remain selected if it passes those limits; a failed
or timed-out solve cannot silently restore an unconstrained incumbent. If neither
passes, no new allocation is selected and the prior saved analysis remains intact.
The saved panel records the selected method, current-mix tail budget, modeled net
return/cost comparison, policy assumptions and history date. This fitted comparison
is not proof of out-of-sample improvement. Current-universe finalist validation is
not a point-in-time backtest; future returns and unobserved shocks remain unknown.

Final rounded orders recheck the policy on actual post-fee holdings and cash.
**Update rebalance plan** rebuilds these checks at saved weights and current
holdings/history without an optimizer; unchanged saved target weights may be
withheld when they fail the new limits. Unknown histories cannot pass. Policy/cost
changes invalidate compatible search reuse, while old research rows remain stored.
Old external configuration files without an `allocation_policy` section retain
their previous allocator, rather than silently adopting new assumptions.

Price-integrity validation uses four shared I/O workers, a 60-second corporate-action
request window and a 20-second independent-source request window. Progress shows
completed checks, cache hits and elapsed time; this does not bound the whole
optimization. Unfinished or unavailable evidence stays quarantined (owned positions
retain the existing freeze rule). There is no cascade of individual retries.
Completed provider evidence is cached beside the private database across restarts,
scoped by ticker, history dates, evidence type and integrity version. Successes expire
after 24 hours and unavailable responses after 15 minutes. In-flight requests may
finish into that cache after a request window closes, but cannot change the current
report. Logs record aggregate timing, checked/deferred counts and quarantine totals;
saved search pairs, price histories and analysis results are preserved.

### Whole-process chronological validation (P2)

The private app has a separate **Whole-process chronological validation · P2**
block with **Run / resume chronological test**. It does not run on page load,
download prices or alter saved optimal weights, holdings, pauses or publications.
Default scope is three non-overlapping 30-session tests from up to three years of
cached INR history, a minimum 252-session initial training window, and eight X/Y
fit attempts per allocator per fold. Each click has a 120-second scheduling window;
an in-progress bounded solve/preparation can finish afterwards. Resume reuses
completed trials. The report and private checkpoints survive app/process restarts.

Inside each training prefix the actual production ranking, dated market-cap,
history/momentum, integrity, duplicate and exact-universe filters and adaptive X/Y
search run again. The incumbent (P0 checks, P1 disabled) and risk-aware allocator
are evaluated independently; today's winning pair and personal holdings do not
seed a historical fit. Both begin with a common synthetic equal-weight basket
chosen from training coverage. A failed fit explicitly holds its prior research
mix and counts as a held fold, not a successful model selection. The first forward
return belongs to prior holdings; trades occur at the next close with gross
buy/sell costs funded from capital and drifting holdings thereafter.

The compact comparison shows net return, drawdown, observed one-session ES95,
turnover, modeled costs and compute time, with per-fold evidence below it. Missing
required forward prices withhold the fold rather than erasing the losing asset.
Calendar-closed sessions alone may carry a prior native mark; missing open quotes
and future FX cannot be filled permissively. P2 is conditional on the current
universe/pause settings and vendor-revised cached prices, not reconstructed
historical membership/delistings or a broker backtest. Small samples do not prove
future superiority and never automatically promote a strategy. The existing
finalist stability gate remains separate and keeps its limited-scope label.

Private research lives beside the holdings database under `chronological_research/`.
Input data, active configuration and source fingerprints isolate experiments;
changing them retains prior checkpoint files and starts a new experiment. Version
v7 also removes FX backward-fill from normal analysis; older search rows remain
research, while incompatible results cannot silently resume as current fits.

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
