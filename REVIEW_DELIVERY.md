# Public portfolio model review — v46

Built against Finance main `9e62848545192eadfa5de244f7e697cb96aebb0a`.

## What this delivers

- Separate frozen publication-based model investments, latest active publication
  selected by default; prior created investments continue to be checked. A new
  publication does not reset an earlier investment's entry date or cost basis.
- First eligible NSE opening price after actual `published_at`, only after the
  session has completed. No pre-publication quote is used as an entry.
- Whole-share entry allocation, entry fees, residual cash, estimated distributions,
  security XIRRs and basket cash-flow XIRR, absolute returns and net liquidation value.
- Conservative standard Groww/Zerodha NSE-delivery charge envelope, explicit
  instrument tax categories, slippage and tax assumptions. Not a contract note,
  personal tax calculation, SEBI certification or universal broker tariff.
- Separate profit-taking, risk and rebalance-benefit review gates. Target net
  annualized XIRR is 100%; it is not a 100% absolute-return goal.
- Joint stationary-block path simulation, first-crossing probabilities, retained
  non-crossing scenarios and a proposed next review date. The date is withheld until
  the conditional walk-forward checks pass. No promise of market-timing accuracy.
- Cost-first whole-share comparisons: no trade, full exit, recover initial capital,
  or withdraw estimated profit. Sell only: no optimizer runs or broker orders.
- Independent daily workflow, opt-in email/Telegram notifications, deduplication,
  stale-heartbeat warnings, dated acknowledgements and an append-only hash chain.
- Read-only responsive Streamlit panel with static security tables, expanded
  explanations and downloadable model evidence. Simulations never run on page load.

## Upload using GitHub — no local Python required

1. Extract this ZIP. Upload its **contents**, preserving folders, into Finance's
   repository root. Do not upload the ZIP or a containing `wealth-manager-review`
   folder. GitHub's upload UI may hide `.github`; verify the workflow file exists at
   `.github/workflows/public_portfolio_review.yml` after committing.
2. The only replaced existing files are `public_portfolio_performance.py` and
   `requirements.txt`. If main changed after the SHA above, merge the two integration
   changes instead: import `render_review_panel` and call it after the portfolio
   overview, plus the dependency updates. `CHANGES.patch` is supplied for comparison.
3. Commit to main. Streamlit Cloud installs the requirements and deploys the page.
   Keep your existing `[public_basket].database_url` secret unchanged.
4. Initially the page says monitoring is not ready. This is intentional. The
   workflow defaults to disabled and creates no records until you enable it.

## Owner decisions required before enabling

Edit `public_review_policy.json` in GitHub. It contains **proposed defaults**, not
personal recommendations. Do not simply switch approval on without checking them.

| Setting | Meaning |
|---|---|
| `policy_approved` | Set true only after approving every policy assumption below |
| `tax_profile` | Only resident-individual normal funded NSE delivery is implemented |
| `slab_rate`, `surcharge_rate` | Explicit illustrative assumptions; currently 30% and 0%. Not every investor's tax rate |
| `slippage_bps` | Estimated per-side slippage, currently 10 basis points; not a guarantee for illiquid stocks |
| `capital_inr` | Null derives a full-target reference corpus from **entry** prices plus fee allowance; a number fixes it explicitly |
| `target_xirr` | 1.0 means 100% annualized, after modeled liquidation deductions |
| `drawdown_limit` | Proposed 15% net-model peak drawdown review trigger |
| `concentration_limit` | Proposed 50% maximum single security weight review trigger |
| `drift_limit` | Proposed five-percentage-point allocation drift inspection boundary |
| `crossing_probability` | Proposed 20% cumulative probability limit used to schedule a review |
| `max_review_sessions` | A 20-session uncertainty ceiling; does not force a trade or a calendar-based date |
| Automatic instrument classification | Every target ticker is resolved from NSE/Yahoo metadata at runtime; unknown categories fail closed |
| `tariff_verified_on` | Change only after actually rechecking the referenced current tariffs and implemented numbers |
| `calendar_verified_through` | Verify exchange calendar coverage, including special sessions, before extending |

Instrument types:

- `equity`: domestic listed equity share eligible for the modeled equity tax regime.
- `equity_etf`: qualifying domestic equity-oriented listed ETF.
- `listed_non_equity_etf`: listed gold/international/other non-equity ETF **only if
  confirmed not a specified debt fund** under the applicable tax rules.
- `specified_debt_etf`: verified specified debt-fund category, slab-rate treatment.

Examples of classification syntax (not a verified current constituent list):
`"LAURUSLABS.NS": "equity"`; confirm each ETF's category with its issuer before
assigning it. Never classify all `.NS` instruments as equity. The mapping is empty
on purpose: silent tax classification would make the post-tax trigger misleading.

No annual capital-gains exemption, loss offset, marginal relief or capital-gain
fee deduction is credited in this conservative model. The model does not multiply
an annual exemption across stocks. Applicable cess and configured surcharge are
included. Only entries from FY2026-27 onward are supported. Earlier tax regimes,
non-residents, special accounts/products, foreign-exchange trading, short selling,
MTF, mandates, call-and-trade, penalties and account-maintenance charges are outside
scope. Do not claim this covers all investor circumstances or all charges everywhere.

## Enable GitHub Actions and notifications

In GitHub → Settings → Environments → **PRODUCTION**:

Variables:

- `PUBLIC_REVIEW_ENABLED` = `true`
- `PUBLIC_REVIEW_ALERT_CHANNEL` = `telegram` or `email` (`none` leaves alerts off)

Secrets:

- Existing `PUBLIC_BASKET_DATABASE_URL`: reuse your PostgreSQL URL here.
- For Telegram: `PUBLIC_REVIEW_TELEGRAM_TOKEN` and `PUBLIC_REVIEW_TELEGRAM_CHAT_ID`.
  Create your bot with Telegram BotFather, start a conversation with it, and supply
  the destination chat ID. Never commit the token or put it in the policy JSON.
- For email: `PUBLIC_REVIEW_SMTP_HOST`, `PUBLIC_REVIEW_SMTP_USER`,
  `PUBLIC_REVIEW_SMTP_PASSWORD`, `PUBLIC_REVIEW_EMAIL_FROM`, `PUBLIC_REVIEW_EMAIL_TO`.
  Uses SMTP over verified TLS on port 465 (an app password may be required).

These notification secrets belong in GitHub Actions, **not in the public page**.
Use one channel. An actual successfully delivered first message is your acceptance
test; code cannot confirm a channel without the credentials and destination.

Then Actions → **Public portfolio model review** → Run workflow on main.

The workflow runs unit/UI tests before the monitor and runs daily at 18:15 IST,
including weekends, independently of the existing NAV/forecast workflow. GitHub
scheduled runs can be delayed or disabled; this is not a guaranteed monitoring SLA.
Enable GitHub's failed-workflow email notifications as a separate backup.

## Review-date policy

The engine samples joint consecutive-return blocks from the available common
history (three years by default). It estimates when the profit, drawdown,
concentration or drift boundary may first be crossed. It proposes a review one
session before cumulative crossing probability reaches the configured limit,
bounded by `max_review_sessions`.

At least 126 common return rows are needed to simulate. The default walk-forward
gate needs at least **652 return rows** (252 training + 20 non-overlapping folds of
20 sessions), five crossing and five non-crossing cases, Brier score <=0.20, and no
more first-review late cases than weekly review. These are explicit preliminary
engineering thresholds, not published universal thresholds or proof of accuracy.

The conditional test uses earlier returns to forecast later historical return
blocks at today's model state and holding age. It tests timing conditional on the
chosen basket, **not whether choosing this basket historically was possible or
profitable**. It is not a full strategy backtest with actual rebalance executions.
It does not prove lifetime tax savings or alpha. Short shared history, recent IPOs
or one-sided test outcomes may leave the date unavailable. Do not lower the gate
just to display a date. Today's deterministic threshold checks still run.

Once an actionable date has been stored, it cannot silently move later. To confirm
you reviewed the model, rerun the workflow with `acknowledge_baseline` set to the
baseline ID from the evidence download. Acknowledgement records **no trade** and
resets only the scheduling promise; it does not reset holdings or XIRR.

MMI is timestamped context only. No untested claim that fear/greed predicts returns
or reduces slippage has been added.

## Rebalance benefit gate

The deterministic gate is implemented and tested. It requires a validated,
comparable **net annual** benefit estimate of at least six percentage points for
return-seeking rebalances. The current repository does not provide such an approved
estimate with the necessary provenance. Therefore this release deliberately shows
`BENEFIT_NOT_ESTABLISHED` when relevant instead of manufacturing an estimate from
the 28-day median. Risk/profit reviews remain independent. No automatic buy/sell
rebalance instruction is generated by this missing-evidence gate.

## Model and execution limitations

- Historical Yahoo OHLC is split-normalized. Model entry reconstruction is explicitly
  retrospective and uses the provider's current normalized historical price basis.
  It is not evidence of fills available in the past. Subsequent splits or a revised
  historical entry quote block assessment for operator review, rather than quietly
  changing frozen units. Mergers/delistings require manual resolution.
- Distributions are estimated as retained model cash on the ex-date, after the
  configured dividend-tax assumption; actual payment dates are not supplied by this
  feed. These are model XIRRs, not broker XIRRs. Core XIRR supports dated conventional
  cash flows but this release does not ingest personal deposits/trades/reports.
- Full withdrawal estimates are exact under the configured cent-rounded model.
  Partial exit optimization enumerates all quantities up to 500 shares per security;
  larger holdings use a bounded quantity grid. It proves minimum estimated fee-plus-
  tax within that grid only. If the solver times out, it explicitly withholds an
  optimality claim. It does not optimize residual diversification or future taxes.
- Capital recovery is not the same as safeguarding wealth: remaining shares can
  lose value, and recovered cash has its own risks. Short holding-period annualized
  returns are unstable. No guaranteed 100% return, stop-loss fill or optimal date.
- No market order, broker connection, optimizer run, ledger reset, schema migration
  or external alert has been executed by delivering these files.

## Storage and verification

The monitor creates only `public_review_events`, an append-only event table and
supporting trigger/index/function. Publication, order, execution and NAV tables are
not modified. Each basket's events have a serialized hash chain and idempotency
keys. The public panel reads this table with a read-only transaction. It never
creates schema or writes ledger records.

The chain detects altered entries and broken internal links; it is not an external
notarization and cannot prevent a database administrator from replacing an entire
chain. No credentials or personal reports are stored in evidence. Configured model
assumptions are public; keep personal information and credentials out of them.

Tests cover math, rounding, flow semantics, tax boundaries, audit tampering,
idempotency, clock/timezone/calendar boundaries, validation gates and Streamlit
rendering. A live PostgreSQL integration test and successful real email/Telegram
delivery still require your deployed environment. Do not equate local tests with
verified production operation.

## Sources and rationale

- Brokerage snapshot: https://groww.in/pricing and https://zerodha.com/charges/
- Fund tax classification: https://www.amfiindia.com/investor/knowledge-center-info?zoneName=TaxRegimeForMutualFunds
- Capital gains: https://www.incometaxindia.gov.in/w/capital-gain
- NSE sessions: https://www.nseindia.com/resources/exchange-communication-holidays
- Politis and Romano, stationary bootstrap: https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf
- Cost-aware threshold rationale (not calibration for this basket): https://corporate.vanguard.com/content/dam/corp/research/pdf/rational_rebalancing_analytical_approach_to_multiasset_portfolio_rebalancing.pdf

## Disable or roll back

Set `PUBLIC_REVIEW_ENABLED=false` to stop monitor writes and alerts. After 30 hours
the UI marks the last heartbeat stale and hides the previous actionable values.
Revert the two page integration lines if you want the panel removed. No deletion
of investment history or production ledger reset is needed.

