# Public portfolio model review

The model-review service monitors each immutable public portfolio publication. It
produces evidence and requests reassessment; it never runs the optimizer, places an
order or declares that a trade is mandatory.

## Current behavior

- Every publication creates a separate frozen model investment. A later
  publication does not reset an earlier baseline, entry date or cost basis.
- Each security enters according to its own exchange session. If the security was
  trading when published, the eligible publication-session evidence is used;
  otherwise entry waits for that market's next open plus the configured delay.
- Domestic and direct-US listings are converted to INR using dated FX evidence.
- Whole-share quantities, residual cash, entry and liquidation costs, estimated
  distributions, taxes, security XIRRs and basket cash-flow XIRR are included.
- A planning review date may be available immediately from pre-publication history.
  Observed monitoring supersedes planning evidence when post-entry sessions become
  available.
- Security-level target-crossing estimates remain separate from the allocation-
  weighted planning date shown to the owner.

## Policy source

`public_review_policy.json` is the single source of current owner-approved policy
values. Do not duplicate its numeric thresholds in this guide. The loader validates
ranges, approval, tariff freshness and calendar coverage before assessment.

Important distinctions:

- `target_xirr` is an annualized net XIRR threshold, not an absolute-return target.
- `minimum_net_return` is an additional break-even margin after modeled round-trip
  friction.
- `minimum_forecast_review_sessions` counts completed post-entry sessions and does
  not include the entry session.
- `crossing_probability` schedules a review; it does not represent confidence that
  a security should be sold.
- `min_annual_improvement` applies only to a comparable, validated net annual
  rebalance-benefit estimate.

## Instruments, costs and taxes

Domestic classifications are resolved from current NSE equity and ETF metadata.
Direct-US Yahoo symbols are classified as foreign US listings. Frozen baseline
classifications are reused and conflicting classifications fail closed. The owner
policy file intentionally contains no manually maintained ticker list.

The cost model uses a conservative envelope for normal funded delivery:

- domestic Groww/Zerodha brokerage, DP, STT, stamp duty, exchange, SEBI, IPFT, GST
  and configured slippage;
- direct-US Tickertape Pro brokerage and regulatory charges, GST, HDFC funding
  assumptions and dated USD/INR; and
- the configured resident-individual tax profile without credit for exemptions,
  loss offsets or marginal relief.

It is a planning model, not a contract note or personal tax return. Unsupported
instruments, stale tariffs, unavailable FX, ambiguous classifications and invalid
calendar coverage stop assessment instead of being guessed.

## Review gates

The deterministic assessment keeps three decisions separate:

1. **Profit review** — net proceeds clear both round-trip friction plus the minimum
   net-return requirement and the configured annualized XIRR threshold.
2. **Risk review** — drawdown, concentration or allocation-drift boundaries require
   inspection.
3. **Rebalance benefit** — a return-seeking rebalance requires a validated,
   comparable net annual improvement estimate. Without approved provenance the
   result remains `BENEFIT_NOT_ESTABLISHED`.

A trigger is a request to review the portfolio. It is not a buy, sell or hold order.

## Forecasting and review dates

The engine uses joint stationary-block simulations over the configured common
history. It retains paths that do not cross within the horizon and validates timing
on earlier historical folds. Dates are withheld when data or validation is
insufficient. “Not reached in horizon” is a valid result and is not converted into
an invented date.

The allocation-weighted planning date combines only usable security estimates using
target weight multiplied by probability by date. The displayed date is adjusted to
an eligible market-review day. The 28-calendar-day public outlook is a separate
statistical horizon and is never presented as the expected return by the review
date.

## Workflow schedule

`.github/workflows/public_portfolio_review.yml` runs:

- NSE and US pre-market planning checks;
- exchange-specific open-plus-entry-delay checks;
- an NSE close-plus-data-buffer check;
- a morning check that can process the completed US session without scheduled work
  during the owner's 22:00–04:00 IST sleep window; and
- a daily heartbeat for weekends and holidays.

GitHub cron is best effort and may start late. The workflow is idempotent and a
delayed run does not invent missing prices.

## Owner acknowledgement

After genuinely reviewing the latest active assessment, manually run **Public
portfolio model review** and select **I completed the latest active model review**.
The workflow appends an acknowledgement bound to the exact assessment, policy and
trigger set. It records no trade, does not reset the model investment, and suppresses
only the unchanged reviewed trigger. A new or recurring trigger opens a new cycle.

## Notifications

Set `PUBLIC_REVIEW_ALERT_CHANNEL` to `telegram`, `email` or `none` in the protected
`PRODUCTION` environment. Channel credentials belong only in GitHub secrets. A real
delivered message is required to validate notifications; unit tests cannot prove
external delivery.

## Storage and limitations

The monitor writes only append-only `public_review_events` and supporting audit
metadata. It does not modify publications, orders, executions or NAV records. The
public panel reads with a read-only transaction.

Historical prices can be revised for splits or corporate actions. Market-data
outages, mergers, delistings and unsupported tax regimes require explicit operator
review. Short-period annualized returns are unstable. No review date, XIRR estimate,
forecast or withdrawal plan is guaranteed.

Run the review tests with:

```powershell
python -m unittest discover -s tests -p 'test_review_*.py' -v
```
