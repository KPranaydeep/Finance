# Implementation status

## P0 — Independent sell-only risk-reduction draft

Recorded 10 November 2026, Asia/Kolkata. **Implemented for the private saved-analysis
workflow**: owned-only evidence, bounded minimum-sales calculation, independent
replay, standalone action, automatic assessment after a blocked run/refresh,
profile-scoped snapshots/archive and stale-state presentation.

Remaining scope: a policy that permits executing an incomplete risk reduction
while other mandatory limits remain breached is deliberately not enabled. The
minimum-sales objective includes position/group/CVaR constraints; nonlinear
historical drawdown/volatility are checked afterwards and can withhold the result.
First-time users without retained analysis still need the normal analysis setup.

### Problem and intended behavior

The final validator correctly withholds a rebalance draft that breaches risk
limits, but the app has no separate route to calculate corrective reductions.
Refreshing saved weights cannot fix an allocation that exceeds newer caps.
Assess a sell-only risk-reduction draft independently of return-seeking buys and
full X/Y discovery. Generate a reviewable proposal; never submit trades.

### Implementation sequence

1. **Evidence and calculation module.** Start with actual current quantities,
   verified prices/FX and a dated policy snapshot. Reuse suitable cached histories
   and joint scenarios; refresh stale evidence without rerunning universe search.
   Do not calculate sell quantities from the existing *after-draft* excess table.
   Missing evidence remains unavailable; configured caps are assumptions, not
   statistically guaranteed loss bounds.
2. **Minimum-turnover repair.** Find the smallest feasible tradable reductions
   needed to meet validated concentration/shock limits. Model funded fees,
   quantity precision, actual cash and post-fee portfolio capital together.
   Proceeds remain cash; do not force immediate reinvestment or a fully invested
   allocation. Preserve frozen positions and active 90-day pauses. Do not raise
   caps or suppress validation merely to obtain a result.
3. **Independent validation.** Replay the proposed quantities through final
   funding, holdings, position, group and joint tail-loss checks. Compare current
   versus proposed risk on identical evidence, including costs and any remaining
   breaches. If frozen holdings or other limits prevent a complete repair, retain
   the proposal and identify the exact blockers. An incomplete improvement must
   not be called fully compliant or silently released through the old validator.
   The optional profit-based partial-release policy below is separate from this
   fully compliant repair workflow.
4. **Integration and persistence.** Add a standalone “Prepare risk-reduction
   draft” action; also assess the route when normal optimization/plan refresh is
   blocked. No expensive calculation during passive rendering. Persist the dated
   proposal, evidence provenance, validation and unresolved blockers separately
   from the last successful analysis and original saved weights. Restore it after
   restart; invalidate it when holdings, pauses, prices or policy evidence change.
5. **Compact presentation.** Show security, current allocation → proposed
   allocation, applicable cap, trim quantity/value, modeled costs, risk before →
   after and remaining breaches. Distinguish a risk-reduction draft from a
   return-enhancement rebalance and from executable orders. Avoid contradictory
   “zero trades / review before execution” states.

### Timing, profit-taking and return claims

- A verified risk breach triggers assessment; an arbitrary day count, profit
  percentage or break-even price is not the risk-reduction trigger.
- Reinvestment is a separate decision requiring available cash, supported
  after-cost benefit and all applicable portfolio checks.
- Selling at a gain does not itself increase whole-account XIRR. Internal sale
  proceeds must not be recorded as external withdrawals to inflate performance.
  Actual XIRR needs dated external cash flows and account valuation including cash.
- Show modeled fee/slippage assumptions and tax coverage. Unknown tax lots/taxes
  cannot be labelled as a known after-tax benefit. Do not promise both higher
  returns and drawdown protection.

### Acceptance checks before completion

- Current holdings differ from the old draft: reductions use current quantities.
- A partial sale can still exceed its cap; report the exact residual numerically.
- Quantity rounding and fees cannot create an unnoticed new cap/funding breach.
- Paused/frozen holdings are never traded; unresolved conflicts remain visible.
- Cash is conserved, no borrowing or buys are invented, and tax uncertainty is
  explicit. Portfolio risk comparisons use the same dated scenario evidence.
- No full optimizer/X/Y search is called by the standalone action or rendering.
- Prior successful analysis, immutable publications and saved target weights are
  preserved; restart restores the independent draft with its original provenance.
- Cover compliant repair, unavailable inputs, impossible complete repair, stale
  evidence, fractional holdings, zero quantities, costs and currency handling using
  synthetic/in-memory fixtures. Verify desktop/mobile and the clean-checkout suite.

## Profit-based partial release (implemented)

### Funding-independent buy candidates

Saved and fresh rebalance views always show positive saved-allocation securities
as research candidates, even when funding removes every buy or final validation
withholds the plan. New snapshots retain pre-funding buy quantities separately;
older snapshots reconstruct the list from immutable saved targets without a
solver or market download. Scores/dates are reused, not invented or refreshed on
render. Pauses, cap conflicts and missing evidence remain labeled. Unfunded
quantities never become executable orders or imply outside income is known.
New cash must be entered for funded sizing and the final risk checks to apply.

- Private profile control, disabled by default, minimum additional net benefit
  initially ₹8,079. Apply controls, then Update rebalance plan at saved weights;
  full optimization also uses the control when a target allocation is available.
  Applying controls alone never runs a solver or submits orders.
- Test each existing whole cap-reducing sell separately. Compare identical joint
  30-session buy-and-hold histories against holding, with sale proceeds at zero
  modeled yield. Expected benefit is the historical mean incremental portfolio
  wealth after stressed trading costs, before tax—not realized capital gains.
- Among qualifying sells choose exactly one with the smallest benefit at or above
  the threshold (ticker breaks exact ties). Require the paired block-bootstrap
  lower confidence bound to be positive in each chronological third and no worse
  historical ES95, with family-wise correction across all tested sells/thirds.
  Full-history nomination bias remains; no independent forecast calibration claimed.
- Replay actual quantity, price, FX, funded fees, pauses, frozen holdings, hard
  position ceiling, correlated groups, historical/adaptive tail budgets, drawdown
  and volatility checks. Only pre-existing modeled per-security cap breaches may
  remain. No holding's absolute marked exposure can increase; no buy or borrowing
  is released. The existing partial-sell Sharpe check remains in force.
- Record Partial release, the single order, estimated benefit, threshold, history
  date, evidence and exact residual breaches. Other draft orders stay withheld.
  Nested allocation-policy checks retain their failed caps; the full allocation
  is not called compliant. Public scheduled publication rejects partial releases.
- Controls persist by profile. Changed controls/holdings/pauses/policy invalidate
  the displayed saved release pending a fresh plan check. Original target weights,
  holdings and immutable publications are not rewritten.
- If no candidate qualifies, no trade is released. In particular a sale of an
  asset with positive expected future return need not beat holding after costs,
  even when selling would realize an accounting profit. Full X/Y optimizer
  feasibility is never relaxed by this execution-only exception.
