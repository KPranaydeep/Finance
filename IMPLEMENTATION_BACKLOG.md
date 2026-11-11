# Implementation status

## Opening cap-search action and scan age (implemented)

`Run / resume cap search` appears directly below `Run optimization` in the first
main-page action block. The sidebar retains search settings and clearing results.
The action uses the existing search/reuse settings; no search or price download
runs while presenting the button.

The adjacent caption reads the profile's durable checkpoint and counts completed
NSE calendar session closes since the scan timestamp. Weekends, exchange holidays
and the current unclosed session do not increment the count. This is an explicitly
NSE-based age; overseas market sessions can differ. Recent-search reuse retains
the original discovery timestamp rather than making the scan look new. Partial
checkpoints are labeled separately, with their saved-pair count; missing dates or
calendar/read failures show unavailable age. The price-history cutoff is separate
from scan age. The caption refreshes after each saved batch/pair and completion.

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

### Allocation-conflict continuation (implemented)

- A frozen position above its canonical risk cap proves that a new fully invested
  allocation is infeasible. Prepared and full-history paths check this before
  trying slow incumbent solvers. No cap, freeze or full-investment rule is relaxed.
- On an infeasible/final-recheck allocation outcome, refresh the last saved target
  using the independent current-holdings/quotes/30-session-scenario pipeline.
  Original target weights, run ID, allocation date and publications remain unchanged.
  New frozen tickers from the failed run/recorded bounds are propagated and marked
  at current weights before final execution replay.
- A successful refresh finishes as **Analysis completed with allocation conflict**,
  with explicit original-allocation and refreshed-review dates. The two buy slots
  and top sell remain visible with their recorded execution conflicts. Neither a
  new feasible allocation nor a new approved publication is claimed.
- **Continue review at saved weights** resumes an existing failed allocation
  attempt directly, without repeating broad X/Y/history discovery or an optimizer.
- Missing saved targets, unavailable inputs or a failed/insufficient-history refresh
  do not become a success. Previous analysis is retained; the genuine data/compute
  failure remains explicit. Continuation requires usable scenarios and at least
  126 complete common observations. Existing concurrent-edit/save guards remain.

### Private rebalance selection: currency-specific buy and sell categories

The current private plan supersedes the profit-threshold selection UI. Select
the highest existing `Expected Annual Return Lift (%)` among non-owned buy
candidates, then the highest remaining buy (owned or new), and the highest sell
among current holdings **within each currency, after excluding active 90-day pauses**, before funding,
partial-sell Sharpe, position/tail controls or profit thresholds. Rank the remaining
eligible pool; keep existing draft quantities with no assumed outside cash or
resizing. This existing score is a historical
risk-adjusted annual-lift estimate, distinct from the 30-session probability score.
INR and USD each have a new-holding buy, open buy and top-sell slot: six rows when
the required candidates exist. Other represented currencies follow the same rule.
The visible plan and CSV include quote currency; draft values remain INR-normalized.
Currency comes only from consistent explicit order/listing metadata, including
exact-ticker metadata retained in older saved allocations; unknown/conflicting
currencies are excluded, never guessed from ticker spelling or suffix. Missing
currency/category slots are explicit. Rank separately by annual lift;
never duplicate a buy, invent a missing slot or unavailable score. Exact ticker
breaks ties. Any positive quantity, including a tiny fractional holding, means
owned; zero/absent tickers in a complete known holdings snapshot are non-owned.
Unknown quantities cannot establish a new holding or an owned-holding sell. Listing-qualified tickers
remain distinct. Saved display uses current master holdings for ownership, so a
recent purchase changes the new-holding slot without rerunning optimization.

Active pauses remove a security from every currency/category slot and the selections CSV,
including older saved drafts. Current profile pauses are read on render, so this
does not require optimization or a plan refresh. Choose the next eligible security;
if a slot cannot be filled, show it as unavailable. Raw candidates remain retained
as evidence and can qualify again after pause expiry. Exact listings and tiny
fractional ownership remain distinct. Selection mode v4 invalidates older check
approval; changed pauses require a fresh sizing/check replay before execution.

Other controls report conflicts without hiding a selected eligible draft. Independent replay
still distinguishes selected drafts from validated executable orders. Public
scheduled publishing retains its existing strict selection/check behavior. Old
saved drafts can display the selections without a solver/download; current sizing
and checks require Update rebalance plan. All pre-control candidates and selections
persist separately, so evidence survives restart without showing a paused pick.

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
