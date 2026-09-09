# Mathematical public-model withdrawal

Upload the root Python files and tests/ file. No new secret, workflow or database migration. scipy is already in repository requirements.

The reference portfolio is constructed from CURRENT published weights, CURRENT planning prices and CURRENT practical full-portfolio entry capital, using the existing lump-sum allocator. It is neither personal holdings nor a historical day-zero account. Its unallocated residual cash is available to this model calculation.

The default cash request remains the small-holding suggestion and the minimum is INR 1. No report or LLM is used in this scenario. The app displays whole-share sales, estimated costs, remaining quantities/cash and a downloadable model CSV.

The mixed-integer linear solver minimizes the sum of absolute remaining position/cash deviations from the target plus estimated selling costs, both expressed in currency units. Remaining target capital is initial capital minus requested withdrawal minus costs. Constraints enforce no purchases, no overselling and sufficient net cash. Fixed costs per traded security discourage extra orders. Costs use the existing disclosed fixed, statutory and slippage assumptions; capital-gains tax is excluded. Prices are rounded to paisa and aggregate costs rounded up. The 10-second solver limit reports unavailable rather than presenting an unfinished solution as optimal. It is an optimum for this specified mathematical objective, not a promise of best investment outcomes.

Cash-only requests produce no sells. Requests exceeding realizable capital report maximum available and shortfall.

Five solver tests passed (balanced sales, INR 1 with fees, cash-only/insufficient funds, full liquidation and invalid inputs); syntax checks passed. Live Streamlit rendering remains unverified.
