# Per-security entry and review controls

Edit `public_review_policy.json` only through a reviewed commit. The policy is
fingerprinted into assessments, so a change creates new evidence rather than
silently rewriting an earlier result.

## Entry rule

- `entry_wait_after_open_minutes`: `60`
  - If a security's exchange is open at publication, its requested entry time
    is exactly the immutable publication timestamp.
  - Otherwise its requested entry time is its own next exchange open plus this
    delay.
- `entry_quote_interval`: `1m`
  - Fixed to one-minute Yahoo bars by validation.
- `entry_quote_tolerance_minutes`: `5`
  - Maximum distance between requested entry time and the price bar. Lower is
    stricter and may delay/fail capture; higher is less faithful to the rule.
- `assessment_wait_after_close_minutes`: `30`
  - Data-availability buffer for completed-session assessments. It does not
    delay another security's entry.

Each captured security entry is an immutable `SECURITY_ENTRY` audit event with
requested timestamp, quote timestamp, exchange, native price, exact-time FX
proxy where applicable, INR price, source and rule basis. A basket baseline is
created only after every target security has an entry.

## Review-date sensitivity, highest impact first

1. `target_xirr`: `1.0` means 100% annualized net XIRR after modeled costs and
   liquidation tax. Lowering it generally makes security review dates earlier.
2. `crossing_probability`: `0.20` requires at least 20% of simulated paths to
   have crossed the trigger by that date. Lower values generally produce
   earlier, less selective research dates.
3. `max_review_sessions`: limits how far the engine searches into the future.
   It is a horizon, not a market-timing assumption.
4. `block_length`: controls persistence retained by stationary-block sampling.
5. `history_years`: controls the historical regime window.
6. `simulation_paths`: controls Monte Carlo stability, not expected return.

`drawdown_limit`, `concentration_limit`, and `drift_limit` can request an earlier
risk review independently of the profit target. `min_annual_improvement` gates
return-seeking rebalancing and does not change the optimal target portfolio.

Do not tune these controls merely to obtain a preferred date. Change one policy
at a time, preserve the prior evidence, and rerun the tests and model-review
workflow.
