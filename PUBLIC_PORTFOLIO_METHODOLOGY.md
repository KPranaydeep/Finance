# Public portfolio methodology

## Calculation and publication

The private rebalancer loads the operator's private holdings and market history, applies the configured liquidity and redundancy filters, and calculates one approved target allocation. The public application never optimizes. After review, the operator publishes that exact result with a unique run ID. Its canonical payload is fingerprinted, assigned an immutable portfolio version, and stored with its positions and a basket-scoped audit event. Repeating the same run returns the existing record; conflicting content for that run is rejected.

## Historical performance

The public model index applies each immutable target-weight version from its effective date. Daily return is the change in model NAV. Horizon metrics use deterministic calendar cutoffs and the first observation on or after the cutoff. Total and cumulative return, annualized return and volatility, downside deviation, positive-day percentage, best/worst day, maximum drawdown, and current drawdown are calculated from observed NAV only. Metrics requiring unavailable history are shown as `N/A`.

## XIRR

XIRR is the annualized money-weighted return that sets the net present value of dated external contributions, withdrawals, and the terminal portfolio value to zero. Actual dates are used. A valid series must contain both negative and positive cash flows. The system does not infer cash flows from percentage returns or invent an initial contribution; without valid cash-flow history, XIRR is `N/A`. The displayed period runs from the first included external cash flow through the terminal valuation date.

## 14-day statistical outlook

The outlook reproducibly resamples observed daily portfolio returns and compounds 14-day simulated paths. Its deterministic seed is derived from the portfolio publication and forecast date. It reports the median, 25th/75th and 5th/95th percentiles, probability of gain, probability of loss, and probability of loss beyond the stated downside threshold. At least 60 daily NAV observations are required.

These ranges describe statistical uncertainty conditional on the available historical sample. They are not targets, promises, trading instructions, or guarantees. Regime changes, illiquidity, taxes, costs, tracking differences, and data errors can make future outcomes materially different.

## Forecast accountability

Every forecast is immutable and linked to its exact portfolio publication, methodology, sample dates, and calculation version. Once 14 subsequent NAV observations exist, a separate realization record stores the corresponding start value, end value, actual return, and realization date. The original forecast is never updated. Public calibration—50% and 90% interval coverage, directional accuracy, error, and bias—appears only after at least 20 completed forecasts, without selecting outcomes.

## Time and data conventions

Public display time is Asia/Kolkata. Database timestamps remain timezone-aware. `as_of` identifies the information cutoff used for a publication; `published_at` identifies when it became public. Market-day calculations use available ordered NAV observations, not assumed calendar days.

## Limitations

The model index may differ from an investor's realized return due to execution timing, fractional shares, fees, taxes, slippage, corporate actions, missing prices, FX conversion, and cash handling. Short histories make annualized metrics and forecasts unstable. Evidence exports allow the exact published state and calculation inputs to be inspected.
