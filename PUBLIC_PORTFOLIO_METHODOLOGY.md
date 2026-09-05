# Public portfolio methodology

## Calculation and publication

The private rebalancer loads the operator's private holdings and market history, applies the configured liquidity and redundancy filters, and calculates one approved target allocation. The public application never optimizes. After review, the operator publishes that exact result with a unique run ID. Its canonical payload is fingerprinted, assigned an immutable portfolio version, and stored with its positions and a basket-scoped audit event. Repeating the same run returns the existing record; conflicting content for that run is rejected.

## Gross and estimated-net model performance

The public model index applies each immutable target-weight version from its effective date. Gross NAV chains the observed returns of those target weights. Estimated-net NAV deducts implementation drag whenever the active publication changes, using target turnover multiplied by 0.10% modeled slippage plus 0.12% modeled transaction and statutory costs. Target turnover is half the sum of absolute security-weight changes, including any implied cash sleeve. The index begins fully allocated at 100; investor-specific initial deployment costs and fixed brokerage are handled by the private execution plan rather than assumed in the public percentage index.

Public performance cards and forecasts use estimated-net NAV. Gross return and cumulative modeled drag remain visible as supporting evidence. Horizon metrics use deterministic calendar cutoffs and the first observation on or after the cutoff. Metrics requiring unavailable history are shown as `N/A`. This is model performance, not the return of a broker account, and no execution claim is made.

## 14-day statistical outlook

The outlook reproducibly resamples estimated-net daily model returns and compounds 14-day simulated paths. Its deterministic seed is derived from the portfolio publication and forecast date. It reports the median, 25th/75th and 5th/95th percentiles, probability of gain, probability of loss, and probability of loss beyond the stated downside threshold. At least 60 daily returns (61 NAV observations) are required.

These ranges describe statistical uncertainty conditional on the available historical sample. They are not targets, promises, trading instructions, or guarantees. Regime changes, illiquidity, taxes, costs, tracking differences, and data errors can make future outcomes materially different.

## Forecast accountability

Every forecast is immutable and linked to its exact portfolio publication, methodology, sample dates, and calculation version. Once 14 subsequent NAV observations exist, a separate realization record stores the corresponding start value, end value, actual return, and realization date. The original forecast is never updated. Public calibration—50% and 90% interval coverage, directional accuracy, error, and bias—appears only after at least 20 completed forecasts, without selecting outcomes.

## Time and data conventions

Public display time is Asia/Kolkata. Database timestamps remain timezone-aware. `as_of` identifies the information cutoff used for a publication; `published_at` identifies when it became public. Market-day calculations use available ordered NAV observations, not assumed calendar days.

## Limitations

The estimated-net model uses disclosed proportional cost assumptions, not broker confirmations. It may differ from an investor's realized return due to capital size, execution timing, whole-share rounding, brokerage, taxes, slippage, corporate actions, missing prices, FX conversion, and cash handling. Short histories make annualized metrics and forecasts unstable. Evidence exports allow the exact published state and calculation inputs to be inspected.
