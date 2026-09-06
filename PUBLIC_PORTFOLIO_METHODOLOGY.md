# Public portfolio methodology

## Calculation and publication

The private rebalancer loads the operator's private holdings and market history, applies the configured liquidity and redundancy filters, and calculates one approved target allocation. The public application never optimizes. After review, the operator publishes that exact result with a unique run ID. Its canonical payload is fingerprinted, assigned an immutable portfolio version, and stored with its positions and a basket-scoped audit event. Repeating the same run returns the existing record; conflicting content for that run is rejected.

## Gross and estimated-net model performance

The public model index applies each immutable target-weight version from its effective date. Gross NAV chains the observed returns of those target weights. Estimated-net NAV deducts implementation drag whenever the active publication changes, using target turnover multiplied by 0.10% modeled slippage plus 0.12% modeled transaction and statutory costs. Target turnover is half the sum of absolute security-weight changes, including any implied cash sleeve. The index begins fully allocated at 100; investor-specific initial deployment costs and fixed brokerage are handled by the private execution plan rather than assumed in the public percentage index.

Public performance cards and forecasts use estimated-net NAV. Gross return and cumulative modeled drag remain visible as supporting evidence. Horizon metrics use deterministic calendar cutoffs and the first observation on or after the cutoff. Metrics requiring unavailable history are shown as `N/A`. This is model performance, not the return of a broker account, and no execution claim is made.

During private development, `PUBLIC_MODEL_BACKFILL_TRADING_DAYS` may be set above zero to extend the earliest active publication over a bounded number of prior trading days. Later immutable publications retain their actual dates and therefore create modeled rebalance transitions, including estimated turnover and implementation drag. The requested duration is a maximum: analysis begins no earlier than the latest first valid price among all securities used by active portfolio versions and ends no later than their earliest last valid price. This prevents use of a period before every included security existed. Interior missing observations are carried forward for mark-to-market. Backfilled rows and forecasts are explicitly labelled as development simulation. The variable defaults to zero and must be zero—and the development ledger reset—before public release. With zero backfill, NAV begins at 100 on the first available close after publication and the first return is measured from the following trading close.

## 28-calendar-day statistical outlook

The outlook uses all complete 28-calendar-day historical blocks in the available estimated-net basket NAV history, aligned to the issuance weekday. Whole blocks preserve the ordering of returns and volatility within four weeks. Weekends and other dates without a NAV observation carry the previous close. The displayed median and 50%/90% ranges are empirical scenario quantiles; gain and loss probabilities are historical scenario frequencies. No doubling of an old 14-day estimate or annual-return scaling is used. At least 126 NAV observations and 20 complete scenarios are required as operational data gates. The blocks overlap and do not represent independent observations. This is a historical scenario estimate, not a fitted or validated predictive model.

These ranges describe statistical uncertainty conditional on the available historical sample. They are not targets, promises, trading instructions, or guarantees. Regime changes, illiquidity, taxes, costs, tracking differences, and data errors can make future outcomes materially different.

## Forecast accountability

Every new forecast records its publication, sample dates, method, horizon unit, target calendar date and reference NAV. Once NAV data reaches or passes expiry, evaluation uses the last available close on or before expiry and the original frozen reference NAV. An expiry on a market holiday can therefore be evaluated after the next NAV arrives. Legacy forecasts keep their original trading-observation evaluation rules and are excluded from the new outlook display. Calibration is restricted to the current publication's 28-day method after 20 completed forecasts. Daily forecast horizons overlap, so completed forecasts are not independent trials. Future publications are unknown at issuance; realizations follow the actual basket changes.

## Time and data conventions

Public display time is Asia/Kolkata. Database timestamps remain timezone-aware. `as_of` identifies the information cutoff used for a publication; `published_at` identifies when it became public. Market-day calculations use available ordered NAV observations, not assumed calendar days.

## Limitations

The estimated-net model uses disclosed proportional cost assumptions, not broker confirmations. It may differ from an investor's realized return due to capital size, execution timing, whole-share rounding, brokerage, taxes, slippage, corporate actions, missing prices, FX conversion, and cash handling. Short histories make annualized metrics and forecasts unstable. Evidence exports allow the exact published state and calculation inputs to be inspected.
