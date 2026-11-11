# Reading the saved allocation image

The image transcribes the saved statistics; downloading it does not run a solver,
fetch prices or recompute evidence. **Current → optimal** compares historical
analysis of the two mixes, not broker-account returns. Displayed saved allocations
can have been rounded after fitting; their replay need not reproduce the saved
statistics exactly. If the allocation has
an unpaused-capital scope, both the weights and statistics refer to that scope.
An optimized in-sample result can overstate an improvement that persists outside
the fitted period. These statistics do not deduct trading costs or taxes.

## Definitions and the app's conventions

For daily asset log returns `r[t,i]`, the current calculator forms
`p[t] = sum(w[i] * r[t,i])`. This weighted-log series is the app's modeling
convention; it is not the exact log return `log(sum(w[i] * exp(r[t,i])))` of a
constant-weight basket of simple-return assets. The export preserves the saved
values and names that convention rather than silently replacing calculations.

- **Annual log return:** `mean(p) × N`, for the recorded sessions per year `N`.
  It is not CAGR, XIRR or an expected future simple return. Exponentiating this
  number would describe compounding the modeled log series, not broker performance.
- **Annual volatility:** sample standard deviation of `p` times `sqrt(N)`.
  This familiar scaling assumes the relevant serial-dependence conditions; it
  does not establish a loss bound or a confidence interval for future returns.
- **Sharpe-like (app approximation):** `(mean(p) × N − saved annual risk-free
  input) / (std(p) × sqrt(N))`. Conventional Sharpe uses the mean and standard
  deviation of matching periodic differential/excess returns. Subtracting an
  annual simple-rate input from the app's annual log rate mixes conventions,
  hence the explicit approximation label. A saved zero produced by the legacy
  zero-volatility branch is shown as unavailable, not as a valid zero ratio.
  See [Sharpe (1994), *The Sharpe Ratio*](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).
- **Median:** the 50th percentile of the resampled horizon's simple returns.
  Approximately half the sampled outcomes lie on each side. It is a scenario
  summary; the historical bootstrap does not calibrate future probabilities.
- **Expected shortfall (ES) 95%:** the average loss in the worst approximately
  5% of the empirical outcomes. The one-session value uses historical simple
  returns `expm1(p)`; the multi-session value uses resampled horizon returns.
  This calculator includes all losses at or above the empirical 95th loss
  percentile (`method='higher'`) and clips the final average below at zero.
  Finite samples and ties can therefore include more than exactly 5%; this
  display statistic is distinct from the strict allocator's fractional-tail
  CVaR calculation. ES is neither VaR (a loss quantile) nor maximum drawdown
  (a peak-to-trough path loss), and it is not a guaranteed loss ceiling.
  For foundational CVaR work, see Rockafellar & Uryasev (2000), *Optimization
  of Conditional Value-at-Risk*, linked in the
  [author's publications](https://uryasev.github.io/publications/).
- **Tail-loss SD 95%:** sample standard deviation (`ddof=1`) of the
  non-negative losses within that empirical worst tail; a singleton tail
  produces zero. This describes variation within severe scenarios, rather
  than total portfolio volatility, uncertainty of ES, or a confidence bound.
- **Moving-block bootstrap:** the calculator samples adjacent blocks of
  daily modeled log returns, joins them to the stated horizon, and applies
  `expm1` to each summed log path. It retains some dependence within short
  blocks; it does not retain all long-run structure or invent unseen crises.
  The saved metric's horizon is preserved, including legacy 20-session data.
  See [Künsch (1989), *The Jackknife and the Bootstrap for General Stationary
  Observations*](https://doi.org/10.1214/aos/1176347265).

The image distinguishes the weights' saved timestamp from the statistics'
history cutoff. Refreshed statistics use their recorded metrics configuration;
older refreshed records without it show the risk-free assumption as unavailable
instead of substituting today's rate or an earlier configuration. Missing,
non-finite or invalid risk readings stay unavailable. No weights, statistics,
execution decisions or public publication records are modified by this export.
