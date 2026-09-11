# P1 partial security reviews

Captured securities no longer have to wait for every market constituent before
receiving a research estimate. The workflow creates immutable
`SECURITY_REVIEW_PREVIEW` events independently for each captured entry.

## Notional convention

Before the complete basket exists, its final capital and whole-share quantities
are unknowable. P1 therefore uses exactly one share of the captured security and
includes modeled entry charges, exit charges, slippage, tax and foreign-funding
FX GST where applicable. This is deterministic and conservative about fixed
costs; it does not invent a portfolio corpus.

The estimate uses that security's own exchange calendar and its own available
INR return history. A direct US listing includes USD/INR history. It reports the
first simulated date at which the configured net-XIRR probability threshold is
reached, or that the target is not reached within the configured horizon.

## Authority boundary

Partial dates are labelled `PROVISIONAL_RESEARCH`. They are not basket review
dates, sell dates, recommendations or validated signals. Once every security is
captured, the completed basket baseline—with its actual model capital,
whole-share quantities and cross-security risk—supersedes all partial estimates.

Failure to calculate an optional partial estimate never blocks entry capture or
changes the ledger. Entry evidence and final basket review remain authoritative.
