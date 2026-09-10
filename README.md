# Review metadata and known-cost funding update

Upload the included files into Finance at their matching paths; commit on main.
Keep your existing public_review_policy.json and secrets. No database migration.
Run the model-review workflow, and run the private optimizer again to produce a
new analysis backup JSON containing review_analysis_context. Existing immutable
publications are not rewritten. This metadata is provenance in the analysis
backup, not yet persisted into the public publication schema or consumed as a
replacement for fresh review history.

Supported NSE names are resolved from exchange metadata. US USD equity/ETF
listings are identified using Yahoo metadata. Existing owner entries are kept;
unknown instruments fail explicitly. The workflow resolves its checkout policy
on each run; you do not need to commit new ticker entries. US listing identity
does not determine domicile, ADR/partnership status or tax treatment.

The new funding card uses Tickertape Pro brokerage plus GST and HDFC FX GST.
It is a known-charge floor on an INR100 grid, not a complete investment minimum.
It does NOT change execution-plan defaults or authorize trades. Zero platform
and gateway fees follow the supplied funding screenshot only. FX spread,
regulatory fees, exit costs, TCS/taxes and subscription costs are excluded.
The pure funding calculator supports amounts up to INR10 lakh only.

Sources checked 2026-09-10:
- https://www.tickertape.in/us-stocks/pricing
- https://www.tickertape.in/blog/how-to-invest-in-us-stocks/
- https://www.hdfc.bank.in/remittance/fees-and-charges

IMPORTANT: This is a partial integration. FOREIGN_REVIEW_COST_MODEL_REQUIRED
remains for mixed/US portfolios. Their after-tax per-security crossing dates
are NOT enabled by this patch. A verified foreign exit/tax model and mixed-market
session support remain necessary. Domestic crossing tables open by default;
dates are probabilistic research estimates, never promises of 100% XIRR.

Validation: 79 review tests and 4 funding tests passed. Both edited Streamlit
scripts passed AST syntax checks. Cloud runtime and production DB were not run.
