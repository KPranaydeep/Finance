# Manual stocks and FX update

Upload the included Python files to the matching Finance repository paths and
commit on main. No secrets, policy JSON or database migration changes are needed.

In the private optimizer, open Master Holdings > Add stocks manually. Enter
comma/newline-separated Yahoo tickers, then Add to my holdings. New rows start
at quantity 1 and a Yahoo native-currency price placeholder. Use Edit quantity
and average price to replace these with your actual holdings before analysis.
Existing rows are never reset. Unresolved tickers and missing prices are reported.

The holdings table now shows reference FX and entered cost converted to INR.
Average Price remains in its native currency. This prevents double conversion.
The INR equivalent is not a reconstruction of historical INR purchase cost.
FX and latest-price caches expire after five minutes. Refresh FX to INR clears
the FX cache explicitly; new additions also clear price and FX caches.
The existing optimization calculations already convert foreign histories to INR.

The attached service.py fixes the diagnostic stage from session_calendar to
instrument_cost_model. It DOES NOT remove FOREIGN_REVIEW_COST_MODEL_REQUIRED.
The US tax/exit model and mixed-market calendar integration remain pending.
Funding GST and Pro brokerage alone do not complete that model. This patch does
not generate unsupported post-tax target dates or turn a failed review green.

Manual persistence tests use an isolated in-memory SQLite database, not your
holdings or production PostgreSQL. The main script passed syntax validation.
The main GitHub optimizer matched the previously delivered file before editing.
