# Security-level target review (v49)

## Install on Streamlit Cloud

Upload all public_review/ and tests/ files in this ZIP to the same paths in Finance/main, replacing existing files. Commit together, then start a NEW Public portfolio model review workflow run on main. Refresh Streamlit after deployment.

Based on main cead7990287ffeb58f9f73ee24022380df56d1bf (v48). No changes to secrets, approved policy, workflow schedule, optimizer, actual trades or existing records. No new dependencies.

## Behavior

- Each modeled held security gets a first-passage estimate for the existing target_xirr. 1.0 means 100% annualized after the configured modeled liquidation deductions, not a doubling of principal.
- Entry outlay includes entry charges. Forecast exits use the existing conservative fee/slippage/tax model; dated net dividends enter the security target calculation at their actual modeled dates.
- The first security whose cumulative crossing probability reaches crossing_probability supplies the earliest security candidate. One lucky path or a union of individually weak security probabilities does not qualify.
- Basket/risk triggers can require an earlier review. Today's security-level XIRR uses the existing exact liquidation engine. A crossed target requests a review, never automatically sells or bypasses the existing rebalance benefit gate.
- New method/version is included in assessment identity. Previous-method validation cannot approve the new method; walk-forward outcomes now include security targets.
- If validation fails, research crossing dates remain visible as research only. The suggested date uses a next-session risk check, not an invented validated target date.
- The page performs a read-only historical assessment, cached up to five minutes, without needing a workflow run. Uses latest completed daily prices with the existing close-plus-30-minute availability buffer, NOT intraday/live quotes.
- Before a frozen entry exists, the page assumes a hypothetical entry today at the latest completed close. This is explicitly provisional, not an executable historic entry or earned performance. Once a baseline exists, it uses that frozen entry instead.
- Workflow-created review dates and already-due triggers are retained until acknowledgment. Page-only earlier dates are retained within the browser session; read-only page views do not persist dates or acknowledgments across new browser sessions. The workflow remains the durable monitoring/notification record.

The existing crossing_probability setting is a model probability threshold (often 0.20), NOT a 95% statistical confidence claim. Tax treatment remains limited to the existing approved account/instrument model, not every investor's actual liability. A review date is not a guaranteed best selling date.

## Verification

61 local review tests passed, including security-versus-basket crossing, entry/exit costs, dated dividends, method validation, probability thresholds, retained review dates, provisional preview and headless Streamlit display. Production database, live quote availability, notification delivery and deployment were not tested or changed.

Native Streamlit status/metric/table components and bounded caching follow the Streamlit skill. The new page calculation skips expensive sale-plan optimization; the scheduled assessment still produces those comparisons.
