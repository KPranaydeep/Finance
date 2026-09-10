# Model review waiting-state fix (v48)

## Deploy through GitHub

1. Extract this ZIP. Replace these four files in Finance/main, preserving folders:
   - public_review/market.py
   - public_review/service.py
   - public_review/ui.py
   - update_public_review.py
2. Add tests/test_review_waiting.py to the existing tests folder. It uses the existing review test fixtures.
3. Commit the files. Do not replace public_review_policy.json or change your secrets.
4. In GitHub Actions, open Public portfolio model review and choose Run workflow on main. Start a new run so it uses this commit; rerunning an old job uses its old commit.
5. Refresh Streamlit after deployment (the monitoring cache can take five minutes).

## What changes

- A publication awaiting its first eligible completed trading session is WAITING, not a failed check. It does not create a baseline, fetch market history, send an alert, or execute trades while waiting.
- The page shows the eligible session and earliest assessment time in Asia/Kolkata, when available.
- Genuine failures still fail the workflow and now expose allowlisted reason codes and the failed stage, without printing raw exception messages or credentials.
- The empty state no longer assumes policy approval is the problem.
- Existing immutable records are retained. Legacy waiting failures are also displayed as waiting before a baseline exists.

For the supplied P006 publication at 19:41 IST on 9 September 2026, the calendar test identifies 10 September at 16:00 IST as the earliest assessment (session close plus the existing 30-minute buffer), subject to price availability. Dates are derived from the publication and calendar, not hardcoded in runtime code.

The original production failure reason was not provided. This patch fixes a confirmed waiting-state bug; if another blocker exists, the new workflow result should identify its safe reason/stage. Do not disable approval or freshness checks to force success.

## Verification

52 review tests passed locally, including seven new regression tests and headless Streamlit checks. No live database was accessed, no workflow was triggered, and no production deployment was performed.

Implementation uses native Streamlit status messages and captions; no custom styling or new dependencies.
