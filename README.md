# CI policy-test isolation fix — v47

The failing test read your real public_review_policy.json and assumed approval was
false. Approving the real policy correctly made that assumption false. This was a
test bug, not a reason to disable your approved production policy.

Replace these two files in GitHub main, preserving paths:

- tests/test_review_operations.py
- tests/review_fixtures.py

The rejection test now supplies explicitly unapproved in-memory data to the real
policy loader. Other tests use independent, fresh fixtures instead of reading your
production settings. Added coverage verifies approved policies, expired tariffs,
strict boolean approval and fixture isolation. Test dates are fixed so these tests
do not change behavior as the calendar advances.

Do not change public_review_policy.json, your secrets, or any runtime code for this
fix. The real policy approval and freshness guards are unchanged.

Commit the replacements to main and use the new push-triggered test run. Re-running
the old failed run uses its old commit and will repeat the old failure.

Based on main 75def9d0af087dcad8673dd3516d5c0739c4a711.
