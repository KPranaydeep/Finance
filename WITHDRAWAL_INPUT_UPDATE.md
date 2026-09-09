# Cash withdrawal input

Based on Finance main c82aab051e2423cd3cd25f47f7f10764286b15ef.

Upload both root Python files and preserve the tests/ path. No workflow or database migration is needed.

Raise cash from existing holdings now has an editable Cash to raise input starting at INR 1. The suggested default uses the practical-entry reference allocation: select the lowest published weight first, then the smallest whole-share position value, subtracting the entry estimate's disclosed fixed and proportional cost assumptions. This is an illustrative amount, not an actual investor valuation or broker charge quotation.

If prices or the estimate are unavailable, the input starts at INR 1 with an unavailable-suggestion caption. The selected amount is embedded in copied/downloaded execution prompts. Actual sales depend on the private broker report and confirmed withdrawable cash. Rebalance thresholds do not block the withdrawal request, and whole-share excess proceeds must be shown.

The buy default, INR 1 buy floor, 1Y chart default, MMI and brokerage links are retained.

Four focused tests passed, including the generated prompt with INR 1, and Python syntax checks passed. Live Streamlit rendering was not tested.
