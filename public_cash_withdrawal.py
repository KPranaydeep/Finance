"""Illustrative withdrawal default; no actual investor holdings are assumed."""
import math


def suggested_withdrawal(model_plan, entry_estimate, constituents):
    if model_plan.get("missing_prices"):
        return None
    weights={str(row["ticker"]):float(row["target_weight"]) for row in constituents}
    assumptions=entry_estimate["assumptions"]
    fixed=float(assumptions["fixed_cost_per_order"])
    rate=float(assumptions["statutory_cost_rate"])+float(assumptions["slippage_rate"])
    if not math.isfinite(fixed) or fixed<0 or not math.isfinite(rate) or not 0<=rate<1:
        raise ValueError("Invalid cost assumptions")
    candidates=[]
    for row in model_plan.get("orders",[]):
        ticker=str(row["ticker"])
        quantity=int(row["quantity"])
        gross=quantity*float(row["planning_price"])
        weight=weights.get(ticker,0)
        if quantity<=0 or not math.isfinite(gross) or gross<=0 or not math.isfinite(weight) or weight<=0:
            continue
        costs=fixed+gross*rate
        net=math.floor((gross-costs)*100)/100
        if net>=1:
            candidates.append((weight,gross,ticker,net,costs,quantity))
    if not candidates:
        return None
    weight,gross,ticker,net,costs,quantity=min(candidates)
    return {"amount_inr":net,"reference_ticker":ticker,"reference_quantity":quantity,
            "reference_gross_inr":gross,"estimated_cost_inr":costs,
            "reference_capital_inr":float(entry_estimate["minimum_capital_inr"])}


def withdrawal_instructions(amount):
    amount=float(amount)
    if not math.isfinite(amount) or amount<1:
        raise ValueError("Requested withdrawal must be at least INR 1")
    return f"""REQUESTED NET CASH: INR {amount:.2f}.
The user has already chosen this amount. Do not ask for it again.
Read the user's broker report; ask for it if missing. The website's suggested
amount is illustrative, not evidence of the user's capital or holdings.
Use existing cash only when confirmed withdrawable. Otherwise assume zero
withdrawable cash and state this once. Raise only the remaining shortfall.
Prioritize non-target holdings, then excess allocations calculated against
the portfolio value remaining AFTER withdrawal and estimated selling costs.
If necessary, trim other holdings while limiting target drift and excess sales.
Do not sell a stock just because it was the model's small reference holding.
Use report prices, quantities and values; never invent holdings or cost basis.
For this withdrawal, do not apply the rebalance minimum-trade or drift filters
as a hard barrier. INR 1 is a valid cash request; whole-share sales can exceed it.
Choose few practical sales, estimate charges explicitly, and show any excess cash.
Do not invent capital-gains tax without cost-basis information.
If confirmed cash already covers the request, show no stock sales.
If holdings cannot cover the net request, show the shortfall; never oversell shares.
Show requested cash, estimated net proceeds, charges and excess/shortfall.
For the withdrawal audit, net funds = confirmed withdrawable cash + gross sales
- estimated selling charges; excess/shortfall = net funds - requested cash.
Use this withdrawal reconciliation rather than the fresh-investment residual formula.
Do not recommend buys for this scenario. This is a planning estimate to verify
with the broker before execution."""
