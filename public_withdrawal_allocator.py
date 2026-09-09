"""Whole-share sell-only optimization for explicitly supplied model holdings."""
import math
from decimal import Decimal, ROUND_HALF_UP


def cents(value):
    value=Decimal(str(value))
    if not value.is_finite() or value<0:
        raise ValueError("Money values must be finite and nonnegative")
    return int((value*100).quantize(Decimal("1"),rounding=ROUND_HALF_UP))


def allocate_withdrawal(plan, constituents, requested, assumptions):
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    request=cents(requested)
    if request<100:
        raise ValueError("Withdrawal must be at least INR 1")
    if plan.get("missing_prices"):
        raise ValueError("Complete model prices are required")
    rows=sorted(plan["orders"],key=lambda r:r["ticker"])
    if len({r["ticker"] for r in rows})!=len(rows):
        raise ValueError("Duplicate model holdings")
    cash=cents(plan["residual_cash_inr"])
    fixed=cents(assumptions["fixed_cost_per_order"])
    rate=float(assumptions["statutory_cost_rate"])+float(assumptions["slippage_rate"])
    if not math.isfinite(rate) or not 0<=rate<1:
        raise ValueError("Invalid proportional costs")
    prices=np.array([cents(r["planning_price"]) for r in rows],dtype=float)
    quantities=np.array([r["quantity"] for r in rows],dtype=float)
    if np.any(prices<=0) or np.any(quantities<0) or np.any(quantities!=np.floor(quantities)):
        raise ValueError("Invalid whole-share model holdings")
    target={r["ticker"]:float(r["target_weight"]) for r in constituents}
    if any(not math.isfinite(w) or w<0 for w in target.values()) or sum(target.values())>1.000001:
        raise ValueError("Invalid target weights")
    n=len(rows)
    weights=np.array([target.get(r["ticker"],0) for r in rows])
    value=float(np.dot(prices,quantities)+cash)
    if cash>=request:
        return dict(status="CASH_ONLY",orders=[],gross_sales=0.0,costs=0.0,
                    requested=request/100,available_cash=cash/100,residual_cash=(cash-request)/100)
    # Sale quantities, binary order indicators, absolute position/cash deviations,
    # and an integer-cent aggregate fee rounded upward.
    size=3*n+2; fee=size-1; cashdev=3*n
    objective=np.zeros(size)
    objective[2*n:3*n+1]=1
    objective[fee]=1
    lower=np.zeros(size); upper=np.full(size,np.inf)
    upper[:n]=quantities; upper[n:2*n]=1
    integer=np.zeros(size); integer[:2*n]=1; integer[fee]=1
    matrix=[]; lows=[]; highs=[]
    def constraint(coeff,lo=-np.inf,hi=np.inf):
        matrix.append(coeff); lows.append(lo); highs.append(hi)
    for i in range(n):
        a=np.zeros(size); a[i]=1; a[n+i]=-quantities[i]
        constraint(a,hi=0)
        a=np.zeros(size); a[i]=1; a[n+i]=-1
        constraint(a,lo=0)
    a=np.zeros(size); a[fee]=1; a[:n]=-prices*rate; a[n:2*n]=-fixed
    constraint(a,lo=-1e-7,hi=1-1e-7)
    a=np.zeros(size); a[:n]=prices; a[fee]=-1
    constraint(a,lo=request-cash)
    for i in range(n):
        # remaining holding minus weight * (initial value - request - fees)
        constant=prices[i]*quantities[i]-weights[i]*(value-request)
        a=np.zeros(size); a[i]=-prices[i]; a[fee]=weights[i]
        positive=a.copy(); positive[2*n+i]=-1
        negative=-a; negative[2*n+i]=-1
        constraint(positive,hi=-constant); constraint(negative,hi=constant)
    cash_weight=max(0,1-sum(target.values()))
    a=np.zeros(size); a[:n]=prices; a[fee]=cash_weight-1
    constant=cash-request-cash_weight*(value-request)
    positive=a.copy(); positive[cashdev]=-1
    negative=-a; negative[cashdev]=-1
    constraint(positive,hi=-constant); constraint(negative,hi=constant)
    result=milp(objective,integrality=integer,bounds=Bounds(lower,upper),
                constraints=LinearConstraint(np.array(matrix),lows,highs),
                options={"time_limit":10,"mip_rel_gap":0.0})
    if result.status==2:
        full_values=prices*quantities
        profitable=full_values*(1-rate)>fixed
        gross=float(full_values[profitable].sum())
        costs=math.ceil(gross*rate+fixed*int(np.count_nonzero(profitable))-1e-7)
        maximum=max(0,cash+gross-costs)
        return dict(status="INSUFFICIENT",orders=[],requested=request/100,
                    maximum_available=maximum/100,shortfall=max(0,request-maximum)/100)
    if result.status!=0 or result.x is None:
        raise RuntimeError("Withdrawal calculation did not finish; try a different amount")
    sold=np.rint(result.x[:n]).astype(int)
    gross=int(np.dot(prices,sold))
    fees=math.ceil(gross*rate+fixed*int(np.count_nonzero(sold))-1e-7)
    residual=cash+gross-fees-request
    if np.any(sold<0) or np.any(sold>quantities) or residual<0:
        raise RuntimeError("Withdrawal reconciliation failed")
    orders=[{"Action":"SELL","Ticker":r["ticker"],"Shares":int(sold[i]),
             "Price":prices[i]/100,"Approx. value":sold[i]*prices[i]/100,
             "Remaining shares":int(quantities[i]-sold[i])} for i,r in enumerate(rows) if sold[i]>0]
    return dict(status="PLANNED",orders=orders,requested=request/100,gross_sales=gross/100,
                costs=fees/100,available_cash=cash/100,residual_cash=residual/100)
