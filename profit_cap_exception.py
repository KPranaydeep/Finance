"""Explicit, profile-scoped cap exception; no unconstrained profit-only approval."""
import json
import math

import pandas as pd

from execution_diagnostics import cap_breach_ticker
from rebalance_review import assess_executable_rebalance


DEFAULT_PROFIT_EXCEPTION = {"enabled":False,"minimum_net_profit_inr":8079.}


def validate_profit_exception(settings):
    if not isinstance(settings,dict) or set(settings) != set(DEFAULT_PROFIT_EXCEPTION):
        raise ValueError("Invalid profit-cap exception settings")
    amount = settings["minimum_net_profit_inr"]
    if (not isinstance(settings["enabled"],bool) or isinstance(amount,bool)
            or not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount < 1):
        raise ValueError("Profit threshold must be a positive finite INR amount")
    return {"enabled":settings["enabled"],"minimum_net_profit_inr":float(amount)}


def load_profit_exception(conn, owner):
    conn.execute("CREATE TABLE IF NOT EXISTS profit_cap_controls (owner TEXT PRIMARY KEY,settings_json TEXT NOT NULL)")
    row=conn.execute("SELECT settings_json FROM profit_cap_controls WHERE owner=?",(str(owner),)).fetchone()
    return validate_profit_exception(json.loads(row[0])) if row else dict(DEFAULT_PROFIT_EXCEPTION)


def save_profit_exception(conn, owner, settings):
    settings=validate_profit_exception(settings)
    load_profit_exception(conn,owner)
    conn.execute("INSERT INTO profit_cap_controls VALUES(?,?) ON CONFLICT(owner) DO UPDATE SET settings_json=excluded.settings_json",
                 (str(owner),json.dumps(settings,allow_nan=False)))
    conn.commit()


def evaluate_profit_cap_exception(holdings, orders, prices, log_returns, settings, check,
                                   *, cost_bps, cost_stress_multiple, available_cash_inr=0.):
    """Choose one cap-reducing sell nearest above the expected-benefit hurdle.

    `check` must replay the actual quantities with every original guard. All
    non-cap errors remain blockers, including hard maximum weight and funding.
    Unchanged inherited cap breaches remain recorded; the full portfolio is not
    certified compliant by this partial release.
    A historical mean benefit is a model estimate before tax, not a guarantee.
    """
    settings=validate_profit_exception(settings)
    result={"applied":False,"settings":settings,"horizon_sessions":30,"qualified_candidates":0}
    if not settings["enabled"]:
        result["reason"]="Profit-based exception is disabled"
        return None,None,result
    frame=pd.DataFrame(holdings)
    orders=pd.DataFrame(orders).to_dict(orient="records")
    values={}
    try:
        for row in frame.to_dict(orient="records"):
            q=float(row["Quantity"])
            if q < 0 or not math.isfinite(q): raise ValueError("Invalid holding quantity")
            if q:
                t=str(row["Yahoo Ticker"])
                value=q*float(prices[t])*float(row["FX to INR"])
                if not math.isfinite(value) or value <= 0: raise ValueError("Missing positive price/FX")
                values[t]=values.get(t,0.)+value
        baseline=check([])
        initial_caps={t for reason in baseline.get("violations") or [] if (t:=cap_breach_ticker(reason))}
        singles=[[o] for o in orders if o.get("Action")=="Sell" and str(o.get("Yahoo Ticker")) in initial_caps]
        if not singles:
            result["reason"]="No cap-reducing sell order is available"
            return None,None,result
        review=assess_executable_rebalance(log_returns,values,orders,cost_bps=cost_bps,cash=available_cash_inr,
            minimum_net_gain_inr=settings["minimum_net_profit_inr"],cost_stress_multiple=cost_stress_multiple,
            candidate_subsets=singles)
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        result["reason"]=str(exc)
        return None,None,result
    result.update(as_of=review.get("as_of"),observations=review.get("observations",0),
                  confidence_note=review.get("confidence_note"), candidates=review.get("candidates",[]))
    qualified=review.get("qualified_candidates") or []
    result["qualified_candidates"]=len(qualified)
    # Above the threshold, nearest first. Exactly one existing whole sell order.
    for candidate in sorted(qualified,key=lambda c:(c["evidence"]["Stressed net mean gain (INR)"],str(c["orders"][0]["Yahoo Ticker"]))):
        validation=check(candidate["orders"])
        violations=validation.get("violations") or []
        caps={t for reason in violations if (t:=cap_breach_ticker(reason))}
        # Even a high estimated benefit never excuses a different failed guard.
        if any(cap_breach_ticker(reason) is None for reason in violations) or not caps.issubset(initial_caps):
            continue
        order=candidate["orders"][0]
        ticker=str(order["Yahoo Ticker"])
        if order.get("Action")!="Sell" or float(order["Executable Value"])<=0:
            continue
        # Residual inherited breaches are explicit; no exposure may increase.
        capital=float(validation["marked_capital_inr"])-float(validation["estimated_cost_inr"])
        if any(float(w)*capital > values.get(t,0.)+1e-6 for t,w in validation["final_weights"].items()):
            continue
        evidence=candidate["evidence"]
        result.update(applied=True,estimated_additional_net_profit_inr=evidence["Stressed net mean gain (INR)"],
            released_ticker=ticker, remaining_over_cap=sorted(caps), evidence=evidence,
            selection_rule="Nearest qualifying net benefit at or above the INR threshold; one cap-reducing sell",
            reason="One cap-reducing sale passed paired 30-session benefit/stability and cost-stress checks. Residual inherited caps remain unresolved; all non-cap checks passed.")
        # Preserve every original breach for audit, rather than changing its cap.
        validation=dict(validation,passed=True,full_allocation_passed=not violations,
                        execution_scope="single-cap-reducing-sell",violations=[],cap_exception_violations=list(violations),
                        profit_cap_exception=result)
        if validation.get("allocation_policy"):
            policy=dict(validation["allocation_policy"])
            policy.update(cap_exception_applied=True)
            validation["allocation_policy"]=policy
        return pd.DataFrame(candidate["orders"]),validation,result
    result["reason"]="No tested candidate met the INR threshold, stability/tail conditions and all non-cap execution checks"
    return None,None,result
