"""Private pause-triggered optimization and explicit frozen-sleeve research.

Research exceptions never change the strict policy or authorize execution.
"""
from datetime import datetime, timezone
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from risk_aware_allocator import current_weights, build_policy_limits, joint_block_scenarios, scenario_cvar, check_policy_weights

FLAG = "Paused 90 days"
NOTE = "Fresh tradable weights calculated; frozen exposure and inherited risk breaches remain explicit. Research only; execution requires strict checks."


def archive_before_conditional_save(conn, owner, payload):
    """Preserve the prior snapshot verbatim in the same transaction as replacement."""
    if ((payload.get("pause_allocation_research") or {}).get("status") != "conditional_research"
            and not (payload.get("optimization_scope") or {}).get("excluded_tickers")):
        return
    conn.execute("""CREATE TABLE IF NOT EXISTS pause_analysis_archive(
        owner TEXT NOT NULL, payload_hash TEXT NOT NULL, saved_at TEXT NOT NULL,
        payload_json TEXT NOT NULL, archived_at TEXT NOT NULL,
        PRIMARY KEY(owner,payload_hash))""")
    previous=conn.execute("SELECT saved_at,payload_json FROM latest_analysis WHERE owner=?",(owner,)).fetchone()
    if previous is None:
        return
    saved_at,raw=previous
    conn.execute("""INSERT OR IGNORE INTO pause_analysis_archive VALUES(?,?,?,?,?)""",
                 (owner,hashlib.sha256(raw.encode()).hexdigest(),saved_at,raw,datetime.now(timezone.utc).isoformat()))


def prepare_pause_allocation(allocation, pauses):
    """Optimize only eligible capital; a pause never implies liquidation/funding."""
    frame=pd.DataFrame(allocation).copy()
    attributes=dict(getattr(allocation,"attrs",{}))
    active=set(pauses["yahoo_ticker"].astype(str).str.strip().str.upper()) if pauses is not None and not pauses.empty else set()
    names=frame["Yahoo Ticker"].astype(str).str.strip().str.upper()
    owned=pd.to_numeric(frame["Quantity"],errors="coerce")
    if owned.isna().any() or owned.lt(0).any():
        raise ValueError("Valid owned quantities required for pause-aware optimization")
    keep=~names.isin(active)
    previous=attributes.get("optimization_scope") or {}
    excluded={str(r["Yahoo Ticker"]).upper():r for r in previous.get("excluded_holdings") or []}
    excluded.update({str(r["Yahoo Ticker"]).upper():r for r in frame.loc[~keep & owned.gt(0)].to_dict(orient="records")})
    frame=frame.loc[keep].copy()
    weights=pd.to_numeric(frame["Weight"],errors="coerce")
    if weights.isna().any() or weights.lt(0).any():
        raise ValueError("Valid current weights required for pause exclusion")
    total=float(weights.sum())
    frame["Weight"]=weights/total if total>0 else weights
    frame[FLAG]=False
    frame.attrs.update(attributes)
    frame.attrs["pause_reoptimization"]=bool(active)
    frame.attrs["optimization_scope"]={"mode":"unpaused-capital-v1",
        "excluded_tickers":sorted(active | set(previous.get("excluded_tickers") or [])),
        "excluded_holdings":list(excluded.values()),
        "note":"Weights, metrics and trade sizes apply only to unpaused capital. Paused holdings remain in the holdings database and are not sold or used to fund buys."}
    return frame,int((~keep).sum())


def eligible_pause_history(log_returns, allocation):
    excluded=set((allocation.attrs.get("optimization_scope") or {}).get("excluded_tickers") or [])
    return log_returns.loc[:,~log_returns.columns.astype(str).str.upper().isin(excluded)].copy()


def render_pause_exclusion(scope):
    if scope and scope.get("mode")=="unpaused-capital-v1" and scope.get("excluded_tickers"):
        import streamlit as st
        st.info(f"Optimization excludes {len(scope['excluded_tickers'])} paused securities. "
                "Weights, risk metrics and trade sizes cover unpaused capital only; paused holdings stay unchanged.")


def claim_pause_reoptimization(conn, owner, payload, pauses):
    """Claim once per durable pause change, not once per page load or new run ID."""
    if not payload or pauses is None or pauses.empty:
        return False
    targets={str(r.get("ticker","")).upper() for r in payload.get("optimal_allocation") or [] if float(r.get("target_weight") or 0)>0}
    active=set(pauses["yahoo_ticker"].astype(str).str.upper())
    if not targets & active:
        return False
    rows=pauses[["yahoo_ticker","deferred_at","expires_at"]].sort_values("yahoo_ticker").to_dict(orient="records")
    signature=hashlib.sha256(("exclude-paused-v1:"+json.dumps(rows,sort_keys=True)).encode()).hexdigest()
    conn.execute("CREATE TABLE IF NOT EXISTS pause_allocation_attempts(owner TEXT PRIMARY KEY, signature TEXT NOT NULL, claimed_at TEXT NOT NULL)")
    result=conn.execute("""INSERT INTO pause_allocation_attempts VALUES(?,?,?)
        ON CONFLICT(owner) DO UPDATE SET signature=excluded.signature,claimed_at=excluded.claimed_at
        WHERE pause_allocation_attempts.signature!=excluded.signature""",
        (str(owner),signature,datetime.now(timezone.utc).isoformat()))
    conn.commit()
    return result.rowcount==1


def paused_owned_names(allocation):
    frame=pd.DataFrame(allocation)
    if FLAG not in frame:
        return set()
    return set(frame.loc[frame[FLAG].eq(True) & pd.to_numeric(frame["Quantity"],errors="coerce").gt(0),"Yahoo Ticker"].astype(str).str.upper())


def solve_frozen_sleeve_research(log_returns, allocation, bounds, policy, *, cost_bps=22.):
    """Fresh weights with fixed full-portfolio exposure; strict checks stay separate.

First require non-worsening tail/group risk versus today's whole portfolio.
If that is impossible, minimize worst-book CVaR with explicit failed risk gates.
Both lanes retain frozen equality and strict position caps on tradable assets.
Costs are charged to scenario returns, not a claim of exact funded execution.
"""
    started=time.perf_counter()
    report={"status":"unavailable","selected":"none","constraints_applied":False,"research_only":True}
    try:
        frame=pd.DataFrame(log_returns)
        initial=current_weights(allocation,frame.columns)
        stressed,limits=build_policy_limits(frame,initial,bounds,policy,cost_bps)
        policy=limits["settings"]
        books=[stressed,joint_block_scenarios(frame,policy)] if policy["adaptive_enabled"] else [stressed]
        n=len(frame.columns); m=len(stressed); k=len(books)
        frozen={t:float(lo) for t,(lo,hi) in zip(frame.columns,bounds) if lo>0 and abs(lo-hi)<1e-12}
        caps=np.array([limits["position_caps"][t] for t in frame.columns])
        box=[(frozen[t],frozen[t]) if t in frozen else (float(lo),float(cap)) for t,(lo,_),cap in zip(frame.columns,bounds,caps)]
        if sum(b[1] for b in box)<1-1e-9:
            raise ValueError("Tradable position capacity cannot invest the remaining capital; cash-sleeve planning is required")
        # Variables: full-base weights, absolute turnover, per-book VaR and tail slacks.
        size=2*n+k*(m+1)
        eye=sparse.eye(n,format="csr"); zeros=sparse.csr_matrix((n,size-2*n))
        base_matrices=[sparse.hstack([eye,-eye,zeros]),sparse.hstack([-eye,-eye,zeros])]
        base_rhs=[initial,-initial]; tail_rows=[]; thresholds=[]
        rate=limits["cost_rate"]
        for b,book in enumerate(books):
            var=sparse.csr_matrix((-np.ones(m),(np.arange(m),np.full(m,b))),shape=(m,k))
            slack=sparse.hstack([sparse.csr_matrix((m,b*m)),-sparse.eye(m),sparse.csr_matrix((m,(k-b-1)*m))])
            base_matrices.append(sparse.hstack([-sparse.csr_matrix(book),sparse.csr_matrix(np.full((m,n),rate)),var,slack]))
            base_rhs.append(np.zeros(m))
            row=np.zeros(size); row[2*n+b]=1.; row[2*n+k+b*m:2*n+k+(b+1)*m]=1/((1-policy["confidence"])*m)
            tail_rows.append(row); thresholds.append(scenario_cvar(book@initial,policy["confidence"]))
        guarded=list(base_matrices); guarded_rhs=list(base_rhs)
        guarded.extend(sparse.csr_matrix(r.reshape(1,-1)) for r in tail_rows)
        guarded_rhs.extend(np.array([v]) for v in thresholds)
        group_caps=[]
        for group in limits["groups"]:
            row=np.zeros(size); members=[frame.columns.get_loc(t) for t in group["tickers"]]
            row[members]=1.; cap=max(group["cap"],float(initial[members].sum()))
            guarded.append(sparse.csr_matrix(row.reshape(1,-1)));guarded_rhs.append(np.array([cap]));group_caps.append(cap)
        means=np.array([limits["objective_means"][t] for t in frame.columns])
        means=(1-policy["return_shrinkage"])*means+policy["return_shrinkage"]*np.median(means)
        objective=np.r_[-means,np.full(n,rate),np.zeros(size-2*n)]
        equal=sparse.csr_matrix(np.r_[np.ones(n),np.zeros(size-n)].reshape(1,-1))
        kwargs=dict(A_eq=equal,b_eq=[1.],bounds=box+[(0.,2.)]*n+[(None,None)]*k+[(0.,None)]*(k*m),method="highs",options={"time_limit":policy["solver_time_limit_seconds"]})
        solved=linprog(objective,A_ub=sparse.vstack(guarded,format="csr"),b_ub=np.concatenate(guarded_rhs),**kwargs)
        lane="non_worsening"
        if not solved.success and solved.status==2:
            # Explicit research fallback, not a passed policy or executable plan.
            objective=np.sum(tail_rows,axis=0)+1e-6*objective
            solved=linprog(objective,A_ub=sparse.vstack(base_matrices,format="csr"),b_ub=np.concatenate(base_rhs),**kwargs)
            lane="minimum_tail_research"
        if not solved.success:
            raise ValueError("Conditional research solve did not establish weights: "+str(solved.message))
        weights=solved.x[:n]
        if not np.isfinite(weights).all() or abs(weights.sum()-1)>1e-7 or any(w<lo-1e-7 or w>hi+1e-7 for w,(lo,hi) in zip(weights,box)):
            raise ValueError("Conditional research weights failed independent bound checks")
        cost=rate*float(np.abs(weights-initial).sum())
        tails=[scenario_cvar(book@weights-cost,policy["confidence"]) for book in books]
        non_worsening=all(v<=threshold+1e-7 for v,threshold in zip(tails,thresholds)) and all(sum(weights[frame.columns.get_loc(t)] for t in g["tickers"])<=cap+1e-7 for g,cap in zip(limits["groups"],group_caps))
        if lane=="non_worsening" and not non_worsening:
            raise ValueError("Conditional non-worsening risk checks failed")
        inherited=[{"ticker":t,"frozen_weight":w,"risk_cap":limits["position_caps"][t],"excess":w-limits["position_caps"][t]} for t,w in frozen.items() if w>limits["position_caps"][t]+1e-9]
        strict=check_policy_weights(weights,frame,limits)
        report.update(status="conditional_research",selected="tradable-research",new_allocation_established=True,
            limits=limits,policy=policy,frozen_weights=frozen,inherited_breaches=inherited,
            frozen_fraction=sum(frozen.values()),tradable_fraction=1-sum(frozen.values()),
            candidate=strict,conditional_lane=lane,non_worsening=non_worsening,
            conditional_tail_references=thresholds,conditional_tail_results=tails,
            conditional_group_caps=group_caps,history_through=limits["history_through"],reason=NOTE,
            cost_convention="Scenario-return cost drag; exact self-funded execution is not established")
        return weights,report
    except (ValueError,TypeError,KeyError,OverflowError) as exc:
        report["reason"]=str(exc)
        return None,report
    finally:
        report["elapsed_seconds"]=round(time.perf_counter()-started,3)


def withhold_research_plan(plan):
    """Research weights cannot acquire execution approval through plan refresh."""
    attributes=dict(plan.attrs)
    validation=dict(attributes.get("execution_validation") or {})
    validation.update(passed=False,violations=list(validation.get("violations") or [])+[
        "Conditional allocation research requires verified inputs and a new strict optimization before execution approval."])
    attributes["execution_validation"]=validation
    attributes["proposed_rebalancing_plan"]=attributes.get("two_trade_plan") or attributes.get("proposed_rebalancing_plan") or plan.to_dict(orient="records")
    result=plan.iloc[:0].copy();result.attrs.update(attributes)
    return result


def render_pause_research(report):
    if not report or report.get("status")!="conditional_research":
        return
    import streamlit as st
    if report.get("research_kind")=="price_freeze_override":
        st.info("Weight freeze released: "+", ".join(report.get("weight_freeze_overrides") or [])+". The optimizer can resize these holdings within its risk limits.")
        st.warning("Historical price jumps remain unverified. These are fresh review-only weights; no executable plan or public publication is approved.")
        return
    if report["tradable_fraction"]<=1e-9:
        st.info("Frozen portfolio recalculated: no capital is currently tradable. Frozen holdings were not reassigned to new securities; change the pauses or supply new capital before a new allocation can be funded.")
        return
    st.info(f"Fresh tradable allocation calculated: {report['tradable_fraction']:.2%} tradable · {report['frozen_fraction']:.2%} frozen. This is new allocation research, not reused saved weights. Execution remains subject to strict checks.")
    if not report.get("non_worsening"):
        st.warning("The least-tail-risk research candidate could not establish non-worsening whole-portfolio risk. It is not an executable allocation.")
    with st.expander("Inherited frozen exposure and research limits",expanded=False):
        st.write(report.get("inherited_breaches") or "No inherited single-security breach.")
        st.caption(report.get("cost_convention", ""))
