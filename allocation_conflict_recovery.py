"""Continue private research at saved weights; never invent a feasible optimum."""
from datetime import datetime, timezone
import math
from allocation_attempts import failure_summary
from optimization_run_timer import safe_stop_reason

RECOVERY_NOTE = "New allocation infeasible; saved weights retained and candidates/scenarios refreshed."


def recover_saved_allocation_review(report, owner, *, load, refresh, frozen_tickers=()):
    if report.get("status") not in {"infeasible","failed_recheck"}:
        return {"status":"not_applicable"}
    saved=load(owner)
    if not saved or not saved.get("optimal_allocation"):
        return {"status":"unavailable","reason":"No saved allocation is available for a research refresh."}
    frozen=set(str(t).upper() for t in frozen_tickers)
    frozen.update(str(r["ticker"]).upper() for r in (report.get("capacity") or {}).get("conflicts") or []
                  if r.get("kind")=="Frozen holding")
    limits=report.get("limits") or {}
    for ticker,lower in (limits.get("position_lower_bounds") or {}).items():
        upper=(limits.get("position_upper_bounds") or {}).get(ticker)
        if isinstance(lower,(int,float)) and isinstance(upper,(int,float)) and lower>0 and math.isclose(lower,upper,abs_tol=1e-12,rel_tol=0):
            frozen.add(str(ticker).upper())
    metadata={"status":"completed_with_conflicts","new_allocation_established":False,
        "source_run_id":saved.get("run_id"),"source_saved_at":saved.get("saved_at"),
        "optimizer_status":report["status"],"reason":failure_summary(report),
        "frozen_tickers":sorted(frozen),"capacity":report.get("capacity") or {},
        "refresh_started_at":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        refreshed=refresh(saved,owner,allocation_recovery=metadata)
    except Exception as exc:
        return {"status":"refresh_failed","reason":safe_stop_reason(str(exc))}
    return {"status":"completed_with_conflicts","analysis":refreshed,"recovery":metadata}


def render_allocation_recovery(metadata):
    if not metadata:
        return
    import streamlit as st
    st.info("Analysis completed with allocation conflict. Using saved weights; candidate selections and scenarios were refreshed. No new optimized allocation was established.")
    st.caption(f"Original allocation: {metadata.get('source_saved_at','unavailable')} · review refreshed: {metadata.get('refreshed_at','unavailable')}. Risk limits and frozen holdings were not relaxed.")
    with st.expander("Why no new allocation was established",expanded=False):
        st.write(metadata.get("reason") or "Allocation constraints conflict.")
