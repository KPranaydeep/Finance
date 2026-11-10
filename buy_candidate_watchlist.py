"""Funding-independent research candidates; never an execution approval."""
import math
import pandas as pd
from security_scenario_forecast import normalize_saved_scenario_ranking


def _number(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):
        return None


def buy_candidate_watchlist(payload, paused=()):
    """Restore positive saved targets even when no funded buy order survives.

    Preserve listing-qualified tickers. Do not guess missing weights, quotes,
    scores, cash or quantities. Old payloads work without a new optimization.
    """
    targets={}
    for row in payload.get("optimal_allocation") or []:
        ticker=str(row.get("ticker") or row.get("Yahoo Ticker") or "").strip().upper()
        weight=_number(row.get("target_weight",row.get("Optimal Weight")))
        if ticker and weight is not None and weight>0:
            targets[ticker]=weight
    current={str(r.get("Yahoo Ticker") or "").strip().upper():r for r in payload.get("current_allocation") or []}
    drafts={}
    # New snapshots retain pre-funding orders; older snapshots retain their draft.
    for field in ("rebalancing_plan","proposed_rebalancing_plan","buy_candidate_orders"):
        for row in payload.get(field) or []:
            if row.get("Action")=="Buy":
                ticker=str(row.get("Yahoo Ticker") or "").strip().upper()
                drafts[ticker]=row
                weight=_number(row.get("Optimal Weight"))
                if ticker and ticker not in targets and weight is not None and weight>0:
                    targets[ticker]=weight
    records=[dict(r) for r in payload.get("security_scenario_ranking") or [] if r.get("Yahoo Ticker")]
    for r in records:
        r.setdefault("Expected net scenario return (%)",None)
        r.setdefault("Probability of positive net return (%)",None)
    scenarios=normalize_saved_scenario_ranking(records)
    scores={}
    for r in scenarios.to_dict(orient="records"):
        scores.setdefault(str(r["Yahoo Ticker"]).strip().upper(),r)
    caps=((payload.get("allocation_policy") or {}).get("limits") or {}).get("position_caps") or {}
    paused={str(t).strip().upper() for t in paused}
    rows=[]
    for ticker,target in targets.items():
        holding=current.get(ticker,{})
        draft=drafts.get(ticker,{})
        weight=_number(holding.get("Weight",holding.get("Current Weight",draft.get("Current Weight"))))
        cap=_number(caps.get(ticker))
        score=scores.get(ticker,{})
        status="Cash unknown · sizing/checks required"
        if ticker in paused:
            status="Paused · research only"
        elif cap is not None and weight is not None and weight>=cap-1e-9:
            status="At/above position cap · research only"
        elif cap is not None and target>cap+1e-9:
            status="Saved target exceeds cap · resize/check"
        elif weight is not None and weight>=target-1e-9:
            status="At/above saved allocation · review with new cash"
        rows.append({"Security":ticker,"Currency":holding.get("Currency") or draft.get("Currency") or "Unknown",
            "Score (%)":_number(score.get("Return × positive-probability score (%)")),
            "Expected net return (%)":_number(score.get("Expected net scenario return (%)")),
            "Positive paths (%)":_number(score.get("Probability of positive net return (%)")),
            "Saved allocation (%)":100*target,"Current allocation (%)":None if weight is None else 100*weight,
            "Unfunded draft quantity":_number(draft.get("Executable Quantity")),
            "Price data through":score.get("Price data through") or "Unavailable","Status":status})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Score (%)","Security"],ascending=[False,True],na_position="last",kind="stable").reset_index(drop=True)


def render_buy_candidates(payload, paused=()):
    import streamlit as st
    st.markdown("**Buy candidates · independent of available cash**")
    st.caption("Saved optimal securities ranked by return × positive-path frequency. Your outside income/cash is unknown to the app. These remain visible when the execution plan is blocked; quantities below are unfunded drafts, not released orders. Enter new money under Lumpsum allocation, then recheck sizing and risk limits.")
    frame=buy_candidate_watchlist(payload,paused)
    if frame.empty:
        st.info("No positive saved allocation is available to list. Existing research is retained; candidates are not invented from missing evidence.")
        return
    visible=["Security","Currency","Score (%)","Unfunded draft quantity","Status","Price data through"]
    if frame["Unfunded draft quantity"].isna().all():
        visible.remove("Unfunded draft quantity")
    display=frame[visible].style.format({"Score (%)":"{:+.2f}%","Unfunded draft quantity":"{:.6f}"},na_rep="—")
    st.dataframe(display,hide_index=True,width="stretch",column_config={
        **{n:st.column_config.NumberColumn(format="%+.2f%%") for n in ("Score (%)","Expected net return (%)")},
        **{n:st.column_config.NumberColumn(format="%.2f%%") for n in ("Positive paths (%)","Saved allocation (%)","Current allocation (%)")},
        "Unfunded draft quantity":st.column_config.NumberColumn(format="%.6f"),
    })
