"""One buy and one sell by the existing annual-lift column, without control filters."""
import math
import pandas as pd

LIFT = "Expected Annual Return Lift (%)"
MODE = "highest-annual-lift-buy-sell"


def select_annual_lift_pair(orders):
    """Select before funding/risk/pauses; retain whole draft sizes and exact listings.

    Nonzero, finite modeled quantities, values and lift are required. Never invent
    the missing side or silently substitute a lower-ranked 'permitted' security.
    The input column is the existing historical risk-adjusted lift estimate,
    not the separate 30-session return-times-probability score or a forecast.
    """
    frame=pd.DataFrame(orders).copy()
    required={"Yahoo Ticker","Action","Executable Quantity","Executable Value",LIFT}
    if frame.empty or not required.issubset(frame):
        return frame.iloc[:0].copy()
    for col in ("Executable Quantity","Executable Value",LIFT):
        frame[col]=pd.to_numeric(frame[col],errors="coerce")
    valid=frame[["Executable Quantity","Executable Value",LIFT]].map(lambda v:pd.notna(v) and math.isfinite(v)).all(axis=1)
    frame=frame.loc[valid & frame["Executable Quantity"].gt(0) & frame["Executable Value"].gt(0)]
    frame=frame.sort_values([LIFT,"Yahoo Ticker"],ascending=[False,True],kind="stable")
    return pd.concat([frame.loc[frame["Action"].eq(side)].head(1) for side in ("Buy","Sell")],ignore_index=True)


def saved_annual_lift_pair(payload):
    if "rebalance_candidate_orders" in payload:
        return select_annual_lift_pair(payload["rebalance_candidate_orders"])
    # Display-only compatibility for older saved analyses; no solver/download.
    records=[]; seen=set()
    for field in ("two_trade_plan","proposed_rebalancing_plan","buy_candidate_orders","rebalancing_plan"):
        for row in payload.get(field) or []:
            key=(row.get("Yahoo Ticker"),row.get("Action"))
            if key not in seen:
                records.append(row); seen.add(key)
    return select_annual_lift_pair(records)


def render_two_trade_plan(payload, validation=None, paused=(), *, key="two_trade_plan"):
    import streamlit as st
    pair=saved_annual_lift_pair(payload)
    st.markdown("**Rebalance plan · highest annual lift per side**")
    st.caption("One Buy + one Sell, selected before controls. Scores use the existing historical risk-adjusted annual-lift estimate; the 30-session probability score is not used. Draft quantities are not resized for assumed outside income.")
    if pair.empty:
        st.info("No nonzero trade with a recorded annual-lift estimate is available. Update rebalance plan at saved weights; optimizer search is not needed.")
        return
    missing={"Buy","Sell"}-set(pair["Action"])
    if missing:
        st.info("No nonzero " + "/".join(sorted(missing)) + " candidate exists in the saved draft; no trade was invented.")
    paused={str(t).upper() for t in paused}
    checked=validation or {}
    replayed=checked.get("selection_mode")==MODE and checked.get("pair_validation_available",True)
    display=pair.rename(columns={LIFT:"Annual lift (pp)","Executable Quantity":"Draft quantity","Executable Value":"Draft value (INR)"})
    display["Check note"]=["Paused · still selected" if str(t).upper() in paused else
        "Checks passed" if replayed and checked.get("passed") else
        "Funding/risk conflicts flagged" if replayed else "Pair needs current sizing/checks" for t in pair["Yahoo Ticker"]]
    st.dataframe(display[["Action","Yahoo Ticker","Annual lift (pp)","Draft quantity","Draft value (INR)","Check note"]],
        hide_index=True,width="stretch",column_config={
            "Annual lift (pp)":st.column_config.NumberColumn(format="%+.2f"),
            "Draft quantity":st.column_config.NumberColumn(format="%.6f"),
            "Draft value (INR)":st.column_config.NumberColumn(format="₹%.2f")})
    st.caption("Selection stays visible despite control conflicts. A selected draft is not broker execution approval; prices, funding and restrictions are shown separately.")
    st.download_button("Download two-trade plan CSV",pair.rename(columns={"Executable Quantity":"Draft quantity","Executable Value":"Draft value (INR)"}).to_csv(index=False).encode(),
        file_name="two_trade_rebalance_plan.csv",mime="text/csv",key=key)
