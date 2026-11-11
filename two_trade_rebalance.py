"""Two distinct buys (one new holding) and the top sell, before control filters.

Legacy module/function/payload names are retained for saved-analysis compatibility.
"""
import math
import pandas as pd

LIFT = "Expected Annual Return Lift (%)"
MODE = "annual-lift-two-buys-one-new-plus-sell-v2"
_SAVED_HOLDINGS = object()


def _owned_tickers(holdings):
    if holdings is None:
        return None
    frame=pd.DataFrame(holdings)
    if frame.empty:
        return set()
    if not {"Yahoo Ticker","Quantity"}.issubset(frame):
        return None
    quantity=pd.to_numeric(frame["Quantity"],errors="coerce")
    if quantity.isna().any() or (quantity<0).any() or not quantity.map(math.isfinite).all():
        return None
    names=frame["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    if (names.eq("") & quantity.gt(0)).any():
        return None
    return set(names.loc[quantity.gt(0)])


def select_annual_lift_pair(orders, holdings=None):
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
    frame["Yahoo Ticker"]=frame["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    frame=frame.loc[frame["Yahoo Ticker"].ne("")].sort_values([LIFT,"Yahoo Ticker"],ascending=[False,True],kind="stable")
    buys=frame.loc[frame["Action"].eq("Buy")].drop_duplicates("Yahoo Ticker")
    sells=frame.loc[frame["Action"].eq("Sell")].drop_duplicates("Yahoo Ticker")
    owned=_owned_tickers(holdings)
    new=buys.iloc[:0].copy()
    if owned is not None:
        new=buys.loc[~buys["Yahoo Ticker"].isin(owned)].head(1).copy()
        sells=sells.loc[sells["Yahoo Ticker"].isin(owned)]
    # The open slot can be owned or another new security, but never a duplicate.
    other=buys.loc[~buys["Yahoo Ticker"].isin(new["Yahoo Ticker"])].head(1).copy()
    sell=sells.head(1).copy()
    new["Selection"],other["Selection"],sell["Selection"]="New holding buy","Open buy","Top sell"
    return pd.concat([new,other,sell],ignore_index=True)


def saved_annual_lift_pair(payload, holdings=_SAVED_HOLDINGS):
    if holdings is _SAVED_HOLDINGS:
        holdings=payload.get("current_allocation")
    if "rebalance_candidate_orders" in payload:
        return select_annual_lift_pair(payload["rebalance_candidate_orders"],holdings)
    # Display-only compatibility for older saved analyses; no solver/download.
    records=[]; seen=set()
    for field in ("two_trade_plan","proposed_rebalancing_plan","buy_candidate_orders","rebalancing_plan"):
        for row in payload.get(field) or []:
            key=(row.get("Yahoo Ticker"),row.get("Action"))
            if key not in seen:
                records.append(row); seen.add(key)
    return select_annual_lift_pair(records,holdings)


def render_two_trade_plan(payload, validation=None, paused=(), *, key="two_trade_plan",holdings=_SAVED_HOLDINGS,anchor=None):
    import streamlit as st
    pair=saved_annual_lift_pair(payload,holdings)
    if anchor:
        st.subheader("Rebalance plan · two buys + one sell",anchor=anchor)
    else:
        st.markdown("**Rebalance plan · two buys + one sell**")
    st.caption("Buy 1: highest annual-lift security not currently owned. Buy 2: highest remaining buy, owned or new. Sell: highest annual-lift sell. Selection precedes controls; draft sizes assume no outside income.")
    if pair.empty:
        st.info("No nonzero trade with a recorded annual-lift estimate is available. Update rebalance plan at saved weights; optimizer search is not needed.")
        return
    missing={"New holding buy","Open buy","Top sell"}-set(pair["Selection"])
    if missing:
        st.info("Unavailable slot: " + ", ".join(sorted(missing)) + ". No duplicate or missing trade was invented.")
        if "New holding buy" in missing and _owned_tickers(payload.get("current_allocation") if holdings is _SAVED_HOLDINGS else holdings) is None:
            st.caption("Current holding quantities are unavailable; a new-holding candidate cannot be verified.")
    paused={str(t).upper() for t in paused}
    checked=validation or {}
    replayed=checked.get("selection_mode")==MODE and checked.get("pair_validation_available",True)
    display=pair.rename(columns={LIFT:"Annual lift (pp)","Executable Quantity":"Draft quantity","Executable Value":"Draft value (INR)"})
    display["Check note"]=["Paused · still selected" if str(t).upper() in paused else
        "Checks passed" if replayed and checked.get("passed") else
        "Funding/risk conflicts flagged" if replayed else "Selection needs current sizing/checks" for t in pair["Yahoo Ticker"]]
    st.dataframe(display[["Action","Selection","Yahoo Ticker","Annual lift (pp)","Draft quantity","Draft value (INR)","Check note"]],
        hide_index=True,width="stretch",column_config={
            "Annual lift (pp)":st.column_config.NumberColumn(format="%+.2f"),
            "Draft quantity":st.column_config.NumberColumn(format="%.6f"),
            "Draft value (INR)":st.column_config.NumberColumn(format="₹%.2f")})
    st.caption("Selection stays visible despite control conflicts. A selected draft is not broker execution approval; prices, funding and restrictions are shown separately.")
    st.download_button("Download rebalance selections CSV",pair.rename(columns={"Executable Quantity":"Draft quantity","Executable Value":"Draft value (INR)"}).to_csv(index=False).encode(),
        file_name="rebalance_selections.csv",mime="text/csv",key=key)
