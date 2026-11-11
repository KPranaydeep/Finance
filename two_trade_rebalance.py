"""Currency-aware eligible buys (one new holding) and owned-holding sells.

Legacy module/function/payload names are retained for saved-analysis compatibility.
"""
import math
import pandas as pd

LIFT = "Expected Annual Return Lift (%)"
MODE = "annual-lift-currency-categories-v4"
CATEGORIES = ("New holding buy", "Open buy", "Top sell")
REQUIRED_CURRENCIES = ("INR", "USD")
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


def _with_currencies(orders, metadata=()):
    """Use explicit listing metadata; missing/conflicting currencies stay unknown."""
    frame=pd.DataFrame(orders).copy()
    if "Yahoo Ticker" not in frame:
        return frame
    frame["Yahoo Ticker"]=frame["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    sources=[]
    for source in (frame,pd.DataFrame(metadata)):
        if {"Yahoo Ticker","Currency"}.issubset(source):
            known=source[["Yahoo Ticker","Currency"]].copy()
            for col in known:
                known[col]=known[col].fillna("").astype(str).str.strip().str.upper()
            known=known.loc[known["Currency"].str.fullmatch("[A-Z]{3}") & ~known["Currency"].isin(["UNK","NAN"])]
            sources.append(known)
    mapping={}
    if sources:
        for ticker,rows in pd.concat(sources).groupby("Yahoo Ticker"):
            currencies=rows["Currency"].unique()
            if len(currencies)==1:
                mapping[ticker]=currencies[0]
    frame["Currency"]=frame["Yahoo Ticker"].map(mapping).fillna("")
    return frame


def select_annual_lift_pair(orders, holdings=None, *, excluded_tickers=(), currency_metadata=None):
    """Exclude active pauses, then rank before funding/risk checks.

    Nonzero, finite modeled quantities, values and lift are required. Never invent
    the missing side. Funding/risk checks do not substitute different securities.
    The input column is the existing historical risk-adjusted lift estimate,
    not the separate 30-session return-times-probability score or a forecast.
    """
    metadata=currency_metadata if currency_metadata is not None else holdings if holdings is not None else ()
    frame=_with_currencies(orders,metadata)
    required={"Yahoo Ticker","Action","Executable Quantity","Executable Value",LIFT}
    if frame.empty or not required.issubset(frame):
        return frame.iloc[:0].copy()
    for col in ("Executable Quantity","Executable Value",LIFT):
        frame[col]=pd.to_numeric(frame[col],errors="coerce")
    valid=frame[["Executable Quantity","Executable Value",LIFT]].map(lambda v:pd.notna(v) and math.isfinite(v)).all(axis=1)
    frame=frame.loc[valid & frame["Executable Quantity"].gt(0) & frame["Executable Value"].gt(0)]
    frame["Yahoo Ticker"]=frame["Yahoo Ticker"].fillna("").astype(str).str.strip().str.upper()
    excluded={str(t).strip().upper() for t in excluded_tickers}
    frame=frame.loc[frame["Yahoo Ticker"].ne("") & frame["Currency"].ne("") & ~frame["Yahoo Ticker"].isin(excluded)].sort_values([LIFT,"Yahoo Ticker"],ascending=[False,True],kind="stable")
    owned=_owned_tickers(holdings)
    selections=[]
    for currency in sorted(set(frame["Currency"])):
        group=frame.loc[frame["Currency"].eq(currency)]
        buys=group.loc[group["Action"].eq("Buy")].drop_duplicates("Yahoo Ticker")
        sells=group.loc[group["Action"].eq("Sell")].drop_duplicates("Yahoo Ticker")
        new=buys.iloc[:0].copy()
        if owned is not None:
            new=buys.loc[~buys["Yahoo Ticker"].isin(owned)].head(1).copy()
            sells=sells.loc[sells["Yahoo Ticker"].isin(owned)]
        # Each currency gets an open buy distinct from its new-holding buy.
        other=buys.loc[~buys["Yahoo Ticker"].isin(new["Yahoo Ticker"])].head(1).copy()
        sell=sells.head(1).copy()
        for selected,label in zip((new,other,sell),CATEGORIES):
            selected["Selection"]=label
            selections.append(selected)
    return pd.concat(selections,ignore_index=True) if selections else frame.iloc[:0].copy()


def saved_annual_lift_pair(payload, holdings=_SAVED_HOLDINGS, *, excluded_tickers=()):
    if holdings is _SAVED_HOLDINGS:
        holdings=payload.get("current_allocation")
    metadata=pd.concat([pd.DataFrame(payload.get("current_allocation") or []),pd.DataFrame(holdings)],ignore_index=True)
    if "rebalance_candidate_orders" in payload:
        return select_annual_lift_pair(payload["rebalance_candidate_orders"],holdings,excluded_tickers=excluded_tickers,currency_metadata=metadata)
    # Display-only compatibility for older saved analyses; no solver/download.
    records=[]; seen=set()
    for field in ("two_trade_plan","proposed_rebalancing_plan","buy_candidate_orders","rebalancing_plan"):
        for row in payload.get(field) or []:
            key=(row.get("Yahoo Ticker"),row.get("Action"))
            if key not in seen:
                records.append(row); seen.add(key)
    return select_annual_lift_pair(records,holdings,excluded_tickers=excluded_tickers,currency_metadata=metadata)


def render_two_trade_plan(payload, validation=None, paused=(), *, key="two_trade_plan",holdings=_SAVED_HOLDINGS,anchor=None):
    import streamlit as st
    paused={str(t).strip().upper() for t in paused}
    pair=saved_annual_lift_pair(payload,holdings,excluded_tickers=paused)
    if anchor:
        st.subheader("Rebalance plan · buys and sells by currency",anchor=anchor)
    else:
        st.markdown("**Rebalance plan · buys and sells by currency**")
    st.caption("For each currency: highest annual-lift new-holding buy, highest remaining buy (owned or new), and highest annual-lift owned-holding sell. INR and USD slots are tracked separately. Active pauses are excluded; draft sizes assume no outside income.")
    present=set(zip(pair["Currency"],pair["Selection"])) if not pair.empty else set()
    currencies=sorted(set(REQUIRED_CURRENCIES) | {c for c,_ in present})
    missing=[f"{currency} {label.lower()}" for currency in currencies for label in CATEGORIES if (currency,label) not in present]
    if missing:
        st.info("Unavailable slots: " + "; ".join(missing) + ". No duplicate or missing trade was invented.")
    if pair.empty:
        st.info("No eligible currency-classified trade with a recorded annual-lift estimate remains after active pauses. Missing or conflicting currencies are not guessed. Update rebalance plan at saved weights if more candidates are needed; optimizer search is not needed.")
        return
    if missing:
        if any(label.endswith("new holding buy") for label in missing) and _owned_tickers(payload.get("current_allocation") if holdings is _SAVED_HOLDINGS else holdings) is None:
            st.caption("Current holding quantities are unavailable; a new-holding candidate cannot be verified.")
    checked=validation or {}
    replayed=(checked.get("selection_mode")==MODE and checked.get("pair_validation_available",True)
              and paused=={str(t).strip().upper() for t in checked.get("excluded_tickers",())})
    display=pair.rename(columns={LIFT:"Annual lift (pp)","Executable Quantity":"Draft quantity","Executable Value":"Draft value (INR)"})
    display["Check note"]=["Checks passed" if replayed and checked.get("passed") else
        "Funding/risk conflicts flagged" if replayed else "Selection needs current sizing/checks" for t in pair["Yahoo Ticker"]]
    st.dataframe(display[["Currency","Action","Selection","Yahoo Ticker","Annual lift (pp)","Draft quantity","Draft value (INR)","Check note"]],
        hide_index=True,width="stretch",column_config={
            "Annual lift (pp)":st.column_config.NumberColumn(format="%+.2f"),
            "Draft quantity":st.column_config.NumberColumn(format="%.6f"),
            "Draft value (INR)":st.column_config.NumberColumn(format="₹%.2f")})
    st.caption("Paused securities stay out of these selections and the CSV. Remaining funding/risk conflicts are flagged; a selected draft is not broker execution approval.")
    st.download_button("Download rebalance selections CSV",pair.rename(columns={"Executable Quantity":"Draft quantity","Executable Value":"Draft value (INR)"}).to_csv(index=False).encode(),
        file_name="rebalance_selections.csv",mime="text/csv",key=key)
