from copy import deepcopy
import json
import pandas as pd
import pytest
from two_trade_rebalance import LIFT, MODE, saved_annual_lift_pair, select_annual_lift_pair


def orders():
    return [{"Yahoo Ticker":t,"Currency":"INR","Action":a,LIFT:score,"Executable Quantity":10.,"Executable Value":100.} for t,a,score in
        [("OWNED","Buy",10.),("NEW_A","Buy",8.),("NEW_B","Buy",7.),("EXIT","Sell",9.)]]


def holdings():
    return [{"Yahoo Ticker":t,"Quantity":q} for t,q in [("OWNED",.000001),("NEW_A",0.),("EXIT",2.)]]


def test_one_mandatory_new_then_best_remaining_can_be_owned_without_duplicates():
    source=orders(); before=deepcopy(source)
    selected=select_annual_lift_pair(source,holdings())
    assert selected["Yahoo Ticker"].tolist()==["NEW_A","OWNED","EXIT"]
    assert selected["Selection"].tolist()==["New holding buy","Open buy","Top sell"]
    assert source==before


def test_both_buys_may_be_new_and_same_ticker_cannot_fill_both_slots():
    source=orders()
    source[0][LIFT]=6.
    source.append(dict(source[1],**{"Yahoo Ticker":" new_a ",LIFT:7.5}))
    selected=select_annual_lift_pair(source,holdings())
    assert selected["Yahoo Ticker"].tolist()==["NEW_A","NEW_B","EXIT"]
    assert selected.loc[selected["Action"].eq("Buy"),"Yahoo Ticker"].is_unique


def test_tiny_fractional_holding_is_owned_even_if_display_weight_is_zero():
    held=holdings();held[0]["Weight"]=0.
    selected=select_annual_lift_pair(orders(),held)
    assert selected.iloc[0]["Yahoo Ticker"]=="NEW_A"


def test_current_holdings_override_saved_ownership_after_a_purchase_without_optimizer():
    p={"rebalance_candidate_orders":orders(),"current_allocation":holdings()}
    current=holdings()+[{"Yahoo Ticker":"NEW_A","Quantity":1.}]
    result=saved_annual_lift_pair(p,current)
    assert result["Yahoo Ticker"].tolist()==["NEW_B","OWNED","EXIT"]
    assert saved_annual_lift_pair(p).iloc[0]["Yahoo Ticker"]=="NEW_A"


def test_no_new_candidate_or_no_second_buy_is_explicit_not_duplicated():
    held=[{"Yahoo Ticker":r["Yahoo Ticker"],"Quantity":1.} for r in orders()]
    selected=select_annual_lift_pair(orders(),held)
    assert selected["Selection"].tolist()==["Open buy","Top sell"]
    one=[r for r in orders() if r["Yahoo Ticker"] in {"NEW_A","EXIT"}]
    selected=select_annual_lift_pair(one,holdings())
    assert selected["Selection"].tolist()==["New holding buy","Top sell"]


@pytest.mark.parametrize("bad",[None,-1.,float("nan"),float("inf")])
def test_unknown_or_invalid_quantities_never_falsely_label_a_buy_as_new(bad):
    held=holdings(); held[0]["Quantity"]=bad
    selected=select_annual_lift_pair(orders(),held)
    assert "New holding buy" not in set(selected["Selection"])


def test_exchange_qualified_listings_remain_distinct_and_unowned_sells_are_not_shown():
    source=orders()
    source[0]["Yahoo Ticker"]="SBC.NS"
    source[1]["Yahoo Ticker"]="SBC"
    selected=select_annual_lift_pair(source,[{"Yahoo Ticker":"SBC.NS","Quantity":5.}])
    assert selected["Yahoo Ticker"].tolist()==["SBC","SBC.NS"]
    assert "Sell" not in set(selected["Action"])


def test_private_core_persists_three_eligible_rows_after_pause_exclusion():
    import portfolio_optimizer_core as core
    import numpy as np
    held=pd.DataFrame({"Symbol":list("ABCDEF"),"Yahoo Ticker":list("ABCDEF"),"Quantity":[40.,30.,30.,0.,0.,0.],"FX to INR":1.,"Currency":"INR"})
    lr=pd.DataFrame({t:np.full(300,mu) for t,mu in zip("ABCDEF",[.004,.001,.002,.003,.0015,.001])},index=pd.bdate_range("2025-01-01",periods=300))
    plan,_,_=core.rebalance_plan_multi(held,[.2,.1,.25,.2,.15,.1],lr,{t:10. for t in "ABCDEF"},1,two_trade_plan=True,excluded_trade_tickers=["D"])
    restored=json.loads(json.dumps(plan.attrs["two_trade_plan"]))
    assert [r["Action"] for r in restored]==["Buy","Buy","Sell"]
    assert restored[0]["Yahoo Ticker"]!=restored[1]["Yahoo Ticker"]
    assert "D" not in {r["Yahoo Ticker"] for r in restored}
    assert plan.attrs["execution_validation"]["selection_mode"]==MODE
    assert plan.attrs["execution_validation"]["selected_trade_count"]==3
    if not plan.attrs["execution_validation"]["passed"]:
        assert plan.empty


def test_ui_labels_mandatory_new_and_unrestricted_buy_and_keeps_sell():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/two_buy_app.py").run()
    assert not app.exception
    frame=app.dataframe[0].value
    assert frame["Action"].tolist()==["Buy","Buy","Sell"]
    assert frame["Selection"].tolist()==["New holding buy","Open buy","Top sell"]
    assert frame["Yahoo Ticker"].tolist()==["NEW2","OWNED","EXIT"]


def test_pauses_replace_all_three_slots_without_duplicates_and_keep_exchange_listings_distinct():
    source=orders()+[
        {"Yahoo Ticker":t,"Currency":"INR","Action":side,LIFT:score,"Executable Quantity":1.,"Executable Value":10.}
        for t,side,score in [("NEW_C","Buy",6.),("EXIT2","Sell",5.),("NEW_A.NS","Buy",4.)]]
    held=holdings()+[{"Yahoo Ticker":"EXIT2","Quantity":1.}]
    before=deepcopy(source)
    selected=select_annual_lift_pair(source,held,excluded_tickers=[" new_a ","owned","exit"])
    assert selected["Yahoo Ticker"].tolist()==["NEW_B","NEW_C","EXIT2"]
    assert selected["Yahoo Ticker"].is_unique and source==before
    selected=select_annual_lift_pair(source,held,excluded_tickers=["NEW_A","NEW_B","NEW_C","OWNED"])
    assert selected.iloc[0]["Yahoo Ticker"]=="NEW_A.NS"


def test_all_paused_leaves_no_selection_or_executable_order():
    import portfolio_optimizer_core as core
    from test_two_trade_rebalance import fixture
    held,lr,prices=fixture()
    plan,_,_=core.rebalance_plan_multi(held,[.2,.1,.3,.4],lr,prices,1,two_trade_plan=True,excluded_trade_tickers=list("ABCD"))
    assert plan.empty and not plan.attrs["two_trade_plan"]
    assert not plan.attrs["proposed_rebalancing_plan"]


def test_saved_pool_reselects_on_restart_and_pause_expiry_without_mutating_payload(tmp_path):
    import sqlite3
    from trade_deferrals import defer_trades_for_quarter,active_trade_deferrals
    path=tmp_path/"pauses.db"
    payload={"rebalance_candidate_orders":orders(),"current_allocation":holdings()}
    before=deepcopy(payload)
    with sqlite3.connect(path) as conn:
        conn.row_factory=sqlite3.Row
        defer_trades_for_quarter(conn,"alice",[{"Yahoo Ticker":"NEW_A"},{"Yahoo Ticker":"EXIT"}],now="2026-11-11T00:00:00Z")
    with sqlite3.connect(path) as conn:
        conn.row_factory=sqlite3.Row
        pauses=active_trade_deferrals(conn,"alice",now="2026-11-12T00:00:00Z")
        selected=saved_annual_lift_pair(json.loads(json.dumps(payload)),excluded_tickers=pauses["yahoo_ticker"])
        assert selected["Yahoo Ticker"].tolist()==["NEW_B","OWNED"]
        assert active_trade_deferrals(conn,"bob",now="2026-11-12T00:00:00Z").empty
        expired=active_trade_deferrals(conn,"alice",now="2027-02-09T00:00:00Z")
        assert expired.empty
        assert saved_annual_lift_pair(payload,excluded_tickers=expired["yahoo_ticker"])["Yahoo Ticker"].tolist()==["NEW_A","OWNED","EXIT"]
    assert payload==before


def test_render_and_csv_use_identical_eligible_picks_and_never_reuse_old_check_approval(monkeypatch):
    import io
    import streamlit as st
    from two_trade_rebalance import render_two_trade_plan
    shown=[]; downloads=[]
    for name in ("subheader","markdown","caption","info"):
        monkeypatch.setattr(st,name,lambda *a,**k:None)
    monkeypatch.setattr(st,"dataframe",lambda frame,**kw:shown.append(frame.copy()))
    monkeypatch.setattr(st,"download_button",lambda label,data,**kw:downloads.append(data))
    payload={"rebalance_candidate_orders":orders(),"current_allocation":holdings()}
    render_two_trade_plan(payload,{"passed":True,"selection_mode":MODE,"excluded_tickers":[]},paused=["NEW_A"])
    csv=pd.read_csv(io.BytesIO(downloads[0]))
    assert shown[0]["Yahoo Ticker"].tolist()==csv["Yahoo Ticker"].tolist()==["NEW_B","OWNED","EXIT"]
    assert not shown[0]["Check note"].eq("Checks passed").any()
    assert "NEW_A" not in set(csv["Yahoo Ticker"])
