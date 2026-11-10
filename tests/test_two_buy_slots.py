from copy import deepcopy
import json
import pandas as pd
import pytest
from two_trade_rebalance import LIFT, MODE, saved_annual_lift_pair, select_annual_lift_pair


def orders():
    return [{"Yahoo Ticker":t,"Action":a,LIFT:score,"Executable Quantity":10.,"Executable Value":100.} for t,a,score in
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


def test_private_core_persists_three_rows_and_controls_do_not_change_them():
    import portfolio_optimizer_core as core
    import numpy as np
    held=pd.DataFrame({"Symbol":list("ABCDE"),"Yahoo Ticker":list("ABCDE"),"Quantity":[40.,30.,30.,0.,0.],"FX to INR":1.,"Currency":"INR"})
    lr=pd.DataFrame({t:np.full(300,mu) for t,mu in zip("ABCDE",[.004,.001,.002,.003,.0015])},index=pd.bdate_range("2025-01-01",periods=300))
    plan,_,_=core.rebalance_plan_multi(held,[.2,.1,.3,.25,.15],lr,{t:10. for t in "ABCDE"},1,two_trade_plan=True,excluded_trade_tickers=list("ABCDE"))
    restored=json.loads(json.dumps(plan.attrs["two_trade_plan"]))
    assert [r["Action"] for r in restored]==["Buy","Buy","Sell"]
    assert restored[0]["Yahoo Ticker"]!=restored[1]["Yahoo Ticker"]
    assert plan.attrs["execution_validation"]["selection_mode"]==MODE
    assert plan.attrs["execution_validation"]["selected_trade_count"]==3
    assert plan.empty  # Recorded pause conflicts are not mislabeled execution approval.


def test_ui_labels_mandatory_new_and_unrestricted_buy_and_keeps_sell():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/two_buy_app.py").run()
    assert not app.exception
    frame=app.dataframe[0].value
    assert frame["Action"].tolist()==["Buy","Buy","Sell"]
    assert frame["Selection"].tolist()==["New holding buy","Open buy","Top sell"]
