from copy import deepcopy
import io
import json

import numpy as np
import pandas as pd

from two_trade_rebalance import LIFT, MODE, CATEGORIES, select_annual_lift_pair, saved_annual_lift_pair


def fixture():
    orders=[]; holdings=[]
    for currency,scale in (("INR",1.),("USD",10.)):
        for t,action,score in (("NEW","Buy",8.),("NEXT","Buy",6.),("OWNED","Buy",10.),("EXIT","Sell",9.),("EXIT2","Sell",5.)):
            orders.append({"Yahoo Ticker":f"{currency}_{t}","Currency":currency,"Action":action,LIFT:score*scale,
                           "Executable Quantity":2.,"Executable Value":100.})
        holdings.extend({"Yahoo Ticker":f"{currency}_{t}","Currency":currency,"Quantity":.001} for t in ("OWNED","EXIT","EXIT2"))
    return orders,holdings


def assert_six_slots(selected):
    assert set(zip(selected["Currency"],selected["Selection"]))=={(c,s) for c in ("INR","USD") for s in CATEGORIES}
    assert len(selected)==6 and selected["Yahoo Ticker"].is_unique


def test_each_category_has_both_currencies_even_when_all_usd_scores_are_higher():
    orders,held=fixture();before=deepcopy(orders)
    selected=select_annual_lift_pair(orders,held)
    assert_six_slots(selected)
    assert set(selected["Yahoo Ticker"])=={f"{c}_{t}" for c in ("INR","USD") for t in ("NEW","OWNED","EXIT")}
    assert orders==before and selected["Executable Quantity"].eq(2.).all()


def test_pause_exclusion_and_exact_listing_ties_apply_inside_each_currency():
    orders,held=fixture()
    selected=select_annual_lift_pair(orders,held,excluded_tickers=["usd_new","inr_exit"])
    assert_six_slots(selected)
    assert "USD_NEXT" in set(selected["Yahoo Ticker"]) and "INR_EXIT2" in set(selected["Yahoo Ticker"])
    assert not {"USD_NEW","INR_EXIT"}&set(selected["Yahoo Ticker"])


def test_old_saved_rows_resolve_currency_from_exact_metadata_without_solver_or_mutation():
    orders,held=fixture()
    metadata=[{"Yahoo Ticker":r["Yahoo Ticker"],"Currency":r["Currency"],"Quantity":0.} for r in orders]
    payload={"rebalance_candidate_orders":[{k:v for k,v in r.items() if k!="Currency"} for r in orders],"current_allocation":metadata}
    before=deepcopy(payload)
    selected=saved_annual_lift_pair(json.loads(json.dumps(payload)),held)
    assert_six_slots(selected)
    assert payload==before
    held[0]["Currency"]="USD"  # Conflicting saved/current quote metadata cannot establish a currency.
    selected=saved_annual_lift_pair(payload,held)
    assert held[0]["Yahoo Ticker"] not in set(selected["Yahoo Ticker"])


def test_missing_currency_is_not_guessed_from_ticker_or_from_another_listing():
    orders,held=fixture()
    for r in orders:
        r.pop("Currency")
    assert select_annual_lift_pair(orders,[]).empty
    one=[dict(orders[0],**{"Yahoo Ticker":"EXAMPLE.NS"})]
    assert select_annual_lift_pair(one,[{"Yahoo Ticker":"EXAMPLE","Currency":"INR","Quantity":0.}]).empty


def test_partial_currency_pool_never_invents_second_buy_or_owned_sell():
    orders,held=fixture()
    subset=[r for r in orders if r["Yahoo Ticker"] in {"INR_NEW","USD_OWNED","USD_EXIT"}]
    selected=select_annual_lift_pair(subset,held)
    assert selected["Yahoo Ticker"].tolist()==["INR_NEW","USD_OWNED","USD_EXIT"]
    assert len(selected)==3


def test_unknown_current_ownership_cannot_establish_a_currency_sell_or_new_holding_slot():
    orders,held=fixture()
    held[0]["Quantity"]=None
    selected=select_annual_lift_pair(orders,held)
    assert set(selected["Selection"])=={"Open buy"}
    assert set(selected["Currency"])=={"INR","USD"}


def test_real_core_persists_and_checks_all_six_selections_together():
    import portfolio_optimizer_core as core
    names=[f"{c}_{t}" for c in ("INR","USD") for t in ("NEW","OWNED","EXIT")]
    held=pd.DataFrame({"Yahoo Ticker":names,"Symbol":names,"Currency":["INR"]*3+["USD"]*3,
                       "Quantity":[0.,10.,20.]*2,"FX to INR":1.})
    lr=pd.DataFrame(np.random.default_rng(22).normal(.002,.01,(300,6)),columns=names,index=pd.bdate_range("2025-01-01",periods=300))
    plan,_,_=core.rebalance_plan_multi(held,[.1,.25,.15]*2,lr,{n:10. for n in names},1,two_trade_plan=True)
    assert_six_slots(pd.DataFrame(plan.attrs["two_trade_plan"]))
    assert plan.attrs["execution_validation"]["selected_trade_count"]==6
    assert plan.attrs["execution_validation"]["selection_mode"]==MODE
    if not plan.attrs["execution_validation"]["passed"]:
        assert plan.empty


def test_render_csv_and_missing_slot_messages_agree(monkeypatch):
    import streamlit as st
    from two_trade_rebalance import render_two_trade_plan
    orders,held=fixture();shown=[];downloads=[];messages=[]
    for name in ("subheader","markdown","caption"):
        monkeypatch.setattr(st,name,lambda *a,**k:None)
    monkeypatch.setattr(st,"info",messages.append)
    monkeypatch.setattr(st,"dataframe",lambda frame,**kw:shown.append(frame.copy()))
    monkeypatch.setattr(st,"download_button",lambda label,data,**kw:downloads.append(data))
    payload={"rebalance_candidate_orders":orders,"current_allocation":held}
    render_two_trade_plan(payload,{},paused=["USD_NEW"])
    assert_six_slots(shown[0])
    csv=pd.read_csv(io.BytesIO(downloads[0]))
    assert csv["Yahoo Ticker"].tolist()==shown[0]["Yahoo Ticker"].tolist()
    assert "Currency" in shown[0] and "Draft value (INR)" in shown[0]
    payload["rebalance_candidate_orders"]=[r for r in orders if r["Currency"]=="INR"]
    render_two_trade_plan(payload,{})
    assert any(all(s in m for s in ("USD new holding buy","USD open buy","USD top sell")) for m in messages)


def test_currency_plan_streamlit_view_shows_six_eligible_rows():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/currency_rebalance_app.py").run()
    assert not app.exception
    frame=app.dataframe[0].value
    assert_six_slots(frame)
    assert "USD_NEW" not in set(frame["Yahoo Ticker"])
    assert "USD_NEXT" in set(frame["Yahoo Ticker"])
