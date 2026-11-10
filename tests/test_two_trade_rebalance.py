from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from two_trade_rebalance import LIFT, MODE, select_annual_lift_pair, saved_annual_lift_pair


def orders():
    return [{"Yahoo Ticker":t,"Action":side,LIFT:lift,"Executable Quantity":10.,"Executable Value":100.}
        for t,side,lift in [("BUY_A","Buy",1.),("BUY_B","Buy",4.),("SELL_A","Sell",7.),("SELL_B","Sell",8.)]]


def test_highest_each_side_not_top_two_overall_and_no_probability_or_profit_threshold():
    source=orders(); before=deepcopy(source)
    source[0]["Return × positive-probability score (%)"]=1000.
    pair=select_annual_lift_pair(source)
    assert pair["Yahoo Ticker"].tolist()==["BUY_B","SELL_B"]
    assert pair["Action"].tolist()==["Buy","Sell"]
    assert pair["Executable Value"].tolist()==[100.,100.]
    assert all(r[LIFT]==b[LIFT] for r,b in zip(source,before))


def test_ties_are_deterministic_and_negative_lifts_are_not_silently_hidden():
    source=orders()
    for r in source:r[LIFT]=-1.
    pair=select_annual_lift_pair(source)
    assert pair["Yahoo Ticker"].tolist()==["BUY_A","SELL_A"]


@pytest.mark.parametrize("bad",[0.,-1.,float("nan"),float("inf")])
def test_non_trades_are_not_invented_to_fill_two_slots(bad):
    source=orders()
    for r in source:
        if r["Action"]=="Buy":r["Executable Quantity"]=bad
    pair=select_annual_lift_pair(source)
    assert pair["Action"].tolist()==["Sell"]


def test_older_saved_draft_works_without_solver_and_prefilter_records_take_priority():
    p={"proposed_rebalancing_plan":orders(),"execution_validation":{"passed":False}}
    before=deepcopy(p)
    assert saved_annual_lift_pair(p)["Yahoo Ticker"].tolist()==["BUY_B","SELL_B"]
    assert p==before
    p["rebalance_candidate_orders"]=orders()[:2]
    assert saved_annual_lift_pair(json.loads(json.dumps(p)))["Yahoo Ticker"].tolist()==["BUY_B"]
    p["rebalance_candidate_orders"]=[]
    assert saved_annual_lift_pair(p).empty  # An explicit empty new snapshot is authoritative.


def fixture():
    names=list("ABCD")
    held=pd.DataFrame({"Symbol":names,"Yahoo Ticker":names,"Quantity":[40.,30.,30.,0.],"Currency":"INR","FX to INR":1.})
    history=pd.DataFrame({t:np.full(300,mu) for t,mu in zip(names,[.004,.001,.002,.003])},index=pd.bdate_range("2025-01-01",periods=300))
    return held,history,{t:10. for t in names}


def test_core_pair_survives_pauses_funding_caps_frozen_positions_and_profit_controls():
    import portfolio_optimizer_core as core
    held,lr,prices=fixture()
    pair_ids=None
    for kwargs in ({},{"excluded_trade_tickers":list("ABCD")},{"maximum_weight":.01},
                   {"frozen_weights":{t:.25 for t in "ABCD"}},
                   {"profit_exception_settings":{"enabled":True,"minimum_net_profit_inr":1e9}},
                   {"profit_exception_settings":{"disabled_obsolete_controls":"ignored"}}):
        plan,_,_=core.rebalance_plan_multi(held,[.2,.1,.3,.4],lr,prices,1,two_trade_plan=True,**kwargs)
        selected=plan.attrs["two_trade_plan"]
        assert len(selected)==2 and {r["Action"] for r in selected}=={"Buy","Sell"}
        ids=[r["Yahoo Ticker"] for r in selected]
        pair_ids=ids if pair_ids is None else pair_ids
        assert ids==pair_ids
        report=plan.attrs["execution_validation"]
        assert report["selection_mode"]==MODE and report["selected_trade_count"]==2
        if not report["passed"]:
            assert plan.empty and report["violations"]  # No false execution approval.


def test_controls_do_not_call_lower_ranked_substitute_or_funding_resizer(monkeypatch):
    import portfolio_optimizer_core as core
    held,lr,prices=fixture()
    ns=core.rebalance_plan_multi.__globals__
    for name in ("fund_buy_orders","review_partial_sells","evaluate_profit_cap_exception"):
        monkeypatch.setitem(ns,name,lambda *a,**k:pytest.fail("selection filtered by control"))
    plan,_,_=core.rebalance_plan_multi(held,[.2,.1,.3,.4],lr,prices,1,two_trade_plan=True,excluded_trade_tickers=["A"])
    assert len(plan.attrs["two_trade_plan"])==2


def test_ui_shows_two_selected_rows_despite_failed_caps_and_paused_buy():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/two_trade_app.py").run()
    assert not app.exception
    frame=app.dataframe[0].value
    assert len(frame)==2 and frame["Action"].tolist()==["Buy","Sell"]
    assert "Paused" in frame.iloc[0]["Check note"]
    assert not any("Plan withheld" in x.value for x in app.warning)
