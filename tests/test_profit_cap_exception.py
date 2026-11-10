from copy import deepcopy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from allocation_integrity import execution_check
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from execution_diagnostics import execution_summary_metric
from profit_cap_exception import (DEFAULT_PROFIT_EXCEPTION, evaluate_profit_cap_exception,
    load_profit_exception, save_profit_exception, validate_profit_exception)
from rebalance_review import assess_executable_rebalance
from risk_aware_allocator import build_policy_limits


def fixture():
    names=list("ABCDE")
    held=pd.DataFrame({"Yahoo Ticker":names,"Symbol":names,"Quantity":10000.,"FX to INR":1.,"Currency":"INR"})
    lr=pd.DataFrame({t:np.full(360,-.02) for t in names},index=pd.bdate_range("2025-01-01",periods=360))
    prices={t:10. for t in names}
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,group_weight_limit=1.,single_name_loss_budget=.075,scenario_paths=100)
    _,limits=build_policy_limits(lr,np.full(5,.2),[(0.,.5)]*5,policy,22.)
    orders=[{"Yahoo Ticker":t,"Action":"Sell","Executable Quantity":q,"Executable Value":q*10.} for t,q in zip(names,[2000.,1900.,3000.,1000.,100.])]
    def check(orders,**kwargs):
        return execution_check(held,orders,prices,lr,cost_bps=22.,max_weight=.5,policy_limits=limits,**kwargs)
    return held,lr,prices,orders,check


def run(settings=None,check_override=None):
    held,lr,prices,orders,check=fixture()
    return evaluate_profit_cap_exception(held,orders,prices,lr,settings or dict(DEFAULT_PROFIT_EXCEPTION,enabled=True),
        check_override or check,cost_bps=22.,cost_stress_multiple=2.)


def test_nearest_above_threshold_releases_exactly_one_sale_with_residual_caps_and_audit():
    held,lr,prices,orders,check=fixture()
    before=deepcopy((held,orders))
    released,validation,audit=run()
    assert list(released["Yahoo Ticker"])==["B"]
    assert validation["passed"] and audit["applied"]
    assert 8079 <= audit["estimated_additional_net_profit_inr"] < 9000
    assert audit["remaining_over_cap"]==list("ABCDE")
    assert len(validation["cap_exception_violations"])==5
    assert validation["allocation_policy"]["position_breaches"]
    assert validation["cash_after_costs_inr"]==pytest.approx(19000*(1-.0022))
    assert execution_summary_metric(validation,1)[1]=="Partial release"
    assert all(r["Orders"]==1 for r in audit["candidates"])
    pd.testing.assert_frame_equal(held,before[0])
    assert orders==before[1]


def test_threshold_costs_and_equality_boundary():
    held,lr,prices,orders,check=fixture()
    first=run()[2]["estimated_additional_net_profit_inr"]
    assert run(dict(DEFAULT_PROFIT_EXCEPTION,enabled=True,minimum_net_profit_inr=first))[0]["Yahoo Ticker"].tolist()==["B"]
    assert run(dict(DEFAULT_PROFIT_EXCEPTION,enabled=True,minimum_net_profit_inr=first+1))[0]["Yahoo Ticker"].tolist()==["A"]
    assert run(dict(DEFAULT_PROFIT_EXCEPTION,enabled=True,minimum_net_profit_inr=1e8))[0] is None
    stress=evaluate_profit_cap_exception(held,orders,prices,lr,dict(DEFAULT_PROFIT_EXCEPTION,enabled=True),check,cost_bps=22.,cost_stress_multiple=3.)
    assert stress[2]["estimated_additional_net_profit_inr"] < first
    assert "Post-trade ES95 (3× costs)" in stress[2]["evidence"]


@pytest.mark.parametrize("error",["Funding failed","Frozen position changed: B","Duplicate, paused or unknown order ticker: B",
    "Correlated group exceeded","Portfolio tail loss exceeded","Historical drawdown exceeded",
    "Final position exceeds its weight cap: B","Missing final holding history: B"])
def test_other_failures_never_waived(error):
    check=fixture()[-1]
    def bad(orders):
        report=check(orders)
        report["violations"].append(error)
        report["passed"]=False
        return report
    released,validation,audit=run(check_override=bad)
    assert released is validation is None and not audit["applied"]


def test_positive_return_history_sell_does_not_create_expected_incremental_profit():
    held,lr,prices,orders,check=fixture()
    lr[:]=.02
    result=evaluate_profit_cap_exception(held,orders,prices,lr,dict(DEFAULT_PROFIT_EXCEPTION,enabled=True),check,cost_bps=22.,cost_stress_multiple=2.)
    assert result[0] is None


def test_disabled_does_not_even_call_validation_and_profile_state_persists():
    assert run(DEFAULT_PROFIT_EXCEPTION,lambda _:pytest.fail("disabled review"))[0] is None
    conn=sqlite3.connect(":memory:")
    assert load_profit_exception(conn,"first")==DEFAULT_PROFIT_EXCEPTION
    settings=dict(DEFAULT_PROFIT_EXCEPTION,enabled=True,minimum_net_profit_inr=9000.)
    save_profit_exception(conn,"first",settings)
    assert load_profit_exception(conn,"first")==settings
    assert load_profit_exception(conn,"second")==DEFAULT_PROFIT_EXCEPTION


@pytest.mark.parametrize("amount",[0,-1,float("nan"),float("inf"),True,"8079"])
def test_bad_threshold_cannot_enable_release(amount):
    with pytest.raises(ValueError):
        validate_profit_exception(dict(DEFAULT_PROFIT_EXCEPTION,enabled=True,minimum_net_profit_inr=amount))


def test_partial_release_integrates_with_actual_plan_and_retains_other_drafts(monkeypatch):
    import portfolio_optimizer_core as core
    held,lr,prices,_,check=fixture()
    # Target sums to one: A/B trims would otherwise fund C/D/E buys.
    targets=[.16,.162,.3,.2,.178]
    namespace=core.rebalance_plan_multi.__globals__
    monkeypatch.setitem(namespace,"review_partial_sells",lambda *a,**k:{"allowed":True})
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,group_weight_limit=1.,single_name_loss_budget=.075,scenario_paths=100)
    _,limits=build_policy_limits(lr,np.full(5,.2),[(0.,.5)]*5,policy,22.)
    plan,_,_=core.rebalance_plan_multi(held,targets,lr,prices,1,policy_limits=limits,
        profit_exception_settings=dict(DEFAULT_PROFIT_EXCEPTION,enabled=True))
    assert plan["Yahoo Ticker"].tolist()==["B"]
    assert plan.attrs["execution_validation"]["profit_cap_exception"]["applied"]
    assert plan.attrs["proposed_rebalancing_plan"]
    assert "B" not in {o["Yahoo Ticker"] for o in plan.attrs["proposed_rebalancing_plan"]}


def test_single_sale_must_pass_its_own_sharpe_check_not_only_the_combined_plan(monkeypatch):
    import portfolio_optimizer_core as core
    held,lr,prices,_,_=fixture()
    monkeypatch.setitem(core.rebalance_plan_multi.__globals__,"review_partial_sells",
        lambda _lr,_values,orders,**kw:{"allowed":len(orders)!=1})
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,group_weight_limit=1.,single_name_loss_budget=.075,scenario_paths=100)
    _,limits=build_policy_limits(lr,np.full(5,.2),[(0.,.5)]*5,policy,22.)
    plan,_,_=core.rebalance_plan_multi(held,[.16,.162,.3,.2,.178],lr,prices,1,policy_limits=limits,
        profit_exception_settings=dict(DEFAULT_PROFIT_EXCEPTION,enabled=True))
    assert plan.empty and not plan.attrs["execution_validation"]["passed"]


@pytest.mark.parametrize("guard",["paused","frozen"])
def test_real_replay_skips_restricted_nearest_sale_and_chooses_next_qualified(guard):
    held,lr,prices,orders,check=fixture()
    restriction={"excluded_tickers":["B"]} if guard=="paused" else {"frozen_weights":{"B":.2}}
    actual=lambda orders:check(orders,**restriction)
    released,validation,_=evaluate_profit_cap_exception(held,orders,prices,lr,
        dict(DEFAULT_PROFIT_EXCEPTION,enabled=True),actual,cost_bps=22.,cost_stress_multiple=2.)
    assert released["Yahoo Ticker"].tolist()==["A"]
    assert validation["final_weights"]["B"]*(validation["marked_capital_inr"]-validation["estimated_cost_inr"])==pytest.approx(100000.)


def test_latest_chronological_reversal_blocks_sell_even_when_average_looks_good():
    held,lr,prices,orders,check=fixture()
    lr.iloc[-120:]=.005
    result=evaluate_profit_cap_exception(held,orders,prices,lr,dict(DEFAULT_PROFIT_EXCEPTION,enabled=True),check,cost_bps=22.,cost_stress_multiple=2.)
    assert result[0] is None


def test_short_or_missing_history_cannot_establish_partial_release():
    held,lr,prices,orders,check=fixture()
    for incomplete in (lr.iloc[:50],lr.drop(columns="A")):
        result=evaluate_profit_cap_exception(held,orders,prices,incomplete,dict(DEFAULT_PROFIT_EXCEPTION,enabled=True),check,cost_bps=22.,cost_stress_multiple=2.)
        assert result[0] is None


def test_ui_defaults_and_submission_do_not_compute_or_trade():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/profit_release_app.py").run()
    assert not app.exception
    assert not app.checkbox[0].value
    assert app.number_input[0].value==8079.
    app.checkbox[0].check()
    app.number_input[0].set_value(9000.)
    app.button[0].click().run()
    assert not app.exception
    assert app.json[0].value=='{"enabled": true, "minimum_net_profit_inr": 9000.0}'
    app.run()
    assert app.checkbox[0].value
