from copy import deepcopy
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from paused_allocation import prepare_pause_allocation, claim_pause_reoptimization, solve_frozen_sleeve_research, withhold_research_plan, FLAG, NOTE


def fixture():
    names=list("ABCDEFGHIJKL")
    rng=np.random.default_rng(74)
    lr=pd.DataFrame(rng.normal(.001,.012,(300,len(names))),columns=names,index=pd.bdate_range("2025-01-01",periods=300))
    lr["A"]=rng.normal(.0003,.035,300)
    held=pd.DataFrame({"Yahoo Ticker":names,"Quantity":[40.,60.]+[0.]*10,
                       "Weight":[.4,.6]+[0.]*10,"Currency":"INR"})
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,scenario_paths=100,
                single_name_loss_budget=.1,group_weight_limit=.25,solver_time_limit_seconds=5.)
    pauses=pd.DataFrame({"yahoo_ticker":["A","L"],"deferred_at":["2026-11-11T00:00:00Z"]*2,"expires_at":["2027-02-09T00:00:00Z"]*2})
    return lr,held,policy,pauses


def test_conditional_save_preserves_previous_snapshot_verbatim_and_owner_scoped():
    from paused_allocation import archive_before_conditional_save
    research={"pause_allocation_research":{"status":"conditional_research"}}
    conn=sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE latest_analysis(owner TEXT PRIMARY KEY,saved_at TEXT,payload_json TEXT)")
    raw='{"optimal_allocation": [{"ticker":"A", "target_weight":1.0}]}'
    conn.executemany("INSERT INTO latest_analysis VALUES(?,?,?)",[("alice","old",raw),("bob","other","{}")])
    conn.commit()
    with conn:
        archive_before_conditional_save(conn,"alice",research)
        archive_before_conditional_save(conn,"alice",research)
    assert conn.execute("SELECT owner,saved_at,payload_json FROM pause_analysis_archive").fetchall()==[("alice","old",raw)]
    with pytest.raises(RuntimeError):
        with conn:
            archive_before_conditional_save(conn,"bob",research)
            raise RuntimeError("replacement failed")
    assert conn.execute("SELECT COUNT(*) FROM pause_analysis_archive").fetchone()[0]==1
    archive_before_conditional_save(conn,"bob",{})
    assert conn.execute("SELECT COUNT(*) FROM pause_analysis_archive").fetchone()[0]==1
    conn.close()


def test_paused_owned_exposure_is_retained_while_paused_new_candidate_is_removed():
    lr,held,policy,pauses=fixture(); before=held.copy(deep=True)
    allocation,count=prepare_pause_allocation(held,pauses)
    assert count==1 and "L" not in set(allocation["Yahoo Ticker"])
    assert allocation.set_index("Yahoo Ticker").loc["A","Weight"]==.4
    assert allocation.set_index("Yahoo Ticker").loc["A",FLAG]
    assert allocation.attrs["pause_reoptimization"]
    pd.testing.assert_frame_equal(held,before)


def test_frozen_cap_conflict_gets_fresh_research_not_saved_weight_reuse_or_fake_approval():
    lr,held,policy,pauses=fixture()
    held,_=prepare_pause_allocation(held,pauses);lr=lr.drop(columns="L")
    bounds=[(.4,.4)]+[(0.,.5)]*(len(lr.columns)-1)
    weights,report=solve_frozen_sleeve_research(lr,held,bounds,policy)
    assert weights is not None and np.isclose(weights.sum(),1.)
    assert weights[0]==pytest.approx(.4) and weights[1]<=.2+1e-7
    assert weights[2:].sum()>.39
    assert report["status"]=="conditional_research" and report["new_allocation_established"]
    assert not report["constraints_applied"] and report["research_only"]
    assert report["inherited_breaches"][0]["ticker"]=="A"
    assert report["limits"]["position_caps"]["A"]==pytest.approx(.2)
    assert not report["candidate"]["passed"]
    assert report["frozen_fraction"]==pytest.approx(.4)
    assert report["tradable_fraction"]==pytest.approx(.6)


def test_conditional_lane_preserves_whole_book_tail_reference_and_reports_failed_gates(monkeypatch):
    import paused_allocation as module
    lr,held,policy,pauses=fixture();held,_=prepare_pause_allocation(held,pauses);lr=lr.drop(columns="L")
    real=module.linprog;calls=[]
    def solve(*a,**kw):
        calls.append(kw)
        if len(calls)==1:return SimpleNamespace(success=False,status=2,message="synthetic inherited risk conflict")
        return real(*a,**kw)
    monkeypatch.setattr(module,"linprog",solve)
    weights,report=module.solve_frozen_sleeve_research(lr,held,[(.4,.4)]+[(0.,.5)]*10,policy)
    assert weights is not None and weights[0]==pytest.approx(.4)
    assert report["conditional_lane"]=="minimum_tail_research"
    assert report["research_only"] and not report["constraints_applied"]
    assert len(report["conditional_tail_results"])==len(report["conditional_tail_references"])==1


def test_missing_owned_history_and_insufficient_tradable_capacity_are_not_fabricated():
    lr,held,policy,pauses=fixture();held,_=prepare_pause_allocation(held,pauses)
    weights,report=solve_frozen_sleeve_research(lr.drop(columns="A"),held,[(0.,.5)]*11,policy)
    assert weights is None and "Owned history missing" in report["reason"]
    weights,report=solve_frozen_sleeve_research(lr[["A","B"]],held,[(.4,.4),(0.,.1)],policy)
    assert weights is None and "capacity" in report["reason"]


def test_real_full_and_prepared_paths_avoid_slow_incumbents_and_reoptimize_fresh(monkeypatch):
    import portfolio_optimizer_core as core
    lr,held,policy,pauses=fixture();held,_=prepare_pause_allocation(held,pauses);lr=lr.drop(columns="L")
    ns=core.run_portfolio_analysis_multi.__globals__
    monkeypatch.setitem(ns,"ALLOCATION_POLICY",policy)
    monkeypatch.setitem(ns,"get_daily_log_returns",lambda *a,**kw:(lr,{}))
    for name in ("optimize_portfolio_max_return_given_daily_risk","optimize_max_sharpe_ratio"):
        monkeypatch.setitem(ns,name,lambda *a,**kw:pytest.fail("slow incumbent rerun"))
    weights,_,_,_,meta=core.run_portfolio_analysis_multi(tuple(lr.columns),held,exact_optimizer_asset_cap=12,max_dd=.95)
    assert weights is not None and weights[0]==pytest.approx(.4)
    assert meta["allocation_policy"]["status"]=="conditional_research"
    assert meta["price_integrity_frozen_owned_weights"]=={"A":.4}
    assert not meta["allocation_validation"]["passed"]
    weights,_,_,_,meta=core.optimize_prepared_return_matrix(lr,held,maximum_assets=12,max_dd=.95)
    assert weights is not None and meta["allocation_policy"]["new_allocation_established"]


def test_paused_unowned_target_can_get_a_strict_new_allocation_without_research_fallback(monkeypatch):
    import portfolio_optimizer_core as core
    lr,held,policy,pauses=fixture()
    held,_=prepare_pause_allocation(held,pauses.loc[pauses["yahoo_ticker"].eq("L")])
    lr=lr.drop(columns="L")
    weights,report,validation=core.solve_pause_reoptimization(lr,held,{},policy=policy,max_dd=.95)
    assert weights is not None and validation["passed"]
    assert report["status"]=="selected" and report["constraints_applied"]
    assert max(weights)<=.2+1e-7


def test_all_frozen_exposure_cannot_be_reallocated_or_fabricate_available_capital():
    lr,held,policy,pauses=fixture()
    weights,report=solve_frozen_sleeve_research(lr,held,[(.4,.4),(.6,.6)]+[(0.,.5)]*10,policy)
    assert weights is not None and np.allclose(weights[:2],[.4,.6])
    assert np.allclose(weights[2:],0.) and report["tradable_fraction"]==pytest.approx(0.)


def test_successful_solver_output_still_requires_independent_frozen_bound_validation(monkeypatch):
    import paused_allocation as module
    lr,held,policy,pauses=fixture()
    monkeypatch.setattr(module,"linprog",lambda *a,**k:SimpleNamespace(success=True,x=np.full(200,1/12)))
    weights,report=module.solve_frozen_sleeve_research(lr,held,[(.4,.4)]+[(0.,.5)]*11,policy)
    assert weights is None and "independent bound" in report["reason"]


def test_pause_change_claim_is_durable_owner_scoped_and_does_not_loop_after_new_saved_run(tmp_path):
    _,_,_,pauses=fixture()
    payload={"run_id":"old","optimal_allocation":[{"ticker":"A","target_weight":.4}]}
    path=tmp_path/"queue.sqlite"
    with sqlite3.connect(path) as conn:
        assert claim_pause_reoptimization(conn,"alice",payload,pauses)
        assert not claim_pause_reoptimization(conn,"alice",dict(payload,run_id="new"),pauses)
        assert claim_pause_reoptimization(conn,"bob",payload,pauses)
    with sqlite3.connect(path) as conn:
        assert not claim_pause_reoptimization(conn,"alice",payload,pauses)
        changed=pauses.copy();changed.loc[0,"expires_at"]="2027-02-10T00:00:00Z"
        assert claim_pause_reoptimization(conn,"alice",payload,changed)
        assert not claim_pause_reoptimization(conn,"alice",{"optimal_allocation":[{"ticker":"NOT_PAUSED","target_weight":1.}]},pauses)


def test_research_cannot_acquire_execution_approval_from_plan_refresh():
    plan=pd.DataFrame([{"Yahoo Ticker":"B","Action":"Buy","Executable Quantity":1.,"Executable Value":10.}])
    plan.attrs={"execution_validation":{"passed":True,"violations":[]},"two_trade_plan":plan.to_dict(orient="records")}
    result=withhold_research_plan(plan)
    assert result.empty and not result.attrs["execution_validation"]["passed"]
    assert result.attrs["two_trade_plan"] and result.attrs["proposed_rebalancing_plan"]
    assert plan.attrs["execution_validation"]["passed"]


def test_conditional_timer_reports_new_research_instead_of_saved_weight_fallback(monkeypatch):
    import portfolio_optimizer_core as core
    from optimization_run_timer import start_run_timer
    ns=core.run_portfolio_analysis_multi.__globals__;state={ns["OPTIMIZATION_TIMER_KEY"]:start_run_timer()}
    monkeypatch.setitem(ns,"st",SimpleNamespace(session_state=state))
    timer=ns["_finish_optimization_plan_timer"](completion_note=NOTE,completion_kind="conditional_optimization")
    kind,message=ns["_timer_summary"](timer)
    assert kind=="conditional_optimization" and "Fresh tradable" in message
    assert "saved weights retained" not in message and "Run stopped" not in message


def test_public_publisher_rejects_conditional_analysis_before_reading_constituents():
    import ast,json
    from pathlib import Path
    tree=ast.parse(Path("pages/02_Public_Basket_Publisher.py").read_text(encoding="utf-8"))
    block=next(n for n in tree.body if isinstance(n,ast.Try) and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="source" for t in x.targets) for x in n.body))
    errors=[]
    class Stop(Exception):pass
    def stop():raise Stop()
    uploaded=SimpleNamespace(getvalue=lambda:json.dumps({"pause_allocation_research":{"status":"conditional_research"},"optimal_allocation":[{"ticker":"A","target_weight":1.}]}).encode())
    with pytest.raises(Stop):
        exec(compile(ast.Module(body=[block],type_ignores=[]),"<publisher-import>","exec"),{"json":json,"uploaded":uploaded,"st":SimpleNamespace(error=errors.append,stop=stop)})
    assert "not an approved public allocation" in errors[0]


def test_automatic_pause_action_uses_raw_peak_before_widgets_without_cap_search():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/raw_peak_ui_fixture.py",default_timeout=10)
    app.session_state["fixture_auto_pause"]=True
    app.run()
    assert not app.exception
    assert app.number_input(key="universal_preselection_cap").value==8350
    assert app.number_input(key="exact_optimizer_asset_cap").value==550
    assert any("Paused securities intersect" in message.value for message in app.info)
