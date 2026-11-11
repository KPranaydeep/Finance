from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from allocation_conflict_recovery import RECOVERY_NOTE, recover_saved_allocation_review


def report():
    return {"status":"infeasible","reason":"Frozen position exceeds cap","capacity":{"conflicts":[
        {"ticker":"A","kind":"Frozen holding","excess_weight":.2}]}}


def test_recovery_uses_saved_targets_and_propagates_every_new_frozen_name_without_optimizer():
    saved={"run_id":"original","saved_at":"2026-11-09T00:00:00+00:00","optimal_allocation":[{"ticker":"A","target_weight":1.}]}
    before=deepcopy(saved); calls=[]
    def refresh(p,owner,**kwargs):
        calls.append((p,owner,kwargs))
        return dict(p,allocation_recovery=kwargs["allocation_recovery"])
    result=recover_saved_allocation_review(report(),"synthetic",load=lambda _:saved,refresh=refresh,frozen_tickers=["B"])
    assert result["status"]=="completed_with_conflicts"
    assert calls[0][0]==before and calls[0][2]["allocation_recovery"]["frozen_tickers"]==["A","B"]
    assert not result["recovery"]["new_allocation_established"]
    assert result["analysis"]["run_id"]=="original" and saved==before


def test_manual_continuation_recovers_nonconflicting_freezes_from_recorded_bounds_too():
    r=report()
    r["limits"]={"position_lower_bounds":{"A":.4,"B":.1,"C":0.},"position_upper_bounds":{"A":.4,"B":.1,"C":.5}}
    saved={"optimal_allocation":[{"ticker":"A","target_weight":1.}]}
    result=recover_saved_allocation_review(r,"test",load=lambda _:saved,refresh=lambda p,o,**k:dict(p,allocation_recovery=k["allocation_recovery"]))
    assert result["recovery"]["frozen_tickers"]==["A","B"]


@pytest.mark.parametrize("status",["unavailable","time_limit","disabled","selected"])
def test_missing_data_or_other_outcomes_are_not_relabelled_as_recovered(status):
    result=recover_saved_allocation_review({"status":status},"test",load=lambda _:pytest.fail("unexpected lookup"),refresh=lambda *a,**k:pytest.fail("unexpected refresh"))
    assert result["status"]=="not_applicable"


def test_no_saved_allocation_cannot_be_invented_and_failed_refresh_remains_failed():
    result=recover_saved_allocation_review(report(),"test",load=lambda _:None,refresh=lambda *a,**k:pytest.fail("no saved weights"))
    assert result["status"]=="unavailable"
    saved={"optimal_allocation":[{"ticker":"A","target_weight":1.}]};before=deepcopy(saved)
    def fail(*a,**k):raise ValueError("Missing price/FX evidence")
    result=recover_saved_allocation_review(report(),"test",load=lambda _:saved,refresh=fail)
    assert result["status"]=="refresh_failed" and "Missing price/FX" in result["reason"]
    assert saved==before


def fixture():
    rng=np.random.default_rng(32)
    lr=pd.DataFrame(rng.normal(.0005,.01,(300,6)),columns=list("ABCDEF"))
    allocation=pd.DataFrame({"Yahoo Ticker":list("ABCDEF"),"Quantity":1.,"Weight":[.4]+[.12]*5,"Currency":"INR"})
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,scenario_paths=100,single_name_loss_budget=.1,group_weight_limit=1.)
    return lr,allocation,policy


def test_capacity_preflight_is_canonical_non_permissive_and_only_runs_for_frozen_positions(monkeypatch):
    import portfolio_optimizer_core as core
    lr,allocation,policy=fixture()
    ns=core.run_portfolio_analysis_multi.__globals__
    result=ns["frozen_policy_preflight"](lr,allocation,{"A":.4},policy=policy)
    assert result["preflight"] and result["status"]=="infeasible"
    assert result["capacity"]["frozen_conflict_count"]==1
    assert result["limits"]["position_lower_bounds"]["A"]==.4
    assert result["limits"]["position_caps"]["A"]==pytest.approx(.2)
    assert ns["frozen_policy_preflight"](lr,allocation,{"A":.1},policy=policy) is None
    monkeypatch.setitem(ns,"build_policy_limits",lambda *a,**k:pytest.fail("unnecessary preflight"))
    assert ns["frozen_policy_preflight"](lr,allocation,{},policy=policy) is None
    assert ns["frozen_policy_preflight"](lr,allocation,{"A":.4},policy=dict(policy,enabled=False)) is None


def test_frozen_preflight_skips_all_slow_incumbent_solves_in_prepared_path(monkeypatch):
    import portfolio_optimizer_core as core
    lr,allocation,policy=fixture()
    ns=core.run_portfolio_analysis_multi.__globals__
    integrity=pd.DataFrame({"Ticker":["A"],"Owned":[True],"Status":["QUARANTINED_TEST"]})
    for name in ("optimize_portfolio_max_return_given_daily_risk","optimize_max_sharpe_ratio","solve_risk_aware_allocation"):
        monkeypatch.setitem(ns,name,lambda *a,**k:pytest.fail("slow impossible solve"))
    weights,_,_,_,meta=core.optimize_prepared_return_matrix(lr,allocation,maximum_assets=6,max_dd=.5,target_volatility=None,
        integrity_report=integrity,allocation_policy=policy)
    assert weights is None and meta["allocation_policy"]["preflight"]
    assert meta["frozen_weights"]=={"A":.4}


def test_frozen_preflight_skips_all_slow_incumbent_solves_in_full_analysis(monkeypatch):
    import portfolio_optimizer_core as core
    lr,allocation,policy=fixture()
    ns=core.run_portfolio_analysis_multi.__globals__
    monkeypatch.setitem(ns,"ALLOCATION_POLICY",policy)
    monkeypatch.setitem(ns,"get_daily_log_returns",lambda *a,**k:(lr,{"price_integrity_report":pd.DataFrame({"Ticker":["A"],"Owned":[True],"Status":["QUARANTINED_TEST"]})}))
    for name in ("optimize_portfolio_max_return_given_daily_risk","optimize_max_sharpe_ratio","solve_risk_aware_allocation"):
        monkeypatch.setitem(ns,name,lambda *a,**k:pytest.fail("slow impossible solve"))
    weights,_,_,_,meta=core.run_portfolio_analysis_multi(tuple(lr.columns),allocation,exact_optimizer_asset_cap=6)
    assert weights is None and meta["allocation_policy"]["preflight"]
    assert meta["price_integrity_frozen_owned_weights"]=={"A":.4}


def test_recovered_timer_is_completed_with_conflicts_not_stopped_or_new_allocation(monkeypatch):
    import portfolio_optimizer_core as core
    from optimization_run_timer import start_run_timer,serializable_timer
    ns=core.run_portfolio_analysis_multi.__globals__
    state={ns["OPTIMIZATION_TIMER_KEY"]:start_run_timer()}
    monkeypatch.setitem(ns,"st",SimpleNamespace(session_state=state))
    finished=ns["_finish_optimization_plan_timer"](completion_note=RECOVERY_NOTE)
    status,text=ns["_timer_summary"](finished)
    assert status=="completed_with_conflicts" and "Run stopped" not in text
    assert "saved weights" in text and serializable_timer(finished)["completion_note"]==RECOVERY_NOTE


def test_actual_continuation_handler_finishes_prior_stopped_run_and_renders_retained_review(monkeypatch):
    import portfolio_optimizer_core as core
    ns=core.run_portfolio_analysis_multi.__globals__
    state={ns["OPTIMIZATION_TIMER_KEY"]:{"status":"stopped_without_plan"},"approved_publication_candidate":{"old":True}}
    messages=[]; rendered=[]; refreshes=[]
    monkeypatch.setitem(ns,"st",SimpleNamespace(session_state=state,warning=messages.append))
    monkeypatch.setitem(ns,"_app_run_log",lambda *a,**k:None)
    saved={"optimal_allocation":[{"ticker":"A","target_weight":1.}],"saved_at":"2026-11-09"}
    monkeypatch.setitem(ns,"load_latest_analysis",lambda _:saved)
    def refresh(p,o,**kw):
        refreshes.append(kw)
        return dict(p,allocation_recovery=kw["allocation_recovery"])
    monkeypatch.setitem(ns,"update_saved_rebalance_plan",refresh)
    monkeypatch.setitem(ns,"render_saved_analysis",lambda *a,**k:rendered.append(k))
    done=ns["continue_saved_allocation_analysis"](report(),"test",SimpleNamespace(info=messages.append),object())
    assert done and refreshes and rendered
    assert state[ns["OPTIMIZATION_TIMER_KEY"]]["status"]=="finished"
    assert any("Analysis completed with allocation conflict" in m for m in messages)
    assert not any("Run stopped" in m for m in messages)
    assert "approved_publication_candidate" not in state


def test_recovery_ui_explicitly_preserves_allocation_provenance():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/fixtures/allocation_recovery_app.py").run()
    assert not app.exception
    assert "No new optimized allocation" in app.info[0].value
    assert "not relaxed" in app.caption[0].value
