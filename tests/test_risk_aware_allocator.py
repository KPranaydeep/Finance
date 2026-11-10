from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from allocation_policy_config import DEFAULT_ALLOCATION_POLICY, validate_allocation_policy
from risk_aware_allocator import (build_policy_limits, check_policy_weights, correlated_groups,
                                 current_weights, funded_target, joint_block_scenarios,
                                 scenario_cvar, solve_risk_aware_allocation)


def fixture(n=6):
    rng=np.random.default_rng(42)
    frame=pd.DataFrame(rng.normal(.0006,.012,(300,n)),columns=[f"T{i}" for i in range(n)])
    allocation=pd.DataFrame({"Yahoo Ticker":frame.columns,"Quantity":1.,
                             "Weight":[.7]+[.3/(n-1)]*(n-1)})
    policy=dict(DEFAULT_ALLOCATION_POLICY,single_name_loss_budget=.1,scenario_paths=128,
                minimum_history_sessions=60,group_weight_limit=.5)
    return frame,allocation,policy


def test_joint_samples_preserve_identical_and_inverse_moves_deterministically():
    frame,_,policy=fixture()
    frame["CLONE"]=frame["T0"]
    frame["INVERSE"]=-frame["T0"]
    samples=joint_block_scenarios(frame,policy)
    assert np.array_equal(samples,joint_block_scenarios(frame,policy))
    assert np.allclose(samples[:,0],samples[:,-2])
    assert np.allclose((1+samples[:,0])*(1+samples[:,-1]),1.)


def test_sparse_solver_enforces_shock_tail_group_and_self_funding_constraints():
    frame,allocation,policy=fixture()
    frame["T1"]=frame["T0"]
    policy["group_weight_limit"] = .25
    weights,report=solve_risk_aware_allocation(frame,allocation,[(0,.5)]*6,policy)
    assert weights is not None, report
    assert np.isclose(weights.sum(),1.)
    assert weights.max() <= .2+1e-7
    assert weights[0]+weights[1] <= .25+1e-7
    assert report["candidate"]["cvar"] <= report["limits"]["tail_limit"]+1e-7
    initial=current_weights(allocation,frame.columns)
    exposure,cost=funded_target(weights,initial,report["limits"]["cost_rate"])
    assert exposure.sum()+cost == pytest.approx(1.)
    assert cost > 0
    assert report["modeled_post_cost_capital"] == pytest.approx(exposure.sum())


def test_costs_inside_solver_reduce_turnover_instead_of_only_changing_a_label():
    frame,allocation,policy=fixture()
    # Existing mix is feasible; a small mean advantage must pay for its trades.
    allocation["Weight"] = 1/6
    frame -= frame.mean()
    frame += np.array([.001,.0001,.0001,.0001,.0001,.0001])
    free,_=solve_risk_aware_allocation(frame,allocation,[(0,.5)]*6,policy,cost_bps=0.)
    costly,report=solve_risk_aware_allocation(frame,allocation,[(0,.5)]*6,policy,cost_bps=1000.)
    assert free is not None and costly is not None, report
    assert np.abs(costly-1/6).sum() < np.abs(free-1/6).sum()


def test_insufficient_assets_or_conflicting_frozen_weights_fail_without_relaxation():
    frame,allocation,policy=fixture()
    policy["single_name_loss_budget"] = .02
    weights,report=solve_risk_aware_allocation(frame,allocation,[(0,.5)]*6,policy)
    assert weights is None and "fully invested" in report["reason"]
    assert report["status"] == "infeasible"
    assert report["stage"] == "position_and_frozen_capacity"
    policy["single_name_loss_budget"] = .1
    weights,report=solve_risk_aware_allocation(frame,allocation,[(.3,.3)]+[(0,.5)]*5,policy)
    assert weights is None and "frozen" in report["reason"]


def test_missing_owned_history_and_nonfinite_history_do_not_get_renormalized_away():
    frame,allocation,policy=fixture()
    weights,report=solve_risk_aware_allocation(frame.drop(columns="T0"),allocation,[(0,.5)]*5,policy)
    assert weights is None and "Owned history missing" in report["reason"]
    assert report["stage"] == "current_allocation_inputs"
    assert report["input_summary"]["missing_owned"] == 1
    assert report["status"] == "unavailable"
    frame.iloc[1,0] = np.nan
    weights,report=solve_risk_aware_allocation(frame,allocation,[(0,.5)]*6,policy)
    assert weights is None and "Complete chronological" in report["reason"]
    assert report["stage"] == "joint_scenario_inputs"
    assert report["input_summary"]["nonfinite_values"] == 1


def test_inverse_exposure_is_not_part_of_positive_correlation_group():
    frame,_,_=fixture()
    frame["T1"]=frame["T0"]
    frame["T2"]=-frame["T0"]
    groups=correlated_groups(frame,.9)
    assert [0,1] in groups
    assert all(2 not in group for group in groups)


def test_cvar_fractional_tail_matches_linear_program_definition():
    assert scenario_cvar([-.1,-.05,0.,.1],.625) == pytest.approx((.1+.5*.05)/1.5)
    assert scenario_cvar([.01,.02,.03],.95) == 0.


def test_saved_policy_rechecks_new_worse_shock_and_group_history():
    frame,allocation,policy=fixture()
    initial=current_weights(allocation,frame.columns)
    _,limits=build_policy_limits(frame,initial,[(0,.5)]*6,policy,22.)
    weights=np.repeat(1/6,6)
    frame.iloc[-1,0] = np.log(.2)
    result=check_policy_weights(weights,frame,limits)
    assert not result["passed"]
    assert any("shock budget" in v for v in result["violations"])


@pytest.mark.parametrize("key,value",[("scenario_paths",0),("confidence",float("nan")),
                                     ("single_name_loss_budget",-.1),("enabled",1),
                                     ("solver_time_limit_seconds",1000)])
def test_policy_inputs_are_validated(key,value):
    policy=dict(DEFAULT_ALLOCATION_POLICY)
    policy[key]=value
    with pytest.raises(ValueError,match="ALLOCATION_POLICY"):
        validate_allocation_policy(policy)


def test_core_compares_incumbent_under_same_constraints_and_keeps_inputs_intact(monkeypatch):
    import portfolio_optimizer_core as core
    frame,allocation,policy=fixture()
    before=deepcopy(allocation)
    monkeypatch.setitem(core.apply_allocation_policy.__globals__,"ALLOCATION_POLICY",policy)
    weights,report=core.apply_allocation_policy(np.array([.5,.1,.1,.1,.1,.1]),frame,allocation,max_dd=.99)
    assert weights is not None, report
    assert report["selected"] == "risk-aware"
    assert report["constraints_applied"]
    assert not report["incumbent_validation"]["passed"]
    assert report["risk-aware_validation"]["passed"]
    pd.testing.assert_frame_equal(allocation,before)


def test_timeout_is_not_an_unconstrained_success(monkeypatch):
    import risk_aware_allocator as module
    from types import SimpleNamespace
    frame,allocation,policy=fixture()
    monkeypatch.setattr(module,"linprog",lambda *a,**k:SimpleNamespace(status=1,success=False,message="Time limit"))
    weights,report=solve_risk_aware_allocation(frame,allocation,[(0,.5)]*6,policy)
    assert weights is None and report["status"] == "time_limit"


def test_model_fit_ignores_future_rows_when_fitting_a_chronological_fold():
    frame,allocation,policy=fixture()
    altered=frame.copy()
    altered.iloc[220:,0] = .3
    first,first_report=solve_risk_aware_allocation(frame.iloc[:220],allocation,[(0,.5)]*6,policy)
    second,second_report=solve_risk_aware_allocation(altered.iloc[:220],allocation,[(0,.5)]*6,policy)
    assert first is not None and np.array_equal(first,second)
    assert first_report["limits"] == second_report["limits"]


def test_incumbent_fallback_must_still_pass_the_new_policy_when_lp_times_out(monkeypatch):
    import portfolio_optimizer_core as core
    import risk_aware_allocator as module
    from types import SimpleNamespace
    frame,allocation,policy=fixture()
    allocation["Weight"]=1/6
    monkeypatch.setitem(core.apply_allocation_policy.__globals__,"ALLOCATION_POLICY",policy)
    monkeypatch.setattr(module,"linprog",lambda *a,**k:SimpleNamespace(status=1,success=False,message="Time limit"))
    weights,report=core.apply_allocation_policy(np.repeat(1/6,6),frame,allocation,max_dd=.99)
    assert weights is not None and report["selected"] == "incumbent"
    assert report["constraints_applied"] and report["incumbent_validation"]["passed"]
    weights,report=core.apply_allocation_policy(np.array([.5,.1,.1,.1,.1,.1]),frame,allocation,max_dd=.99)
    assert weights is None and report["selected"] == "none"


def test_refreshed_plan_rechecks_all_owned_and_target_weights_without_an_optimizer(monkeypatch):
    import fixed_weight_rebalance as fixed
    import portfolio_optimizer_core as core
    frame,_,policy=fixture(30)
    weights=np.repeat(1/30,30)
    payload={"saved_at":"2026-11-09T00:00:00+00:00",
             "optimal_allocation":[{"ticker":t,"target_weight":float(w)} for t,w in zip(frame.columns,weights)],
             "current_allocation":[{"Yahoo Ticker":t,"Currency":"USD","Symbol":t} for t in frame.columns],
             "settings":{"days_to_flip":1}}
    held=pd.DataFrame({"Yahoo Ticker":frame.columns,"Symbol":frame.columns,"Currency":"USD",
                       "Quantity":1.,"Average Price":10.})
    monkeypatch.setattr(fixed,"ALLOCATION_POLICY",dict(DEFAULT_ALLOCATION_POLICY, enabled=False))
    monkeypatch.setattr(core,"run_portfolio_analysis_multi",lambda *a,**k:pytest.fail("optimizer called"))
    candidate,_,plan=fixed.rebuild_saved_plan(payload,held,frame,{t:10. for t in frame.columns},{"USD":1.},
                                             plan_builder=core.rebalance_plan_multi,stats_builder=core.portfolio_stats,
                                             allocation_policy_settings=policy)
    assert candidate["optimal_allocation"] == payload["optimal_allocation"]
    assert candidate["allocation_policy"]["status"] == "saved-weights-recheck"
    assert candidate["allocation_policy"]["policy"] == policy
    assert plan.attrs["execution_validation"]["allocation_policy"]["passed"]
    assert candidate["rebalance_plan_context"]["optimizer_run"] is False


def test_execution_cannot_bypass_shock_limits_even_when_old_position_cap_passes():
    from allocation_integrity import execution_check
    frame,allocation,policy=fixture()
    _,limits=build_policy_limits(frame,current_weights(allocation,frame.columns),[(0,.5)]*6,policy,22.)
    held=pd.DataFrame({"Yahoo Ticker":frame.columns,"Currency":"USD","FX to INR":1.,
                       "Quantity":[3.]+[1.4]*5})
    checked=execution_check(held,[],{t:10. for t in frame.columns},frame,cost_bps=22.,
                            max_weight=.5,policy_limits=limits)
    assert checked["risk"]["passed"]
    assert not checked["passed"] and not checked["allocation_policy"]["passed"]
    assert any("shock budget" in v for v in checked["violations"])


def test_search_reuse_cannot_cross_policy_or_cost_changes():
    import portfolio_optimizer_core as core
    before={"optimizer_config_version":"test","snapshot_date_ist":"2026-11-09",
            "allocation_policy":{"single_name_loss_budget":.02},"transaction_cost_bps":22.}
    assert core._search_results_are_compatible(before,dict(before))
    assert not core._search_results_are_compatible(before,dict(before,transaction_cost_bps=33.))
    assert not core._search_results_are_compatible(before,dict(before,allocation_policy={"single_name_loss_budget":.01}))
