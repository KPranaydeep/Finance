import numpy as np
import pandas as pd
import pytest

from adaptive_allocation import adaptive_state, scaled_risk_scenarios, conditional_objective_means
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY, validate_allocation_policy
from risk_aware_allocator import (build_policy_limits, check_policy_weights, joint_block_scenarios,
                                 solve_risk_aware_allocation, funded_target, scenario_cvar)


def history(n=30, seed=21):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(.0005, .01, (400,n)),
                         index=pd.bdate_range("2024-01-01", periods=400), columns=[f"T{i}" for i in range(n)])
    allocation = pd.DataFrame({"Yahoo Ticker": frame.columns, "Quantity": 1.,
                               "Weight": [.7] + [.3/(n-1)]*(n-1)})
    return frame, allocation, dict(DEFAULT_ALLOCATION_POLICY, scenario_paths=128)


def test_forecasts_are_causal_and_restart_deterministic():
    frame, _, _ = history()
    altered = frame.copy()
    altered.iloc[250:] = .5
    first = adaptive_state(frame.iloc[:250])
    second = adaptive_state(altered.iloc[:250])
    for key in ("factors", "mean_shift", "recent_covariance"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["summary"] == second["summary"]
    assert first["summary"]["anchor_share"] >= .5
    assert first["summary"]["recent_effective_observations"] < 13
    assert first["summary"]["feedback_observations"] == 224


@pytest.mark.parametrize("rho,winning", [(.95,"trend_share"),(-.95,"reversal_share")])
def test_trend_and_reversal_earn_influence_from_past_prediction_errors(rho, winning):
    rng = np.random.default_rng(4)
    r = np.zeros(600)
    for t in range(1,len(r)):
        r[t] = rho*r[t-1] + rng.normal(0,.003)
    state = adaptive_state(pd.DataFrame({"A":r}))
    losing = "reversal_share" if winning == "trend_share" else "trend_share"
    assert state["summary"][winning] > state["summary"][losing]
    assert state["summary"]["anchor_share"] >= .5


def test_newest_move_tightens_risk_and_release_is_gradual_without_mutating_history():
    frame, _, _ = history()
    frame.iloc[-1,0] = -.2
    original = frame.copy(deep=True)
    now = adaptive_state(frame)
    assert now["factors"][0] > 1.1
    calm = pd.concat([frame, pd.DataFrame(np.zeros((1,30)), index=[frame.index[-1]+pd.offsets.BDay()], columns=frame.columns)])
    after = adaptive_state(calm)
    assert after["factors"][0] >= 1 + np.exp(-1/13)*(now["factors"][0]-1) - 1e-12
    pd.testing.assert_frame_equal(frame, original)


def test_a_persistent_smooth_decline_is_not_mistaken_for_zero_recent_risk():
    frame, _, _ = history()
    frame.iloc[-13:,0] = -.04
    assert adaptive_state(frame)["factors"][0] > 1.1


def test_both_joint_tail_books_and_hard_position_limits_are_enforced():
    frame, allocation, policy = history()
    weights, report = solve_risk_aware_allocation(frame, allocation, [(0,.5)]*30, policy)
    assert weights is not None, report
    assert report["adaptive"]["method"]
    limits = report["limits"]
    assert max(limits["position_caps"].values()) <= .04
    initial = np.array([.7]+[.3/29]*29)
    exposure, cost = funded_target(weights, initial, limits["cost_rate"])
    raw = joint_block_scenarios(frame, policy)
    assert scenario_cvar(raw @ exposure-cost, .95) <= limits["tail_limit"] + 1e-7
    assert report["candidate"]["cvar"] <= limits["tail_limit"] + 1e-7
    # Supplied optimistic scenarios cannot override the adaptive final recheck.
    bad = np.zeros(30)
    bad[0] = 1.
    checked = check_policy_weights(bad, frame, limits, scenarios=np.zeros_like(raw))
    assert not checked["passed"]


def test_scaling_does_not_erase_ruin_or_invent_return_from_volatility():
    raw = np.array([[-1., -.2],[.2,.2],[.3,.1]])
    state = {"long_mean": np.array([.001,.001]), "factors":np.array([2.,3.]), "mean_shift":np.zeros(2)}
    scaled = scaled_risk_scenarios(raw, state, 30)
    assert scaled[0,0] == -1.
    assert scaled[0,1] < raw[0,1]
    np.testing.assert_allclose(conditional_objective_means(raw,state,30), raw.mean(axis=0))


def test_adaptive_risk_can_withhold_instead_of_relaxing_to_fit():
    frame, allocation, policy = history(n=6)
    weights, report = solve_risk_aware_allocation(frame, allocation, [(0,.5)]*6, policy)
    assert weights is None and report["status"] == "infeasible"
    assert report["policy"]["single_name_loss_budget"] == .02


def test_legacy_policy_remains_manual_and_bad_auto_flag_is_rejected():
    old = dict(DEFAULT_ALLOCATION_POLICY)
    old.pop("adaptive_enabled")
    assert validate_allocation_policy(old)["adaptive_enabled"] is False
    with pytest.raises(ValueError, match="ADAPTIVE_ENABLED"):
        validate_allocation_policy(dict(DEFAULT_ALLOCATION_POLICY, adaptive_enabled="true"))


def test_manual_override_keeps_original_limits_and_scenarios():
    frame, allocation, policy = history()
    initial = allocation.Weight.to_numpy()
    samples, limits = build_policy_limits(frame, initial, [(0,.5)]*30,
                                          dict(policy, adaptive_enabled=False), 22.)
    np.testing.assert_array_equal(samples, joint_block_scenarios(frame, policy))
    assert limits["adaptive"] == {"enabled":False}
    assert set(limits["position_caps"].values()) == {.04}


def test_adaptation_preserves_clone_and_inverse_joint_moves():
    frame, _, policy = history()
    x = frame.iloc[:,0]
    pair = pd.DataFrame({"A":x, "CLONE":x, "INVERSE":-x})
    state = adaptive_state(pair)
    raw = joint_block_scenarios(pair, policy)
    scaled = scaled_risk_scenarios(raw, state, 30)
    np.testing.assert_allclose(scaled[:,0], scaled[:,1])
    np.testing.assert_allclose((1+scaled[:,0])*(1+scaled[:,2]), 1., atol=1e-12)
    _, limits = build_policy_limits(pair, np.array([.4,.4,.2]), [(0,.5)]*3, policy, 22.)
    assert any(set(g["tickers"]) == {"A","CLONE"} for g in limits["groups"])
    assert all(not ({"A","INVERSE"} <= set(g["tickers"])) for g in limits["groups"])


def test_no_predictive_signal_keeps_trend_and_reversal_balanced():
    state = adaptive_state(pd.DataFrame(np.full((200,3), .001)))
    assert state["summary"]["trend_share"] == pytest.approx(state["summary"]["reversal_share"])
    np.testing.assert_allclose(state["mean_shift"], 0., atol=1e-14)


def test_separate_raw_tail_constraint_survives_an_optimistic_adaptive_book(monkeypatch):
    import risk_aware_allocator as allocator
    rng = np.random.default_rng(99)
    frame = pd.DataFrame(rng.normal([.003,.0002,.0002,.0002], [.04,.003,.003,.003], (400,4)), columns=list("ABCD"))
    held = pd.DataFrame({"Yahoo Ticker": frame.columns, "Quantity":1., "Weight":.25})
    policy = dict(DEFAULT_ALLOCATION_POLICY, single_name_loss_budget=1., group_weight_limit=1., scenario_paths=128)
    monkeypatch.setattr(allocator, "scaled_risk_scenarios", lambda raw, state, horizon: np.zeros_like(raw))
    weights, report = allocator.solve_risk_aware_allocation(frame, held, [(0,1)]*4, policy, cost_bps=0.)
    assert weights is not None, report
    assert weights[0] < .4
    assert report["candidate"]["historical_cvar"] <= report["limits"]["tail_limit"] + 1e-7
