"""Frozen bounds must intersect an adaptive risk cap, never be scaled by it."""
from copy import deepcopy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from allocation_attempts import capacity_log_summary, load_workspace, save_attempt
from allocation_integrity import allocation_check
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from risk_aware_allocator import build_policy_limits, solve_risk_aware_allocation


def evidence():
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(rng.normal(.0005, .01, (400, 30)), columns=[f"T{i}" for i in range(30)])
    frame.iloc[-1, 0] = -.06
    held = pd.DataFrame({"Yahoo Ticker": frame.columns, "Quantity": 1.,
                         "Weight": [.01, .7] + [.29/28]*28})
    return frame, held, dict(DEFAULT_ALLOCATION_POLICY, single_name_loss_budget=.03, scenario_paths=128)


def test_stressed_frozen_holding_below_true_cap_remains_fixed_and_solves():
    frame, held, policy = evidence()
    bounds = [(.01, .01)] + [(0., .5)]*29
    original_frame, original_held, original_bounds = frame.copy(), held.copy(), deepcopy(bounds)
    weights, report = solve_risk_aware_allocation(frame, held, bounds, policy)
    assert weights is not None, report
    limits = report["limits"]
    assert limits["adaptive_stress_factors"]["T0"] > 1.1  # Old code incorrectly made cap < .01.
    assert limits["policy_position_caps"]["T0"] > .01
    assert limits["position_caps"]["T0"] == .01
    assert weights[0] == pytest.approx(.01, abs=1e-9)
    assert report["capacity"]["passed"]
    assert report["capacity"]["frozen_count"] == 1
    assert report["candidate"]["passed"]
    assert allocation_check(weights, frame, bounds=bounds, policy_limits=limits)["passed"]
    altered = weights.copy()
    altered[0] -= .001
    assert not allocation_check(altered, frame, bounds=bounds, policy_limits=limits)["passed"]
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_frame_equal(held, original_held)
    assert bounds == original_bounds


def test_genuine_frozen_overweight_is_numeric_and_persists_without_loosening(monkeypatch):
    import risk_aware_allocator as allocator
    frame, held, policy = evidence()
    monkeypatch.setattr(allocator, "linprog", lambda *a, **k: pytest.fail("Capacity failure must not start LP"))
    weights, report = solve_risk_aware_allocation(frame, held, [(.1, .1)]+[(0., .5)]*29, policy)
    assert weights is None and report["status"] == "infeasible"
    assert report["stage"] == "position_and_frozen_capacity"
    capacity = report["capacity"]
    assert not capacity["passed"] and capacity["conflict_count"] == capacity["frozen_conflict_count"] == 1
    row = capacity["conflicts"][0]
    assert row["ticker"] == "T0" and row["kind"] == "Frozen holding"
    assert row["minimum_weight"] == .1
    assert row["policy_cap"] == pytest.approx(policy["single_name_loss_budget"] / row["shock"] / row["stress_factor"])
    assert row["excess_weight"] == pytest.approx(.1 - row["policy_cap"])
    assert capacity["position_cap_sum"] > 1.  # It is a fixed-position conflict, not a capacity shortage.
    assert capacity["capacity_shortfall"] == 0
    assert "1 frozen" in report["reason"]
    assert "T0" not in report["reason"]  # Summary can enter logs; names remain in private details.
    assert report["policy"] == policy
    conn = sqlite3.connect(":memory:")
    save_attempt(conn, "synthetic", report)
    assert load_workspace(conn, "synthetic")["attempt"]["capacity"] == capacity
    summary = capacity_log_summary(report)
    assert summary["conflict_count"] == 1
    assert "conflicts" not in summary and "T0" not in json.dumps(summary, allow_nan=False)
    conn.close()


def test_insufficient_combined_capacity_is_not_misreported_as_frozen_conflict():
    frame, held, policy = evidence()
    policy["single_name_loss_budget"] = .001
    weights, report = solve_risk_aware_allocation(frame, held, [(0., .5)]*30, policy)
    assert weights is None
    capacity = report["capacity"]
    assert capacity["frozen_conflict_count"] == capacity["conflict_count"] == 0
    assert capacity["capacity_shortfall"] == pytest.approx(1.-capacity["position_cap_sum"])
    assert "combined position caps permit only" in report["reason"]


def test_minimum_sum_above_one_has_distinct_diagnosis():
    frame, held, policy = evidence()
    policy.update(single_name_loss_budget=1., adaptive_enabled=False)
    weights, report = solve_risk_aware_allocation(frame, held, [(.04, .5)]*30, policy)
    assert weights is None
    assert report["capacity"]["minimum_excess"] == pytest.approx(.2)
    assert not report["capacity"]["conflicts"]
    assert "above 100%" in report["reason"]


@pytest.mark.parametrize("bounds", [[(0., .5)], [(float("nan"), .5)]*30,
                                   [(-.1, .5)]*30, [(.2, .1)]*30, [(0., 1.1)]*30])
def test_invalid_bounds_fail_as_inputs_not_risk_infeasibility(bounds):
    frame, held, policy = evidence()
    weights, report = solve_risk_aware_allocation(frame, held, bounds, policy)
    assert weights is None and report["status"] == "unavailable"
    assert "Position bounds" in report["reason"]


def test_equality_at_cap_is_allowed_and_ordinary_overweight_can_be_reduced():
    frame, held, policy = evidence()
    policy["adaptive_enabled"] = False
    _, limits = build_policy_limits(frame, held.Weight.to_numpy(), [(.06, .06)]+[(0., .5)]*29, policy, 22.)
    assert limits["position_caps"]["T0"] == pytest.approx(.06)
    weights, report = solve_risk_aware_allocation(frame, held, [(.06, .06)]+[(0., .5)]*29, policy)
    assert weights is not None, report
    assert weights[0] == pytest.approx(.06)
    assert held.Weight.iloc[1] > limits["position_caps"]["T1"]
    assert report["capacity"]["conflict_count"] == 0  # Free overweight is not frozen.


def test_fresh_and_restored_ui_show_same_numeric_conflicts_and_old_reports_still_render():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("tests/frozen_capacity_ui_fixture.py", default_timeout=10).run()
    assert not app.exception
    detail = "EXAMPLE · frozen holding 10.00% · risk cap 3.00% · excess 7.00 percentage points."
    assert sum(item.value == detail for item in app.markdown) == 2
    assert any("previous saved analysis is retained" in item.value.lower() for item in app.warning)
    old = AppTest.from_file("tests/allocation_diagnostics_ui_fixture.py", default_timeout=10).run()
    assert not old.exception
    assert not any("Position-cap conflict" in item.value for item in old.markdown)
