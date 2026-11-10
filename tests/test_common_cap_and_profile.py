import sqlite3

import numpy as np
import pandas as pd
import pytest

from allocation_policy_config import DEFAULT_ALLOCATION_POLICY, validate_allocation_policy
from execution_diagnostics import execution_position_rows, execution_summary_metric, CAP_PREFIX
from personal_profile import resolve_personal_profile
from risk_aware_allocator import build_policy_limits, check_policy_weights, current_weights


def test_common_ceiling_replaces_individual_caps_and_never_compounds_itself():
    rng=np.random.default_rng(9)
    frame=pd.DataFrame(rng.normal(.0004,.01,(300,30)),columns=[f"T{i}" for i in range(30)])
    frame.iloc[-1,0] = -.8
    policy=dict(DEFAULT_ALLOCATION_POLICY,common_cap_enabled=True,scenario_paths=100)
    initial=np.repeat(1/30,30)
    _,first=build_policy_limits(frame,initial,[(0,.5)]*30,policy,22.)
    assert first["raw_position_caps"]["T0"] < max(first["raw_position_caps"].values())
    common=1.2*max(first["raw_position_caps"].values())
    assert all(cap == pytest.approx(common) for cap in first["policy_position_caps"].values())
    assert first["position_caps"]["T0"] == pytest.approx(common)
    _,again=build_policy_limits(frame,initial,[(0,.5)]*30,policy,22.)
    assert again["cap_rule"] == first["cap_rule"]
    w=np.zeros(30); w[0]=common+.001
    checked=check_policy_weights(w,frame,first)
    assert any(v.startswith(CAP_PREFIX) for v in checked["violations"])
    assert execution_summary_metric(checked)[1] == "Blocked"
    assert execution_position_rows({"violations":checked["violations"],"allocation_policy":checked})[0]["cap"] == pytest.approx(common)


def test_common_ceiling_preserves_frozen_equality_and_legacy_rules():
    frame=pd.DataFrame(np.random.default_rng(4).normal(0,.01,(200,3)),columns=list("ABC"))
    policy=dict(DEFAULT_ALLOCATION_POLICY,adaptive_enabled=False,common_cap_enabled=True,scenario_paths=100)
    _,limits=build_policy_limits(frame,np.array([.01,.49,.5]),[(.01,.01),(0,.5),(0,.5)],policy,22.)
    assert limits["position_caps"]["A"] == .01
    assert limits["policy_position_caps"]["A"] == pytest.approx(.048)
    old=dict(policy); old.pop("common_cap_enabled")
    assert validate_allocation_policy(old)["common_cap_enabled"] is False
    with pytest.raises(ValueError,match="COMMON_CAP"):
        validate_allocation_policy(dict(policy,common_cap_enabled=1))


def test_personal_profile_opens_only_existing_owner_and_persists_without_a_prompt():
    c=sqlite3.connect(":memory:")
    c.execute("CREATE TABLE master_holdings(owner TEXT)")
    c.executemany("INSERT INTO master_holdings VALUES(?)",[("Example",),("__universal__",)])
    assert resolve_personal_profile(c) == "Example"
    c.execute("INSERT INTO master_holdings VALUES('Other')")
    assert resolve_personal_profile(c) == "Example"
    assert resolve_personal_profile(c,configured="Other") == "Other"
    assert resolve_personal_profile(c) == "Other"
    assert c.execute("SELECT count(*) FROM master_holdings").fetchone()[0] == 3
    with pytest.raises(ValueError,match="shared"):
        resolve_personal_profile(c,configured="__universal__")
    c.close()


def test_ambiguous_profiles_are_not_merged_or_silently_selected():
    c=sqlite3.connect(":memory:")
    c.execute("CREATE TABLE latest_analysis(owner TEXT)")
    c.executemany("INSERT INTO latest_analysis VALUES(?)",[("A",),("B",)])
    with pytest.raises(ValueError,match="Multiple"):
        resolve_personal_profile(c)
    assert resolve_personal_profile(c,session_profile="B") == "B"
    c.close()
    c=sqlite3.connect(":memory:")
    assert resolve_personal_profile(c) == "Personal"
    c.close()
