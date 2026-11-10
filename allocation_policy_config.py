"""Explicit allocation policy; these assumptions are not inferred risk tolerance."""

import math


DEFAULT_ALLOCATION_POLICY = {
    "enabled": True,
    "adaptive_enabled": True,
    "common_cap_enabled": False,
    "method_version": "joint-block-cvar-cost-shock-v1",
    "horizon_sessions": 30,
    "confidence": .95,
    "block_sessions": 5,
    "scenario_paths": 1024,
    "random_seed": 20261109,
    "minimum_history_sessions": 126,
    "single_name_loss_budget": .02,
    "single_name_shock_floor": .50,
    "group_correlation": .90,
    "group_weight_limit": .25,
    "return_shrinkage": .50,
    "cost_stress_multiple": 2.,
    "solver_time_limit_seconds": 10.,
}


def validate_allocation_policy(value):
    if isinstance(value, dict) and "common_cap_enabled" not in value:
        value = dict(value, common_cap_enabled=False)
    # Retained policies/checkpoints keep their original manual semantics.
    if isinstance(value, dict) and "adaptive_enabled" not in value:
        value = dict(value, adaptive_enabled=False)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_ALLOCATION_POLICY):
        raise ValueError("ALLOCATION_POLICY_KEYS_INVALID")
    if not isinstance(value["enabled"], bool) or value["method_version"] != DEFAULT_ALLOCATION_POLICY["method_version"]:
        raise ValueError("ALLOCATION_POLICY_VERSION_INVALID")
    if not isinstance(value["adaptive_enabled"], bool):
        raise ValueError("ALLOCATION_POLICY_ADAPTIVE_ENABLED_INVALID")
    if not isinstance(value["common_cap_enabled"], bool):
        raise ValueError("ALLOCATION_POLICY_COMMON_CAP_ENABLED_INVALID")
    integers = {"horizon_sessions": (2, 126), "block_sessions": (1, 30),
                "scenario_paths": (100, 4096), "random_seed": (0, 2**32 - 1),
                "minimum_history_sessions": (60, 5000)}
    for name, (low, high) in integers.items():
        x = value[name]
        if isinstance(x, bool) or not isinstance(x, int) or not low <= x <= high:
            raise ValueError("ALLOCATION_POLICY_" + name.upper() + "_INVALID")
    fractions = {"confidence": (.5, .999), "single_name_loss_budget": (.0001, 1.),
                 "single_name_shock_floor": (.01, 1.), "group_correlation": (.5, 1.),
                 "group_weight_limit": (.01, 1.), "return_shrinkage": (0., 1.),
                 "cost_stress_multiple": (1., 10.), "solver_time_limit_seconds": (.1, 60.)}
    for name, (low, high) in fractions.items():
        x = value[name]
        if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not low <= x <= high:
            raise ValueError("ALLOCATION_POLICY_" + name.upper() + "_INVALID")
    if (value["block_sessions"] > value["horizon_sessions"]
            or value["minimum_history_sessions"] < value["horizon_sessions"]):
        raise ValueError("ALLOCATION_POLICY_HORIZON_INVALID")
    return dict(value)
