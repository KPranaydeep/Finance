"""Joint-scenario CVaR allocation with self-funded costs and explicit risk budgets.

Only existing adjusted-price returns are used. Scenario draws are shared across
all securities; neither a bootstrap sample nor a fitted optimum is a forecast.
"""

import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from allocation_policy_config import validate_allocation_policy
from adaptive_allocation import adaptive_state, scaled_risk_scenarios, conditional_objective_means


def joint_block_scenarios(log_returns, policy):
    policy = validate_allocation_policy(policy)
    frame = pd.DataFrame(log_returns).apply(pd.to_numeric, errors="coerce")
    matrix = frame.to_numpy(dtype=float)
    if (not len(frame.columns) or frame.columns.duplicated().any()
            or frame.index.duplicated().any() or not frame.index.is_monotonic_increasing
            or len(frame) < policy["minimum_history_sessions"]
            or not np.isfinite(matrix).all()):
        raise ValueError("Complete chronological history is required for joint tail scenarios")
    cumulative = np.vstack([np.zeros((1, matrix.shape[1])), np.cumsum(matrix, axis=0)])
    rng = np.random.default_rng(policy["random_seed"])
    total = np.zeros((policy["scenario_paths"], matrix.shape[1]))
    remaining = policy["horizon_sessions"]
    while remaining:
        block = min(policy["block_sessions"], remaining)
        starts = rng.integers(0, len(matrix) - block + 1, size=len(total))
        total += cumulative[starts + block] - cumulative[starts]
        remaining -= block
    scenarios = np.expm1(total)
    if not np.isfinite(scenarios).all() or (scenarios < -1).any():
        raise ValueError("Invalid joint scenario returns")
    return scenarios


def scenario_cvar(net_returns, confidence):
    """Exact fractional empirical tail, matching the linear-program CVaR."""
    losses = np.sort(-np.asarray(net_returns, dtype=float))[::-1]
    if not len(losses) or not np.isfinite(losses).all() or not 0 < confidence < 1:
        raise ValueError("Invalid tail-loss evidence")
    mass = len(losses) * (1 - confidence)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = losses[:whole].sum()
    if fraction > 1e-12:
        total += fraction * losses[min(whole, len(losses) - 1)]
    return max(float(total / mass), 0.)


def current_weights(allocation, columns):
    frame = pd.DataFrame(allocation).copy()
    names = frame["Yahoo Ticker"].astype(str).str.strip().str.upper()
    quantity = pd.to_numeric(frame["Quantity"], errors="coerce")
    weight = pd.to_numeric(frame["Weight"], errors="coerce")
    if (names.duplicated().any() or not np.isfinite(quantity).all() or quantity.lt(0).any()
            or not np.isfinite(weight).all() or weight.lt(0).any()):
        raise ValueError("Invalid current allocation for cost/tail comparison")
    missing = set(names[quantity.gt(0)]) - set(columns)
    if missing:
        raise ValueError("Owned history missing from allocation policy: " + ", ".join(sorted(missing)))
    mapped = pd.Series(np.where(quantity.gt(0), weight, 0.), index=names)
    result = mapped.reindex(columns).fillna(0.).to_numpy()
    if result.sum() <= 0:
        raise ValueError("Positive current allocation required for funding costs")
    # Pauses define a separately excluded sleeve. Never normalize missing owned
    # history away: every owned row in this active sleeve was checked above.
    return result / result.sum()


def correlated_groups(log_returns, threshold, block_sessions=5):
    frame = pd.DataFrame(log_returns)
    aggregated = frame.groupby(np.arange(len(frame)) // block_sessions).sum()
    correlation = aggregated.corr().to_numpy()
    adjacency = np.nan_to_num(correlation, nan=0.) >= threshold
    np.fill_diagonal(adjacency, False)
    _, labels = connected_components(sparse.csr_matrix(adjacency), directed=False)
    return [np.flatnonzero(labels == group).tolist() for group in sorted(set(labels))
            if np.sum(labels == group) > 1]


def funded_target(weights, initial, rate):
    """Convert post-fee weights to amounts/base satisfying capital + fees = 1."""
    wealth = 1.
    for _ in range(100):
        cost = rate * float(np.abs(wealth * weights - initial).sum())
        updated = 1 - cost
        if updated <= 0 or not math.isfinite(updated):
            raise ValueError("Funding costs exhaust capital")
        if abs(updated - wealth) < 1e-12:
            return updated * weights, cost
        wealth = updated
    raise ValueError("Self-funding calculation did not converge")


def policy_evidence(frame, policy):
    raw = joint_block_scenarios(frame, policy)
    state = adaptive_state(frame) if policy["adaptive_enabled"] else None
    samples = scaled_risk_scenarios(raw, state, policy["horizon_sessions"]) if state else raw
    means = conditional_objective_means(raw, state, policy["horizon_sessions"]) if state else raw.mean(axis=0)
    return raw, samples, means, state


def policy_groups(frame, policy, state):
    groups = correlated_groups(frame, policy["group_correlation"], policy["block_sessions"])
    if state is not None:
        a = state["covariance_blend"]
        cov = (1-a)*frame.cov().to_numpy() + a*state["recent_covariance"]
        sd = np.sqrt(np.maximum(np.diag(cov), 0.))
        denominator = np.outer(sd, sd)
        corr = np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator > 0)
        adjacency = corr >= policy["group_correlation"]
        np.fill_diagonal(adjacency, False)
        _, labels = connected_components(sparse.csr_matrix(adjacency), directed=False)
        groups += [np.flatnonzero(labels == g).tolist() for g in sorted(set(labels)) if np.sum(labels == g) > 1]
    return [list(g) for g in sorted(set(tuple(g) for g in groups))]


def build_policy_limits(log_returns, initial, bounds, policy, cost_bps):
    policy = validate_allocation_policy(policy)
    frame = pd.DataFrame(log_returns)
    columns = list(frame.columns)
    bound_array = np.asarray(bounds, dtype=float)
    if (bound_array.shape != (len(columns), 2) or not np.isfinite(bound_array).all()
            or (bound_array < 0).any() or (bound_array > 1).any()
            or (bound_array[:, 0] > bound_array[:, 1]).any()):
        raise ValueError("Position bounds must provide finite 0 <= minimum <= maximum <= 1 for every security")
    rate = float(cost_bps) / 10000 * policy["cost_stress_multiple"]
    if not math.isfinite(rate) or not 0 <= rate < .25:
        raise ValueError("Invalid modeled cost rate for allocation policy")
    raw_scenarios, scenarios, means, state = policy_evidence(frame, policy)
    shocks = np.maximum(policy["single_name_shock_floor"],
                        np.maximum(-np.expm1(frame.min().to_numpy(dtype=float)), 0.))
    factors = state["factors"] if state is not None else np.ones(len(columns))
    # Tighten the risk ceiling, never a fixed holding's equality bound. Scaling
    # min(frozen_weight, risk_cap) would make any stressed frozen holding fail.
    raw_caps = policy["single_name_loss_budget"] / shocks / factors
    common_cap = min(1., 1.2*float(np.max(raw_caps)))
    policy_caps = np.full(len(columns),common_cap) if policy["common_cap_enabled"] else raw_caps
    caps = np.minimum(bound_array[:, 1], policy_caps)
    groups = policy_groups(frame, policy, state)
    limits = {
        "method": policy["method_version"], "settings": policy,
        "position_caps": dict(zip(columns, map(float, caps))),
        "policy_position_caps": dict(zip(columns, map(float, policy_caps))),
        "raw_position_caps": dict(zip(columns, map(float, raw_caps))),
        "cap_rule": {"mode":"common_120pct_raw_max" if policy["common_cap_enabled"] else "per_security",
                     "raw_max":float(np.max(raw_caps)), "common_cap":common_cap},
        "position_lower_bounds": dict(zip(columns, map(float, bound_array[:, 0]))),
        "position_upper_bounds": dict(zip(columns, map(float, bound_array[:, 1]))),
        "adaptive_stress_factors": dict(zip(columns, map(float, factors))),
        "stress_losses": dict(zip(columns, map(float, shocks))),
        "groups": [{"tickers":[columns[i] for i in members], "cap":policy["group_weight_limit"]}
                   for members in groups],
        "tail_limit": scenario_cvar(raw_scenarios @ initial, policy["confidence"]),
        "current_weights": dict(zip(columns, map(float, initial))),
        "cost_rate": rate,
        "history_through": str(frame.index[-1])[:10],
        "scenario_convention": f"Joint moving-block {policy['horizon_sessions']}-session buy-and-hold returns; costs funded from portfolio",
        "adaptive": state["summary"] if state else {"enabled": False},
        "objective_means": dict(zip(columns, map(float, means))),
    }
    return scenarios, limits


def position_capacity(limits):
    """Diagnose bounds only; enough capacity does not establish joint feasibility."""
    conflicts = []
    frozen_count = 0
    for ticker, cap in limits["position_caps"].items():
        lower = limits["position_lower_bounds"][ticker]
        upper = limits["position_upper_bounds"][ticker]
        frozen = lower > 0 and math.isclose(lower, upper, abs_tol=1e-12, rel_tol=0.)
        frozen_count += int(frozen)
        if lower > cap + 1e-9:
            conflicts.append({"ticker": ticker, "kind": "Frozen holding" if frozen else "Minimum position",
                "minimum_weight": lower, "policy_cap": limits["policy_position_caps"][ticker],
                "effective_cap": cap, "excess_weight": lower - cap,
                "shock": limits["stress_losses"][ticker],
                "stress_factor": limits["adaptive_stress_factors"][ticker]})
    lower_sum = float(sum(limits["position_lower_bounds"].values()))
    cap_sum = float(sum(limits["position_caps"].values()))
    return {"required_weight": 1., "minimum_weight": lower_sum,
            "position_cap_sum": cap_sum, "capacity_shortfall": max(0., 1. - cap_sum),
            "minimum_excess": max(0., lower_sum - 1.),
            "frozen_count": frozen_count, "conflict_count": len(conflicts),
            "frozen_conflict_count": sum(row["kind"] == "Frozen holding" for row in conflicts),
            "conflicts": conflicts,
            "passed": not conflicts and cap_sum >= 1. - 1e-9 and lower_sum <= 1. + 1e-9}


def capacity_failure_reason(capacity):
    parts = []
    if capacity["conflict_count"]:
        parts.append(f"{capacity['conflict_count']} minimum/frozen position(s) exceed their risk caps "
                     f"({capacity['frozen_conflict_count']} frozen); largest excess "
                     f"{max(row['excess_weight'] for row in capacity['conflicts'])*100:.2f} percentage points")
    if capacity["capacity_shortfall"] > 1e-9:
        parts.append(f"combined position caps permit only {capacity['position_cap_sum']:.2%} "
                     f"of the required 100% fully invested allocation")
    if capacity["minimum_excess"] > 1e-9:
        parts.append(f"minimum/frozen weights total {capacity['minimum_weight']:.2%}, above 100%")
    return "; ".join(parts) + ". Risk limits and frozen positions were not relaxed."


def check_policy_weights(weights, log_returns, limits, *, scenarios=None,
                         scenario_weights=None, cost_fraction=None):
    """Check target or actual executable exposures against saved policy limits."""
    frame = pd.DataFrame(log_returns)
    columns = list(frame.columns)
    w = np.asarray(weights, dtype=float)
    errors = []
    result = {"passed":False, "violations":errors, "position_breaches":[]}
    try:
        policy = validate_allocation_policy(limits["settings"])
        raw_samples, current_samples, objective_means, state = policy_evidence(frame, policy)
        if (not math.isfinite(float(limits["tail_limit"])) or not 0 <= float(limits["tail_limit"]) <= 1
                or any(not math.isfinite(float(cap)) or not 0 <= float(cap) <= 1
                       for cap in limits["position_caps"].values())
                or any(not math.isfinite(float(g["cap"])) or not 0 < float(g["cap"]) <= 1
                       for g in limits["groups"])):
            raise ValueError("Invalid saved allocation policy limits")
        if (w.shape != (len(columns),) or not np.isfinite(w).all()
                or w.min() < -1e-8 or w.sum() > 1 + 1e-7):
            raise ValueError("Invalid policy allocation weights")
        missing = [t for t, value in zip(columns,w) if value > 1e-9 and t not in limits["position_caps"]]
        if missing:
            raise ValueError("No saved shock budget for " + ", ".join(missing))
        daily_shocks = np.maximum(-np.expm1(frame.min().to_numpy(dtype=float)), 0.)
        shocks = np.maximum(policy["single_name_shock_floor"], daily_shocks)
        raw_caps = policy["single_name_loss_budget"] / shocks / (state["factors"] if state is not None else 1.)
        current_caps = np.full(len(columns), min(1.,1.2*float(np.max(raw_caps)))) if policy["common_cap_enabled"] else raw_caps
        for i, ticker in enumerate(columns):
            cap = min(float(limits["position_caps"].get(ticker, 0.)),
                      float(current_caps[i]))
            if w[i] > cap + 1e-7:
                errors.append(("Single-security position cap exceeded: " if policy["common_cap_enabled"] else
                               "Single-security shock budget exceeded: ") + str(ticker))
                result["position_breaches"].append({"ticker": str(ticker),
                    "weight": float(w[i]), "cap": float(cap), "excess": float(w[i]-cap),
                    "shock": float(shocks[i]),
                    "stress_factor": float(state["factors"][i]) if state is not None else 1.})
        weights_by_ticker = dict(zip(columns,w))
        group_checks = list(limits["groups"])
        # New positive correlations can tighten, never silently loosen, a saved policy.
        group_checks += [{"tickers":[columns[i] for i in group], "cap":policy["group_weight_limit"]}
                         for group in policy_groups(frame, policy, state)]
        if any(sum(weights_by_ticker.get(t,0.) for t in g["tickers"]) > float(g["cap"]) + 1e-7 for g in group_checks):
            errors.append("Correlated-group exposure limit exceeded")
        samples = current_samples if state is not None or scenarios is None else scenarios
        if scenario_weights is None:
            initial = np.array([limits["current_weights"].get(t,0.) for t in columns])
            exposure, cost = funded_target(w, initial, limits["cost_rate"])
        else:
            exposure, cost = np.asarray(scenario_weights, dtype=float), float(cost_fraction)
        if exposure.shape != w.shape or not np.isfinite(exposure).all() or not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid policy funding exposure")
        cvar = scenario_cvar(samples @ exposure - cost, policy["confidence"])
        historical_cvar = scenario_cvar(raw_samples @ exposure - cost, policy["confidence"])
        if cvar > float(limits["tail_limit"]) + 1e-7:
            errors.append("Joint-scenario tail loss exceeds the current-portfolio budget")
        if historical_cvar > float(limits["tail_limit"]) + 1e-7:
            errors.append("Unscaled historical tail loss exceeds the current-portfolio budget")
        result.update(cvar=cvar, tail_limit=limits["tail_limit"], cost_fraction=cost,
                      historical_cvar=historical_cvar,
                      mean_net_scenario_return=float(np.mean(samples @ exposure - cost)))
        means = objective_means
        means = (1-policy["return_shrinkage"]) * means + policy["return_shrinkage"] * np.median(means)
        result["objective_net_return"] = float(means @ exposure - cost)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        errors.append(str(exc))
    result["passed"] = not errors
    return result


def solve_risk_aware_allocation(log_returns, allocation, bounds, policy, *, cost_bps=22.):
    """Solve sparse LP in amounts/base; normalize only by solved post-fee capital."""
    started = time.perf_counter()
    report = {"method":"joint-block-cvar-cost-shock-v1", "status":"unavailable",
              "policy":dict(policy), "constraints_applied":False, "stage":"policy_validation"}
    try:
        policy = validate_allocation_policy(policy)
        frame = pd.DataFrame(log_returns)
        held = pd.DataFrame(allocation)
        owned = set()
        if {"Yahoo Ticker", "Quantity"}.issubset(held.columns):
            owned = set(held.loc[pd.to_numeric(held["Quantity"], errors="coerce").gt(0), "Yahoo Ticker"].astype(str).str.strip().str.upper())
        report["input_summary"] = {"sessions": len(frame), "assets": len(frame.columns),
            "missing_owned": len(owned - set(frame.columns)),
            "nonfinite_values": int((~np.isfinite(frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float))).sum())}
        report["stage"] = "current_allocation_inputs"
        initial = current_weights(allocation, frame.columns)
        report["stage"] = "joint_scenario_inputs"
        samples, limits = build_policy_limits(frame, initial, bounds, policy, cost_bps)
        n, m = frame.shape[1], len(samples)
        caps = np.array(list(limits["position_caps"].values()))
        lower = np.array([b[0] for b in bounds])
        report["limits"] = limits
        report["adaptive"] = limits["adaptive"]
        report["stage"] = "position_and_frozen_capacity"
        report["capacity"] = position_capacity(limits)
        if not report["capacity"]["passed"]:
            report["status"] = "infeasible"
            raise ValueError(capacity_failure_reason(report["capacity"]))
        report["stage"] = "joint_tail_group_cost_solve"
        rate, confidence = limits["cost_rate"], policy["confidence"]
        means = np.array([limits["objective_means"][ticker] for ticker in frame.columns])
        means = (1 - policy["return_shrinkage"]) * means + policy["return_shrinkage"] * np.median(means)
        # Each scenario book has its own VaR/tail slack. Adaptive stress can
        # never dilute the full-history guard by mixing the two distributions.
        books = [samples]
        if policy["adaptive_enabled"]:
            books.append(joint_block_scenarios(frame, policy))
        k = len(books)
        aux = k*(m+1)
        # Variables: x, u, k VaRs, k*m tail slacks, net wealth v.
        # A separate wealth variable keeps cap rows sparse even with 3,000 assets.
        eye = sparse.eye(n,format="csr")
        zeros = sparse.csr_matrix((n,aux+1))
        matrices = [sparse.hstack([eye,-eye,zeros]), sparse.hstack([-eye,-eye,zeros])]
        rhs = [initial,-initial]
        for multiplier, limit in ((1.,caps),(-1.,lower)):
            matrices.append(sparse.hstack([multiplier * eye,sparse.csr_matrix((n,n+aux)),
                                           sparse.csr_matrix((-multiplier*limit).reshape(-1,1))]))
            rhs.append(np.zeros(n))
        for b, book in enumerate(books):
            var_columns = sparse.csr_matrix((-np.ones(m), (np.arange(m), np.full(m,b))), shape=(m,k))
            slack_columns = sparse.hstack([sparse.csr_matrix((m,b*m)), -sparse.eye(m),
                                          sparse.csr_matrix((m,(k-b-1)*m))])
            matrices.append(sparse.hstack([-sparse.csr_matrix(book),
                sparse.csr_matrix(np.full((m,n),rate)), var_columns, slack_columns,
                sparse.csr_matrix((m,1))]))
            rhs.append(np.zeros(m))
            cvar_row = np.zeros(2*n+aux+1)
            cvar_row[2*n+b] = 1.
            cvar_row[2*n+k+b*m:2*n+k+(b+1)*m] = 1/((1-confidence)*m)
            matrices.append(sparse.csr_matrix(cvar_row.reshape(1,-1)))
            rhs.append(np.array([limits["tail_limit"]]))
        for group in limits["groups"]:
            row = np.zeros(n)
            row[[frame.columns.get_loc(t) for t in group["tickers"]]] += 1
            matrices.append(sparse.csr_matrix(np.r_[row,np.zeros(n+aux),-group["cap"]].reshape(1,-1)))
            rhs.append(np.zeros(1))
        objective = np.r_[-means,np.full(n,rate),np.zeros(aux+1)]
        budget = np.r_[np.ones(n),np.full(n,rate),np.zeros(aux+1)]
        net_wealth = np.r_[np.ones(n),np.zeros(n+aux),-1.]
        solution = linprog(objective,A_ub=sparse.vstack(matrices,format="csr"),b_ub=np.concatenate(rhs),
                           A_eq=sparse.csr_matrix(np.vstack([budget,net_wealth])),b_eq=[1.,0.],
                           bounds=[(0.,1.)]*n+[(0.,2.)]*n+[(None,None)]*k+[(0.,None)]*(k*m)+[(0.,1.)],
                           method="highs",options={"time_limit":policy["solver_time_limit_seconds"]})
        report.update(solver_status=int(solution.status), solver_message=str(solution.message))
        if not solution.success:
            report["status"] = "time_limit" if solution.status == 1 else "infeasible" if solution.status == 2 else "unavailable"
            report["reason"] = (
                "Allocation solve reached its time limit; risk constraints were not waived."
                if solution.status == 1 else
                "No fully invested mix was found under the joint tail-loss, single-security and correlated-group budgets."
                if solution.status == 2 else "Allocation solver could not establish a result: " + str(solution.message)
            )
            return None, report
        invested = solution.x[:n]
        if invested.sum() <= 0:
            raise ValueError("No positive self-funded allocation")
        weights = invested / invested.sum()
        report["stage"] = "candidate_recheck"
        check = check_policy_weights(weights,frame,limits,scenarios=samples)
        report.update(candidate=check, status="candidate" if check["passed"] else "failed_recheck",
                      constraints_applied=bool(check["passed"]), objective_net_return=float(-solution.fun),
                      modeled_post_cost_capital=float(invested.sum()))
        return (weights if check["passed"] else None), report
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        report["reason"] = str(exc)
        return None, report
    finally:
        report["elapsed_seconds"] = round(time.perf_counter()-started,3)
