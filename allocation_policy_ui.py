"""Native controls for explicit assumptions; no optimizer or downloads on render."""
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from allocation_policy_config import validate_allocation_policy
from allocation_attempts import load_workspace, save_controls, failure_summary, stage_label

CONTROL_FIELDS = ("single_name_loss_budget", "single_name_shock_floor", "group_weight_limit",
                  "cost_stress_multiple", "solver_time_limit_seconds")


def render_policy_controls(defaults, owner, connect, *, max_weight=.5):
    with connect() as conn:
        workspace = load_workspace(conn, owner)
    try:
        stored = validate_allocation_policy(workspace.get("controls") or defaults)
        # Persist only the explicit knobs, not an old horizon/version or enable flag.
        applied = validate_allocation_policy(dict(defaults, **{key: stored[key] for key in CONTROL_FIELDS}))
        applied["adaptive_enabled"] = (workspace.get("controls") or defaults).get("adaptive_enabled", defaults["adaptive_enabled"])
    except ValueError:
        applied = validate_allocation_policy(defaults)
    key = hashlib.sha256(str(owner).encode()).hexdigest()[:12]
    with st.expander("Allocation risk controls", expanded=False):
        st.caption("Automatic adaptation is the default. The values below are risk ceilings; changing them does not alter saved weights or execute trades.")
        if not applied["enabled"]:
            st.info("The risk-budget policy is disabled in the loaded configuration.")
            return applied
        with st.form(f"allocation_controls_{key}"):
            adaptive = st.checkbox("Automatically adapt to recent 13 sessions", value=applied["adaptive_enabled"], key=f"adaptive_{key}", help="Balances a long-history anchor, recent trend continuation and short-term reversal using past prediction errors. Tightens risk estimates during stress and releases gradually. Disable for the historical/manual model.")
            loss = st.number_input("Loss budget per security (%)", min_value=.01, max_value=100., value=100*applied["single_name_loss_budget"], step=.25, key=f"loss_budget_{key}", help="Higher allows larger positions and larger modeled single-security losses.")
            shock = st.number_input("Minimum single-security shock (%)", min_value=1., max_value=100., value=100*applied["single_name_shock_floor"], step=5., key=f"shock_floor_{key}", help="Lower permits larger positions but assumes a less severe shock; worse observed daily losses still tighten the cap.")
            group = st.number_input("Correlated-group allocation cap (%)", min_value=1., max_value=100., value=100*applied["group_weight_limit"], step=5., key=f"group_cap_{key}", help="Higher permits more concentration in positively correlated securities.")
            cost = st.number_input("Trading-cost stress multiple", min_value=1., max_value=10., value=float(applied["cost_stress_multiple"]), step=.25, key=f"cost_stress_{key}")
            seconds = st.number_input("Allocation solver budget (seconds)", min_value=.1, max_value=60., value=float(applied["solver_time_limit_seconds"]), step=1., key=f"policy_seconds_{key}", help="Increase for an actual solver timeout; this does not relax any risk constraint.")
            submitted = st.form_submit_button("Apply risk controls")
        if submitted:
            applied = validate_allocation_policy(dict(applied, single_name_loss_budget=loss/100,
                single_name_shock_floor=shock/100, group_weight_limit=group/100,
                cost_stress_multiple=cost, solver_time_limit_seconds=seconds, adaptive_enabled=adaptive))
            with connect() as conn:
                save_controls(conn, owner, applied)
            st.success("Risk controls saved for this profile. They apply to the next run.")
        cap = min(max_weight, applied["single_name_loss_budget"] / applied["single_name_shock_floor"])
        st.caption("Mode: automatic feedback" if applied["adaptive_enabled"] else "Mode: manual historical model")
        st.write(f"Maximum per security before adaptive tightening: {cap:.2%}, before any worse observed shock or other bound. Correlated groups: {applied['group_weight_limit']:.0%} at positive correlation ≥ {applied['group_correlation']:.2f}.")
        st.caption("Auto mode weights the newest of 13 sessions most heavily, retains full-history crash scenarios and updates trend/reversal influence from past forecast errors. At least half of the return forecast stays anchored to longer history. Limits are never raised to force a feasible solve; modeled trading costs still apply.")
        st.caption(f"Tail budget: the current portfolio’s joint {applied['horizon_sessions']}-session {applied['confidence']:.0%} CVaR. At least {applied['minimum_history_sessions']} complete chronological observations and every active owned holding’s history are required. Historical drawdown and target-volatility checks remain separate.")
        st.caption("Input/data failures require repairing the evidence; loosening concentration limits cannot resolve them. A timeout needs more compute time. An infeasible result may require changing a diagnosed binding limit; neither higher limits nor a feasible fit guarantees future returns.")
    return applied


def render_last_attempt(owner, connect):
    with connect() as conn:
        workspace = load_workspace(conn, owner)
    attempt = workspace.get("attempt") or {}
    if not attempt or attempt.get("status") == "selected":
        return
    with st.expander("Last allocation attempt · why no plan", expanded=False):
        st.warning(failure_summary(attempt))
        try:
            recorded = datetime.fromisoformat(attempt["recorded_at"]).astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %H:%M IST")
        except (ValueError, TypeError, KeyError):
            recorded = "unknown"
        st.caption(f"Recorded {recorded} · blocked while {stage_label(attempt.get('stage'))} · no successful analysis was replaced.")
        evidence = attempt.get("input_summary") or {}
        if evidence:
            st.write(f"History: {evidence.get('sessions', 0):,} sessions · {evidence.get('assets', 0):,} assets · {evidence.get('missing_owned', 0):,} owned histories missing · {evidence.get('nonfinite_values', 0):,} nonfinite values.")
        st.caption("Use Allocation risk controls in the sidebar for the current limits and their direction. An unavailable input is not evidence that a risk limit is too tight.")
