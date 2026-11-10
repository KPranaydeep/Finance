"""Shared native Streamlit presentation; no data loading or financial logic."""
from __future__ import annotations

import streamlit as st
import pandas as pd
from allocation_attempts import stage_label


def render_portfolio_heading(title, description, *, context=None, publication=None):
    if context:
        st.caption(context)
    st.title(title)
    st.write(description)
    if publication:
        st.badge(publication, icon=":material/verified:", color="gray")


def render_metric_strip(items):
    """Responsive native cards with visible notes, including unavailable states."""
    items = list(items)
    if not items:
        return
    # Native columns stack on small screens instead of squeezing or clipping
    # dates, currency amounts and unavailable-state explanations.
    for column, (label, value, note) in zip(st.columns(len(items), gap="small"), items):
        with column.container(border=True):
            st.metric(label, value)
            if note:
                st.caption(note)


def render_execution_validation(report, proposed_orders=()):
    """Present final feasibility separately from research and target statistics."""
    report = report or {}
    with st.container(border=True):
        st.markdown("**Final allocation check**")
        if report.get("passed"):
            st.success("Plan passed the recorded funding, position and available risk checks.")
            st.caption("Sale proceeds must be available before buys. This is modeled feasibility, not broker execution approval.")
        elif report:
            st.warning("Plan withheld: the final allocation did not pass every required check. Saved analysis is retained.")
            for reason in dict.fromkeys(report.get("violations") or ["Validation evidence unavailable"]):
                st.write("• " + str(reason))
        else:
            st.info("This older plan has no final allocation check. Use Update plan to validate it without rerunning optimization.")
        if proposed_orders:
            with st.expander("Withheld proposal — review only"):
                st.caption("This is a draft for review, excluded from the executable plan.")
                draft = pd.DataFrame(proposed_orders).rename(columns={
                    "Executable Quantity": "Proposed quantity", "Executable Value": "Proposed value",
                })
                st.dataframe(draft, hide_index=True, width="stretch")


def empty_trade_plan_message(deferred_count=0, validation=None):
    if validation is not None and not validation.get("passed"):
        if not validation:
            return "Saved plan needs validation. Use Update rebalance plan; optimizer search is not required."
        reasons = "; ".join(dict.fromkeys(validation.get("violations") or ["Required checks did not pass"]))
        return "Plan withheld: " + reasons + " Saved analysis and the proposal are retained."
    if deferred_count:
        return f"All {deferred_count} proposed trades are currently paused. See their expiry dates below."
    return "No executable trades in this plan."


def render_allocation_policy(report):
    """Compact policy/comparison evidence, without a claim of forecast improvement."""
    if not report:
        st.caption("This saved allocation predates the risk-budget allocator. Update plan checks its existing weights without optimizing again.")
        return
    if report.get("status") == "disabled":
        st.caption("Risk-budget allocator disabled in the loaded configuration.")
        return
    limits = report.get("limits") or {}
    policy = report.get("policy") or limits.get("settings") or {}
    selected = report.get("selected")
    with st.container(border=True):
        st.markdown("**Allocation policy · tail loss, costs and concentration**")
        adaptive = report.get("adaptive") or limits.get("adaptive") or {}
        if adaptive.get("method"):
            st.write(f"Auto-adaptive · {adaptive['recent_sessions']} recent sessions · trend {adaptive['trend_share']:.0%} / reversal {adaptive['reversal_share']:.0%} / long-history anchor {adaptive['anchor_share']:.0%}.")
            st.caption(f"Average model mix across securities, not portfolio allocation. Median stress multiplier {adaptive['median_stress_factor']:.2f}×; {adaptive['stressed_assets']:,} securities tightened. Historical crash and current-mix tail guards remain active. Feedback uses past one-step prediction errors, not proof of future returns.")
        elif policy.get("adaptive_enabled") is False:
            st.caption("Manual model · recent adaptive adjustments are disabled for this run.")
        if report.get("constraints_applied"):
            title = ("Risk-aware allocator selected" if selected == "risk-aware" else
                     "Existing allocator retained within the same risk limits" if selected == "incumbent" else
                     "Saved weights checked against the current policy")
            st.write(title)
            metrics = report.get(str(selected) + "_comparison") or report.get("candidate") or {}
            if metrics.get("cvar") is not None:
                st.caption(f"Joint {policy.get('horizon_sessions',30)}-session CVaR {policy.get('confidence',.95):.0%}: {metrics['cvar']:.2%} · current-mix budget {float(limits.get('tail_limit',0)):.2%} · modeled cost {metrics.get('cost_fraction',0):.2%} at {policy.get('cost_stress_multiple',1):g}× cost stress.")
        else:
            if report.get("status") == "unavailable":
                st.warning("Allocation inputs unavailable; this does not establish that risk limits are too strict. Previous saved analysis is retained.")
            elif report.get("status") == "time_limit":
                st.warning("Allocation time budget reached; previous saved analysis is retained.")
            else:
                st.warning("No allocation passed this policy; previous saved analysis is retained.")
            st.caption(report.get("reason") or report.get("solver_message") or "Check candidate constraints below.")
            st.caption(f"Blocked while {stage_label(report.get('stage'))}. See Allocation risk controls in the sidebar for the applied limits and trade-offs.")
        if limits.get("history_through"):
            st.caption(f"Model history through {limits['history_through']} · {policy.get('scenario_paths',0):,} joint resamples · {report.get('elapsed_seconds',0):.2f}s allocator solve.")
        with st.expander("Policy assumptions and allocator comparison", expanded=False):
            if policy:
                st.write(f"Single-security loss budget {policy['single_name_loss_budget']:.1%}; assumed shock at least {policy['single_name_shock_floor']:.0%}, increased for worse observed daily losses. Correlated groups are capped at {policy['group_weight_limit']:.0%} when linked by positive correlation ≥ {policy['group_correlation']:.2f}.")
                st.caption("These are explicit policy assumptions, not safe-loss limits learned from prices. Negative correlations remain separate. Missing evidence cannot pass.")
                st.caption(f"Scenario means are shrunk {policy['return_shrinkage']:.0%} toward their cross-security median. Policy CVaR uses joint buy-and-hold scenarios and differs from the separate portfolio-statistics bootstrap.")
                st.caption("Resampling creates scenarios, not additional independent market history. Neither this comparison nor the configured stress budgets establish future returns or a guaranteed loss bound.")
            for label in ("incumbent","risk-aware"):
                check = report.get(label + "_validation") or {}
                comparison = report.get(label + "_comparison") or {}
                if comparison:
                    st.write(f"{label.capitalize()} · estimated net scenario return {comparison['mean_net_scenario_return']:+.2%} · CVaR {comparison['cvar']:.2%} · cost {comparison['cost_fraction']:.2%}")
                elif check.get("violations"):
                    st.write(label.capitalize() + " · " + "; ".join(dict.fromkeys(check["violations"])))
            st.caption(f"Joint block scenarios preserve cross-security moves and short serial dependence. CVaR is the modeled mean loss in the worst {1-policy.get('confidence',.95):.0%} tail, not a maximum loss. Comparison uses the same fitted history; chronological finalist checks remain separate. No additional market download or new library is required.")
