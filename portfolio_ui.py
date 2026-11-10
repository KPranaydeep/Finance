"""Shared native Streamlit presentation; no data loading or financial logic."""
from __future__ import annotations

import streamlit as st
import pandas as pd
from allocation_attempts import stage_label
from execution_diagnostics import execution_position_rows, execution_summary_metric, cap_breach_ticker


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


def render_execution_validation(report, proposed_orders=(), *, allocation_policy=None, weights_saved_at=None):
    """Present final feasibility separately from research and target statistics."""
    report = report or {}
    with st.container(border=True):
        st.markdown("**Selected-pair checks**" if report.get("selection_mode")=="highest-annual-lift-buy-sell" else "**Final allocation check**")
        exception=report.get("profit_cap_exception") or {}
        if report.get("selection_mode")=="highest-annual-lift-buy-sell":
            st.info("The highest annual-lift Buy and Sell stay selected. Control conflicts below do not hide or replace either trade.")
            if not report.get("pair_validation_available",True):
                st.caption("This older saved draft has not had the selected pair replayed together. Update rebalance plan refreshes sizing and checks without optimizer search.")
            elif report.get("passed"):
                st.caption("Selected pair passed the recorded checks; no broker orders have been submitted.")
            for reason in dict.fromkeys(report.get("violations") or []):
                st.write("• "+str(reason))
        elif report.get("passed") and exception.get("applied"):
            st.warning(f"Partial release: {exception['released_ticker']} only. Other draft trades remain withheld.")
            st.write(f"Estimated additional net profit versus holding: ₹{exception['estimated_additional_net_profit_inr']:,.2f} over 30 sessions, after stressed modeled costs, before tax. Threshold ₹{exception['settings']['minimum_net_profit_inr']:,.2f}; nearest qualifying order above it.")
            st.caption(f"Historical comparison through {exception.get('as_of','unavailable')}. This is an estimate, not booked profit or a guaranteed outcome. Pauses, frozen positions, funding and all non-cap risk checks remain enforced.")
            remaining=exception.get("remaining_over_cap") or []
            st.write(f"{len(remaining)} residual cap breaches: " + (", ".join(remaining) or "none"))
            with st.expander("Partial-release evidence"):
                st.write(exception.get("evidence") or {})
                breaches=execution_position_rows(dict(report,violations=report.get("cap_exception_violations") or []),allocation_policy)
                if breaches:
                    st.dataframe(pd.DataFrame([{"Security":r["ticker"],"After sale (%)":None if r["weight"] is None else 100*r["weight"],
                        "Cap (%)":None if r["cap"] is None else 100*r["cap"]} for r in breaches]),hide_index=True,width="stretch",column_config={
                            "After sale (%)":st.column_config.NumberColumn(format="%.2f%%"),"Cap (%)":st.column_config.NumberColumn(format="%.2f%%")})
                st.caption(exception.get("confidence_note") or "")
        elif report.get("passed"):
            st.success("Plan passed the recorded funding, position and available risk checks.")
            st.caption("Sale proceeds must be available before buys. This is modeled feasibility, not broker execution approval.")
        elif report:
            st.warning("Plan withheld: the final allocation did not pass every required check. Saved analysis is retained.")
            rows = execution_position_rows(report, allocation_policy)
            if rows:
                st.write(f"{len(rows)} {'holding would' if len(rows) == 1 else 'holdings would'} still exceed {'its position limit' if len(rows) == 1 else 'their position limits'} after the draft trades.")
                if weights_saved_at:
                    st.caption(f"Allocation saved {str(weights_saved_at)[:10]}. Update rebalance plan keeps these weights; it does not optimize a new allocation under today's limits.")
                st.caption("A partial sale can reduce a holding and still leave it above its cap. Buys can also exceed the cap. This draft is not an executable plan.")
                comparison = pd.DataFrame([{"Security": row["ticker"],
                    "After draft (%)": None if row["weight"] is None else 100*row["weight"],
                    "Recorded cap (%)": None if row["cap"] is None else 100*row["cap"],
                    "Excess (pp)": None if row["excess"] is None else 100*row["excess"]} for row in rows])
                st.dataframe(comparison, hide_index=True, width="stretch", column_config={
                    "After draft (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Recorded cap (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "Excess (pp)": st.column_config.NumberColumn(format="%.2f"),
                })
                if any(row["cap_basis"] != "checked" for row in rows):
                    st.caption("Older check: caps shown are retained upper ceilings; any excess shown is a minimum. Exact checked caps were not recorded. Missing values remain unavailable; no history or solver was rerun.")
                st.caption("A new optimization can change the allocation; Update rebalance plan only resizes it. Frozen-holding conflicts must be resolved before a feasible allocation can be established.")
            for reason in dict.fromkeys(report.get("violations") or ["Validation evidence unavailable"]):
                if cap_breach_ticker(reason) is None:
                    st.write("• " + str(reason))
        else:
            st.info("This older plan has no final allocation check. Use Update plan to validate it without rerunning optimization.")
        if exception and not exception.get("applied"):
            st.caption("Partial release unavailable: " + str(exception.get("reason","required evidence missing")))
        if proposed_orders and report.get("selection_mode")!="highest-annual-lift-buy-sell":
            with st.expander("Withheld proposal — review only"):
                st.caption("This is a draft for review, excluded from the executable plan.")
                draft = pd.DataFrame(proposed_orders).copy()
                for name in ("Current Weight", "Optimal Weight", "Change"):
                    if name in draft:
                        draft[name] = 100*pd.to_numeric(draft[name], errors="coerce")
                draft = draft.rename(columns={
                    "Executable Quantity": "Proposed quantity", "Executable Value": "Proposed value (INR)",
                    "Current Weight": "Current allocation (%)", "Optimal Weight": "Saved allocation (%)" if weights_saved_at else "Proposed allocation (%)",
                    "Change": "Change (pp)", "Expected Annual Return Lift (%)": "Historical annual-return lift (pp)",
                })
                formats = {name: st.column_config.NumberColumn(format="%.2f%%")
                           for name in ("Current allocation (%)", "Saved allocation (%)", "Proposed allocation (%)")}
                formats.update({name: st.column_config.NumberColumn(format="%.2f") for name in (
                    "Change (pp)", "Historical annual-return lift (pp)", "Native Price", "Proposed value (INR)")})
                formats.update({name: st.column_config.NumberColumn(format="%.4f") for name in (
                    "Quantity", "Proposed quantity", "FX to INR")})
                st.dataframe(draft, hide_index=True, width="stretch", column_config=formats)


def empty_trade_plan_message(deferred_count=0, validation=None):
    if validation is not None and not validation.get("passed"):
        if not validation:
            return "Saved plan needs validation. Use Update rebalance plan; optimizer search is not required."
        reasons = "; ".join(dict.fromkeys(validation.get("violations") or ["Required checks did not pass"]))
        return "Plan withheld: " + reasons + " Saved analysis and the proposal are retained."
    if deferred_count:
        return f"All {deferred_count} proposed trades are currently paused. See their expiry dates below."
    return "No executable trades in this plan."


def render_capacity_diagnostics(report):
    """The same private numeric explanation for fresh and restored failures."""
    capacity = report.get("capacity") or {}
    if not capacity or capacity.get("passed"):
        return
    st.markdown("**Position-cap conflict**")
    st.write(f"Minimum/frozen allocation {capacity['minimum_weight']:.2%} · combined position caps {capacity['position_cap_sum']:.2%} · required allocation 100%.")
    if capacity.get("capacity_shortfall", 0) > 1e-9:
        st.write(f"Position capacity is short by {capacity['capacity_shortfall']*100:.2f} percentage points before group or tail constraints.")
    if capacity.get("minimum_excess", 0) > 1e-9:
        st.write(f"Minimum/frozen allocation exceeds available capital by {capacity['minimum_excess']*100:.2f} percentage points.")
    for row in sorted(capacity.get("conflicts") or [], key=lambda row: (-row["excess_weight"], row["ticker"])):
        # Short native paragraphs stay readable on mobile and show the exact gap.
        st.write(f"{row['ticker']} · {row['kind'].lower()} {row['minimum_weight']:.2%} · risk cap {row['policy_cap']:.2%} · excess {row['excess_weight']*100:.2f} percentage points.")
        st.caption(f"Shock {row['shock']:.1%} × adaptive stress {row['stress_factor']:.2f}; effective position ceiling {row['effective_cap']:.2%}.")
    if capacity.get("frozen_conflict_count"):
        st.caption("A frozen holding cannot be reduced by this solve. Repair its quarantined price evidence before allowing it to trade, or explicitly revise the diagnosed risk assumption.")
    st.caption("More solver time cannot fix this bounds conflict. Position capacity alone does not establish group or tail-loss feasibility.")


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
    saved_recheck = (report.get("source") == "saved-weight-refresh" or report.get("status") == "saved-weights-recheck"
        or (report.get("status") == "failed_recheck" and "candidate" in report
            and not report.get("stage") and "selected" not in report))
    with st.container(border=True):
        st.markdown("**Allocation policy · tail loss, costs and concentration**")
        cap_rule = limits.get("cap_rule") or {}
        if cap_rule.get("mode") == "common_120pct_raw_max":
            st.write(f"Common per-security ceiling {cap_rule['common_cap']:.2%} = 1.2 × highest raw cap {cap_rule['raw_max']:.2%}. Recalculated from raw evidence; other hard bounds still apply.")
        adaptive = report.get("adaptive") or limits.get("adaptive") or {}
        if adaptive.get("method"):
            st.write(f"Auto-adaptive · {adaptive['recent_sessions']} recent sessions · trend {adaptive['trend_share']:.0%} / reversal {adaptive['reversal_share']:.0%} / long-history anchor {adaptive['anchor_share']:.0%}.")
            st.caption(f"Average model mix across securities, not portfolio allocation. Median stress multiplier {adaptive['median_stress_factor']:.2f}×; {adaptive['stressed_assets']:,} raw risk estimates tightened. Historical crash and current-mix tail guards remain active. Feedback uses past one-step prediction errors, not proof of future returns.")
        elif policy.get("adaptive_enabled") is False:
            st.caption("Manual model · recent adaptive adjustments are disabled for this run.")
        if (report.get("candidate") or {}).get("cap_exception_applied"):
            st.warning("One cap-reducing sell qualifies for a profit-based partial release. The full allocation still has unresolved position caps; see Final allocation check.")
        elif report.get("constraints_applied") and (report.get("candidate") or {}).get("passed") is not False:
            title = ("Risk-aware allocator selected" if selected == "risk-aware" else
                     "Existing allocator retained within the same risk limits" if selected == "incumbent" else
                     "Saved weights checked against the current policy")
            st.write(title)
            metrics = report.get(str(selected) + "_comparison") or report.get("candidate") or {}
            if metrics.get("cvar") is not None:
                st.caption(f"Joint {policy.get('horizon_sessions',30)}-session CVaR {policy.get('confidence',.95):.0%}: {metrics['cvar']:.2%} · current-mix budget {float(limits.get('tail_limit',0)):.2%} · modeled cost {metrics.get('cost_fraction',0):.2%} at {policy.get('cost_stress_multiple',1):g}× cost stress.")
        else:
            if saved_recheck:
                st.warning("The draft at saved weights failed the current risk check. No new allocation was optimized.")
            elif report.get("status") == "unavailable":
                st.warning("Allocation inputs unavailable; this does not establish that risk limits are too strict. Previous saved analysis is retained.")
            elif report.get("status") == "time_limit":
                st.warning("Allocation time budget reached; previous saved analysis is retained.")
            else:
                st.warning("No allocation passed this policy; previous saved analysis is retained.")
            if saved_recheck:
                st.caption("See Final allocation check below for the holdings and limits blocking this draft. Update rebalance plan preserves saved weights.")
            else:
                st.caption(report.get("reason") or report.get("solver_message") or "Check candidate constraints below.")
                st.caption(f"Blocked while {stage_label(report.get('stage'))}. See Allocation risk controls in the sidebar for the applied limits and trade-offs.")
            render_capacity_diagnostics(report)
        if limits.get("history_through"):
            activity = "saved-weight check · optimizer not run" if saved_recheck else f"{report.get('elapsed_seconds',0):.2f}s allocator solve"
            st.caption(f"Model history through {limits['history_through']} · {policy.get('scenario_paths',0):,} joint resamples · {activity}.")
        with st.expander("Policy assumptions and allocator comparison", expanded=False):
            if policy:
                if policy.get("common_cap_enabled"):
                    st.write(f"Base loss-budget input {policy['single_name_loss_budget']:.1%} and shock floor {policy['single_name_shock_floor']:.0%} generate raw caps. The shared 1.2× maximum replaces individual caps; it can permit a name's modeled shock loss above that base input. Portfolio tail and group limits remain separate.")
                else:
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
