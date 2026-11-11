"""Real action layout with synthetic checkpoint; no private data or solver."""
import ast
from pathlib import Path
import streamlit as st
from test_run_optimization_raw_peak import load_callback, row
from portfolio_ui import render_portfolio_heading
from cap_search_status import render_cap_search_action

st.set_page_config(layout="wide")
callback, _, _ = load_callback(
    checkpoint=None if st.session_state.get("fixture_no_checkpoint") else
    {"results": [row(8350, 550, .513), row(8300, 600, .512)]}, streamlit=st)
if st.session_state.get("fixture_auto_pause"):
    callback("fixture")
render_portfolio_heading("Portfolio Rebalancer", "Review your allocation, compare risk, and prepare a rebalance.", context="PRIVATE WORKSPACE")
slot = st.container()
st.subheader("Last saved analysis")
st.caption("Synthetic saved analysis remains below the primary action.")
with st.sidebar:
    st.number_input("Shortlist cap", min_value=50, value=400, step=50, key="universal_preselection_cap")
    st.number_input("Maximum assets", min_value=50, value=300, step=50, key="exact_optimizer_asset_cap")
    st.number_input("Search warm-start X", min_value=7700, max_value=8300, value=7700,
                    step=100, key="shortlist_cap_search_start_x")
tree = ast.parse(Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8"))
action = next(n for n in tree.body if isinstance(n, ast.With)
              and any(isinstance(i.context_expr, ast.Name) and i.context_expr.id == "run_analysis_placeholder" for i in n.items))
namespace = callback.__globals__
def continue_review(attempt, owner, timer_slot, saved_slot):
    st.session_state['fixture_continued_review'] = attempt['status']
    return True

def cap_search_action(owner, connect, **kwargs):
    checkpoint = {"status":"complete", "updated_at":"2026-11-06T12:00:00+00:00",
                  "results":[row(8350,550,.513)], "analysis_cutoff":"2026-11-05"}
    clicked, age_slot = render_cap_search_action(owner, connect, **kwargs,
        load=lambda conn, profile: checkpoint, now="2026-11-11T08:00:00+00:00")
    if clicked:
        st.session_state["fixture_cap_search_clicked"] = True
    return clicked, age_slot

namespace.update(run_analysis_placeholder=slot, CURRENT_USER="fixture", _start_optimization_at_raw_peak=callback,
                 auto_pause_reoptimization=st.session_state.get("fixture_auto_pause",False),
                 _timer_summary=lambda timer: None, OPTIMIZATION_TIMER_KEY="fixture_timer",
                 load_allocation_workspace=lambda conn, owner: {"attempt": {"status": "infeasible"}} if st.session_state.get("fixture_conflict") else {},
                 continue_saved_allocation_analysis=continue_review, saved_analysis_placeholder=slot,
                 render_cap_search_action=cap_search_action, advanced_cap_search=st.session_state.get("fixture_advanced_search",True),
                 get_unique_holdings_count=lambda owner: st.session_state.get("fixture_holdings_count", 2),
                 universal_preselection_cap=st.session_state["universal_preselection_cap"],
                 exact_optimizer_asset_cap=st.session_state["exact_optimizer_asset_cap"])
exec(compile(ast.Module(body=[action], type_ignores=[]), "<production-action-layout>", "exec"), namespace)
st.caption("Synthetic UI verification. No optimization or trade execution is performed.")
