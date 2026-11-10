"""Real action layout with synthetic checkpoint; no private data or solver."""
import ast
from pathlib import Path
import streamlit as st
from test_run_optimization_raw_peak import load_callback, row
from portfolio_ui import render_portfolio_heading

st.set_page_config(layout="wide")
callback, _, _ = load_callback(
    checkpoint=None if st.session_state.get("fixture_no_checkpoint") else
    {"results": [row(8350, 550, .513), row(8300, 600, .512)]}, streamlit=st)
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
namespace.update(run_analysis_placeholder=slot, CURRENT_USER="fixture", _start_optimization_at_raw_peak=callback,
                 _timer_summary=lambda timer: None, OPTIMIZATION_TIMER_KEY="fixture_timer",
                 get_unique_holdings_count=lambda owner: st.session_state.get("fixture_holdings_count", 2),
                 universal_preselection_cap=st.session_state["universal_preselection_cap"],
                 exact_optimizer_asset_cap=st.session_state["exact_optimizer_asset_cap"])
exec(compile(ast.Module(body=[action], type_ignores=[]), "<production-action-layout>", "exec"), namespace)
st.caption("Synthetic UI verification. No optimization or trade execution is performed.")
