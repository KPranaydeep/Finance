"""Synthetic browser/callback fixture: no production data or solver access."""
import streamlit as st
from test_run_optimization_raw_peak import load_callback, row

st.set_page_config(layout="wide")
callback, _, _ = load_callback(
    checkpoint={"results": [row(8350, 550, .513), row(8300, 600, .512)]}, streamlit=st)
st.subheader("Run analysis")
st.caption("Automatically selects the latest saved raw-return X/Y peak. Refits on current data.")
st.number_input("Shortlist cap", min_value=50, value=400, step=50, key="universal_preselection_cap")
st.number_input("Maximum assets", min_value=50, value=300, step=50, key="exact_optimizer_asset_cap")
st.number_input("Search warm-start X", min_value=7700, max_value=8300, value=7700,
                step=100, key="shortlist_cap_search_start_x")
if st.button("Run optimization", on_click=callback, args=("fixture",), type="primary"):
    selection = st.session_state["optimization_raw_peak_selection"]
    st.success(f"Using saved raw-return peak X {selection['x']:,} × Y {selection['y']:,}. Refitting on current data.")
st.caption("Synthetic UI verification. No optimization or trade execution is performed.")
