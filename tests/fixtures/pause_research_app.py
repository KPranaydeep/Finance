import streamlit as st
from paused_allocation import render_pause_exclusion
st.set_page_config(layout="wide")
st.title("Synthetic pause-aware allocation")
render_pause_exclusion({"mode":"unpaused-capital-v1","excluded_tickers":["EXAMPLE.NS","EXAMPLE"]})
st.caption("Active pauses are excluded from solver inputs and optimal weights.")
