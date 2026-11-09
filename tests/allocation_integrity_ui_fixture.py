"""Synthetic offline fixture: no production DB, provider or optimizer access."""
import streamlit as st
from portfolio_ui import render_execution_validation

st.set_page_config(page_title="Final allocation check preview", layout="wide")
st.title("Saved analysis")
st.metric("Holdings analysed", 15)
state = st.segmented_control("Validation state", ["Passed", "Withheld", "Older plan"], default="Withheld")
report = {"passed": True} if state == "Passed" else (
    {"passed": False, "violations": ["Final position exceeds its weight cap: SYNTHETIC",
                                      "Missing final holding history: UNAVAILABLE"]}
    if state == "Withheld" else None)
proposal = [{"Yahoo Ticker":"SYNTHETIC", "Action":"Buy", "Executable Quantity":2}]
render_execution_validation(report, proposal if state == "Withheld" else [])
st.caption("Analysis retained · no trades submitted")
