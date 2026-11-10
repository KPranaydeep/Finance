"""Synthetic saved-weight refresh failure; display only, no services or solver."""
import streamlit as st
from portfolio_ui import render_execution_validation, render_metric_strip, render_allocation_policy
from execution_diagnostics import execution_summary_metric

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Saved allocation · plan review")
report = {"passed": False, "violations": ["Single-security shock budget exceeded: EXAMPLE", "Funding check failed"],
          "final_weights": {"EXAMPLE": .12}}
render_metric_strip([execution_summary_metric(report, 0)])
render_allocation_policy({"status":"saved-weights-recheck", "constraints_applied":True,
    "candidate":{"passed":False}, "limits":{"history_through":"2026-11-01"}})
proposal = [{"Symbol": "EXAMPLE", "Yahoo Ticker": "EXAMPLE", "Currency": "USD",
    "Current Weight": .2, "Optimal Weight": .12, "Action": "Sell", "Change": .08,
    "Quantity": 8., "Executable Quantity": 8., "Executable Value": 800.,
    "Native Price": 100., "FX to INR": 1., "Expected Annual Return Lift (%)": 1.25}]
render_execution_validation(report, proposal,
    allocation_policy={"limits": {"position_caps": {"EXAMPLE": .04}}},
    weights_saved_at="2026-11-01T10:00:00+00:00")
st.caption("Synthetic preview · no optimizer, downloads or trades")
