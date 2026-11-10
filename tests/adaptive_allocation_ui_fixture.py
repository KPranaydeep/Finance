"""Synthetic adaptive report and actual intervention form; no solver/provider."""
from contextlib import contextmanager
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from adaptive_allocation import adaptive_state
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from allocation_policy_ui import render_policy_controls
from portfolio_ui import render_allocation_policy

st.set_page_config(layout="wide")
st.title("Automatic allocation feedback")
rng = np.random.default_rng(5)
frame = pd.DataFrame(rng.normal(.0005,.01,(300,30)), index=pd.bdate_range("2025-01-01",periods=300))
frame.iloc[-1,0] = -.15
state = adaptive_state(frame)
render_allocation_policy({"status":"selected", "selected":"risk-aware", "constraints_applied":True,
    "policy": DEFAULT_ALLOCATION_POLICY, "adaptive":state["summary"],
    "limits":{"tail_limit":.05, "settings":DEFAULT_ALLOCATION_POLICY},
    "risk-aware_comparison":{"cvar":.035,"cost_fraction":.001,"mean_net_scenario_return":.02}})
if "fixture_db" not in st.session_state:
    st.session_state.fixture_db = sqlite3.connect(":memory:", check_same_thread=False)
@contextmanager
def connect():
    yield st.session_state.fixture_db
applied = render_policy_controls(DEFAULT_ALLOCATION_POLICY, "synthetic", connect)
st.caption("Synthetic UI preview · model settings only · no optimizer or trades invoked")
