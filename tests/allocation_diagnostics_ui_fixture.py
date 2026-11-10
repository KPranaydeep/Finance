"""Synthetic missing-history failure and real risk controls; no providers/optimizer."""
from contextlib import contextmanager
import sqlite3
import streamlit as st
from allocation_attempts import save_attempt
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from allocation_policy_ui import render_policy_controls, render_last_attempt
from portfolio_ui import render_allocation_policy

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
if "db" not in st.session_state:
    st.session_state.db = sqlite3.connect(":memory:", check_same_thread=False)

@contextmanager
def connect():
    yield st.session_state.db

report = {"status":"unavailable", "reason":"Owned history missing from allocation policy: EXAMPLE",
          "stage":"current_allocation_inputs", "policy":DEFAULT_ALLOCATION_POLICY,
          "input_summary":{"sessions":500, "assets":550, "missing_owned":1, "nonfinite_values":0}}
with connect() as conn:
    save_attempt(conn, "synthetic", report)
st.title("Allocation result")
render_allocation_policy(report)
render_last_attempt("synthetic", connect)
render_policy_controls(DEFAULT_ALLOCATION_POLICY, "synthetic", connect)
st.caption("Synthetic preview · previous successful analysis retained · no optimizer invoked")
