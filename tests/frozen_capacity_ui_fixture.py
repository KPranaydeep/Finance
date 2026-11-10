"""Synthetic frozen-cap conflict shown immediately and after a DB restore."""
from contextlib import contextmanager
import sqlite3
import streamlit as st

from allocation_attempts import save_attempt
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from allocation_policy_ui import render_last_attempt
from portfolio_ui import render_allocation_policy

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
if "db" not in st.session_state:
    st.session_state.db = sqlite3.connect(":memory:", check_same_thread=False)


@contextmanager
def connect():
    yield st.session_state.db


report = {"status": "infeasible", "policy": DEFAULT_ALLOCATION_POLICY,
    "reason": "1 minimum/frozen position(s) exceed their risk caps (1 frozen); largest excess 7.00 percentage points. Risk limits and frozen positions were not relaxed.",
    "stage": "position_and_frozen_capacity",
    "capacity": {"passed": False, "minimum_weight": .1, "position_cap_sum": 1.5,
        "required_weight": 1., "conflict_count": 1, "frozen_conflict_count": 1,
        "frozen_count": 1, "capacity_shortfall": 0., "minimum_excess": 0.,
        "conflicts": [{"ticker": "EXAMPLE", "kind": "Frozen holding", "minimum_weight": .1,
            "policy_cap": .03, "effective_cap": .03, "excess_weight": .07,
            "shock": .5, "stress_factor": 4/3}]}}
with connect() as conn:
    save_attempt(conn, "synthetic", report)
st.title("Allocation result")
render_allocation_policy(report)
render_last_attempt("synthetic", connect)
st.caption("Synthetic preview · saved analysis retained · no downloads, optimizer or trades")
