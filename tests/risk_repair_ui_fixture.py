"""Synthetic-only standalone repair, persistence and changed-context preview."""
from contextlib import contextmanager
import sqlite3
import streamlit as st
from risk_repair import prepare_risk_repair, save_repair
from risk_repair_ui import render_risk_repair
from test_risk_repair import evidence

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
st.title("Sell-only risk reduction")
if "db" not in st.session_state:
    st.session_state.db = sqlite3.connect(":memory:",check_same_thread=False)
changed = st.checkbox("Preview holdings changed",False)
@contextmanager
def connect():
    yield st.session_state.db
def context():
    return {"holdings":"changed" if changed else "original"}
def prepare():
    held, history, policy = evidence()
    report = prepare_risk_repair(held,history,{t:10. for t in history},{"USD":1.},policy)
    with connect() as conn:
        save_repair(conn,"synthetic",report,context())
render_risk_repair("synthetic",connect,context,prepare,key="fixture")
st.caption("Synthetic data · no market providers, portfolio optimizer or broker orders")
