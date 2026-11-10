import sqlite3
from contextlib import nullcontext
import streamlit as st
from profit_cap_exception_ui import render_profit_exception_controls
from portfolio_ui import render_execution_validation, render_metric_strip
from execution_diagnostics import execution_summary_metric

st.set_page_config(layout="wide")
st.title("Synthetic partial-release preview")
if "test_db" not in st.session_state:
    st.session_state.test_db=sqlite3.connect(":memory:",check_same_thread=False)
settings=render_profit_exception_controls("synthetic",lambda:nullcontext(st.session_state.test_db))
st.json(settings)
report={"passed":True,"profit_cap_exception":{"applied":True,"released_ticker":"EXAMPLE",
    "estimated_additional_net_profit_inr":8120.,"settings":{"minimum_net_profit_inr":8079.},
    "remaining_over_cap":["EXAMPLE","SAMPLE"],"as_of":"2026-11-09","evidence":{"Orders":1}}}
render_metric_strip([execution_summary_metric(report,1)])
render_execution_validation(report)
