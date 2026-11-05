"""Synthetic pause-editor fixture; no optimizer, quotes or private database."""
from contextlib import contextmanager, closing
import sqlite3

import streamlit as st

from trade_deferrals_ui import render_optimal_weight_pause_editor


@contextmanager
def connection():
    with closing(sqlite3.connect(st.session_state['fixture_db'])) as conn:
        conn.row_factory = sqlite3.Row
        yield conn


weights = [
    {'ticker': 'HOLD.NS', 'target_weight': .6},
    {'ticker': 'NOORDER', 'target_weight': .4},
]
st.title('Optimal weights · synthetic preview')
render_optimal_weight_pause_editor(weights, 'synthetic-owner', 'saved-test', connection)
if st.session_state.get('show_second_view'):
    render_optimal_weight_pause_editor(weights, 'synthetic-owner', 'live-test', connection)
