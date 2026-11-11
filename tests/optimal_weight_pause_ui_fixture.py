"""Synthetic pause-editor fixture; no optimizer, quotes or private database."""
from contextlib import contextmanager, closing
import sqlite3

import streamlit as st

from trade_deferrals_ui import render_active_security_pauses, render_optimal_weight_pause_editor


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
st.session_state['full_app_runs'] = st.session_state.get('full_app_runs', 0) + 1
st.caption(f"Full app executions: {st.session_state['full_app_runs']}")
if st.session_state.get('registry_only'):
    render_active_security_pauses('synthetic-owner', connection, 'registry-test')
    st.stop()
if not st.session_state.get('one_shot_results') or st.button('Run synthetic analysis'):
    st.metric('Saved analysis metric', '12.34%')
    render_optimal_weight_pause_editor(weights, 'synthetic-owner', 'saved-test', connection,
                                      refresh_app_on_pause=st.session_state.get('fixture_refresh_on_pause',False))
    st.markdown('Analysis results remain visible below the pause controls.')
if st.session_state.get('show_second_view'):
    render_optimal_weight_pause_editor(weights, 'synthetic-owner', 'live-test', connection)
