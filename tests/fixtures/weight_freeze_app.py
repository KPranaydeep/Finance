"""Isolated synthetic override controls; never accesses the user's database."""
from contextlib import contextmanager,closing
import sqlite3
from pathlib import Path
import tempfile
import streamlit as st
from weight_freeze_overrides import render_weight_freeze_controls
from paused_allocation import render_pause_research

@contextmanager
def connect():
    with closing(sqlite3.connect(st.session_state['db'])) as conn:
        yield conn

st.title('Weight freeze controls · synthetic')
if 'db' not in st.session_state:
    st.session_state['db']=str(Path(tempfile.mkdtemp(prefix='weight-freeze-synthetic-'))/'preview.sqlite')
render_weight_freeze_controls('synthetic',connect)
render_pause_research({'status':'conditional_research','research_kind':'price_freeze_override',
                       'weight_freeze_overrides':['EXAMPLE.NS']})
