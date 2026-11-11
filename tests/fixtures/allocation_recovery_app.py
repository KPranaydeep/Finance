import streamlit as st
from allocation_conflict_recovery import render_allocation_recovery
st.set_page_config(layout="wide")
st.title("Synthetic completed-with-conflict review")
render_allocation_recovery({"source_saved_at":"2026-11-09T00:00:00+00:00","refreshed_at":"2026-11-11T03:00:00+00:00",
    "reason":"One frozen holding exceeds its position cap. Saved target weights remain unchanged."})
