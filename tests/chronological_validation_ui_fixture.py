import streamlit as st
from chronological_validation_ui import render_report

st.set_page_config(layout="wide")
st.subheader("Whole-process chronological validation")
metrics = {"net_return": .045, "max_drawdown": .026, "observed_one_session_es95": .012,
           "gross_turnover": .40, "cost_fraction": .00088, "fit_seconds": 12., "held_folds": 0}
render_report({"status": "complete", "settings": {"folds": 3}, "as_of": "2026-11-09",
               "folds": [], "coverage": {"included": 120, "omitted": 3, "data_through": "2026-11-09"},
               "summary": {"decision": "No consistent improvement established in this sample",
                           "strategies": {"incumbent": metrics, "risk-aware": dict(metrics, net_return=.05, max_drawdown=.03)}},
               "limitations": ["Current supplied universe; historical membership not reconstructed", "No automatic strategy promotion"]})
