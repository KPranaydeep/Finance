import streamlit as st
from paused_allocation import render_pause_research
st.set_page_config(layout="wide")
st.title("Synthetic pause-aware allocation")
render_pause_research({"status":"conditional_research","tradable_fraction":.7461,"frozen_fraction":.2539,
                      "non_worsening":True,"cost_convention":"Scenario costs; strict execution checks remain separate.",
                      "inherited_breaches":[{"ticker":"EXAMPLE.NS","frozen_weight":.1255,"risk_cap":.048}]})
st.caption("Existing paused holdings are counted; unowned paused candidates are excluded.")
