"""Synthetic policy evidence only; no optimizer, network or production data."""
import streamlit as st
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
from portfolio_ui import render_allocation_policy

st.set_page_config(page_title="Risk allocation preview",layout="wide")
st.title("Saved analysis")
st.metric("Holdings analysed",30)
state=st.segmented_control("Allocator result",["Risk-aware","Incumbent","Withheld"],default="Risk-aware")
policy=dict(DEFAULT_ALLOCATION_POLICY)
selected="risk-aware" if state=="Risk-aware" else "incumbent"
report={"selected":selected,"constraints_applied":state!="Withheld","policy":policy,
        "limits":{"tail_limit":.035,"settings":policy},
        "reason":"Shock limits and frozen positions cannot form a fully invested allocation",
        "risk-aware_comparison":{"mean_net_scenario_return":.018,"cvar":.012,"cost_fraction":.0044},
        "incumbent_comparison":{"mean_net_scenario_return":.015,"cvar":.025,"cost_fraction":.0022}}
render_allocation_policy(report)
st.caption("Synthetic preview · analysis retained · no trades submitted")
