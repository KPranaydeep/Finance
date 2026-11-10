"""Optional profile controls; rendering never runs the optimizer or submits orders."""
import hashlib
import streamlit as st
from profit_cap_exception import load_profit_exception, save_profit_exception


def render_profit_exception_controls(owner, connect):
    with connect() as conn:
        applied=load_profit_exception(conn,owner)
    key=hashlib.sha256(str(owner).encode()).hexdigest()[:12]
    with st.expander("Profit-based partial release", expanded=False):
        st.caption("From blocked over-cap holdings, test individual cap-reducing sells. Release only the qualifying order nearest above the threshold. Other trades stay withheld.")
        with st.form(f"profit_cap_controls_{key}"):
            enabled=st.checkbox("Allow one qualifying cap-reducing sell",value=applied["enabled"],key=f"profit_cap_enabled_{key}")
            amount=st.number_input("Minimum additional net profit versus holding (INR)",min_value=1.,value=applied["minimum_net_profit_inr"],step=100.,key=f"profit_cap_amount_{key}")
            submitted=st.form_submit_button("Apply partial-release controls")
        if submitted:
            applied={"enabled":enabled,"minimum_net_profit_inr":amount}
            with connect() as conn:
                save_profit_exception(conn,owner,applied)
            st.success("Saved. Use Update rebalance plan to evaluate existing saved weights; no optimizer search is needed.")
        st.caption("Expected additional portfolio profit over 30 sessions after stressed modeled trading costs, before tax. Historical paired stability and tail-loss checks must also pass. No buys, paused/frozen sales, borrowing or other failed risk checks are excused. Residual position caps remain visible. This estimate is not a promised realized gain.")
    return applied
