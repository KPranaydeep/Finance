import streamlit as st
from buy_candidate_watchlist import render_buy_candidates
st.set_page_config(layout="wide")
st.title("Synthetic buy-candidate preview")
render_buy_candidates({"optimal_allocation":[{"ticker":"NEW.NS","target_weight":.4},{"ticker":"EXAMPLE","target_weight":.6}],
    "current_allocation":[{"Yahoo Ticker":"NEW.NS","Currency":"INR","Weight":0.},{"Yahoo Ticker":"EXAMPLE","Currency":"USD","Weight":1.}],
    "execution_validation":{"passed":False},"security_scenario_ranking":[{"Yahoo Ticker":"NEW.NS","Expected net scenario return (%)":3.,"Probability of positive net return (%)":55.,"Price data through":"2026-11-09"}]})
