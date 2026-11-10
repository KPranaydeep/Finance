import streamlit as st
from two_trade_rebalance import LIFT,MODE,render_two_trade_plan
from portfolio_ui import render_execution_validation
st.set_page_config(layout="wide")
st.title("Synthetic two-buy selection")
payload={"rebalance_candidate_orders":[{"Yahoo Ticker":t,"Action":a,LIFT:s,"Executable Quantity":10.,"Executable Value":1000.}
    for t,a,s in [("OWNED","Buy",10.),("NEW.NS","Buy",8.),("NEW2","Buy",7.),("EXIT","Sell",9.)]],
    "current_allocation":[{"Yahoo Ticker":"OWNED","Quantity":.1},{"Yahoo Ticker":"EXIT","Quantity":20.}]}
report={"selection_mode":MODE,"selected_trade_count":3,"passed":False,"violations":["Funding not confirmed (synthetic)"]}
render_two_trade_plan(payload,report,paused=["NEW.NS"])
render_execution_validation(report)
