import streamlit as st
from portfolio_ui import render_execution_validation
from two_trade_rebalance import render_two_trade_plan,MODE,LIFT
st.set_page_config(layout="wide")
st.title("Synthetic two-trade plan")
report={"selection_mode":MODE,"selected_trade_count":2,"pair_validation_available":True,"passed":False,"violations":["Position cap conflict (synthetic)"]}
payload={"two_trade_plan":[{"Yahoo Ticker":t,"Currency":"INR","Action":side,LIFT:lift,"Executable Quantity":10.,"Executable Value":1000.} for t,side,lift in [("NEW.NS","Buy",4.),("EXAMPLE","Sell",8.)]]}
payload["current_allocation"]=[{"Yahoo Ticker":"EXAMPLE","Quantity":20.,"Currency":"INR"}]
render_two_trade_plan(payload,report,paused=["NEW.NS"])
render_execution_validation(report)
