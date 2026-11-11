import streamlit as st
from two_trade_rebalance import LIFT, MODE, render_two_trade_plan

st.set_page_config(layout="wide")
st.title("Synthetic currency rebalance plan")
rows=[]
held=[]
for currency in ("INR","USD"):
    for ticker,action,score in (("NEW","Buy",8.),("NEXT","Buy",6.),("OWNED","Buy",10.),("EXIT","Sell",9.)):
        rows.append({"Yahoo Ticker":f"{currency}_{ticker}","Currency":currency,"Action":action,
                     LIFT:score,"Executable Quantity":2.,"Executable Value":1000.})
    held.extend({"Yahoo Ticker":f"{currency}_{ticker}","Currency":currency,"Quantity":.5} for ticker in ("OWNED","EXIT"))
payload={"rebalance_candidate_orders":rows,"current_allocation":held}
report={"selection_mode":MODE,"passed":False,"excluded_tickers":["USD_NEW"],"violations":["Synthetic funding review pending"]}
render_two_trade_plan(payload,report,paused=["USD_NEW"],anchor="saved-rebalancing-plan")
