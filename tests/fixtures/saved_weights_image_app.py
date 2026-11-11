"""Synthetic preview; no private database, optimizer or external data."""
import streamlit as st
from saved_allocation_image import render_saved_weights_download, saved_weights_image_spec, render_saved_weights_page

st.set_page_config(layout="wide")
st.title("Saved optimal weights · image export")
payload={"saved_at":"2026-11-11T08:30:00+00:00","history":{"valid_end":"2026-11-10"},
         "allocation_validation":{"passed":True},
         "optimization_scope":{"mode":"unpaused-capital-v1","excluded_tickers":["EXCLUDED.NS"]},
         "optimal_allocation":[{"ticker":f"EXAMPLE{i:02d}"+(".NS" if i%2 else ""),"target_weight":1/28} for i in range(28)],
         "current_allocation":[{"Yahoo Ticker":f"EXAMPLE{i:02d}"+(".NS" if i%2 else ""),"Currency":"INR" if i%2 else "USD"} for i in range(28)]}
payload["current_stats"]={"Annual Return":.465,"Annual Volatility":.1645,"Sharpe Ratio":2.33,
    "Historical ES 95% (1 Session)":.0208,"Block-Bootstrap ES 95% (30 Sessions)":.0726,
    "Block-Bootstrap Median Return (30 Sessions)":.0386,"Block-Bootstrap Tail Loss SD 95% (30 Sessions)":.0195}
payload["optimal_stats"]={"Annual Return":.4838,"Annual Volatility":.1171,"Sharpe Ratio":3.43,
    "Historical ES 95% (1 Session)":.014,"Block-Bootstrap ES 95% (30 Sessions)":.0413,
    "Block-Bootstrap Median Return (30 Sessions)":.0407,"Block-Bootstrap Tail Loss SD 95% (30 Sessions)":.0156}
payload["portfolio_metrics_config"]={"risk_free_rate_annual":.08225,"trading_days_per_year":252}
render_saved_weights_download(payload,["EXAMPLE00"],key="synthetic-image")
st.image(render_saved_weights_page(saved_weights_image_spec(payload,["EXAMPLE00"])),width=636)
