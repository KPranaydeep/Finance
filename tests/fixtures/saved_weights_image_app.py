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
render_saved_weights_download(payload,["EXAMPLE00"],key="synthetic-image")
st.image(render_saved_weights_page(saved_weights_image_spec(payload,["EXAMPLE00"])),width=636)
