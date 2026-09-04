import streamlit as st


st.set_page_config(
    page_title="Weekly Signal Publisher Retired",
    page_icon="🚫",
    layout="centered",
)

st.title("Weekly Signal Publisher Retired")
st.warning(
    "This page has been permanently disabled. The public basket now uses an "
    "event-driven ledger and does not permit weekly-gated publishing."
)
st.info(
    "Use the approved event-driven rebalance workflow for future signal runs. "
    "This page cannot run the optimizer or write to the public ledger."
)
