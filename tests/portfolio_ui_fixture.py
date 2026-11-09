"""Offline visual fixture for shared portfolio components; synthetic values only."""
import streamlit as st
from portfolio_ui import render_metric_strip, render_portfolio_heading

st.set_page_config(page_title='Portfolio UI preview', layout='wide')
workspace = st.segmented_control('Preview workspace', ['Public', 'Private'], default='Public')
if workspace == 'Public':
    render_portfolio_heading(
        'A risk-aware model portfolio.',
        'Transparent allocation, observed performance, and practical guidance to invest or rebalance.',
        context='SYNTHETIC UI PREVIEW', publication='Immutable publication · P012',
    )
    st.subheader('Portfolio at a glance')
    unavailable = st.checkbox('Preview unavailable outlook', value=False)
    render_metric_strip([
        ('Constituents', 27, 'Allocated securities'),
        ('Median outcome', 'Unavailable' if unavailable else '4.95%',
         'Awaiting sufficient usable model history; no outcome is estimated.' if unavailable else '28 calendar days · estimated'),
        ('Data as of', '06 Nov 2026', 'Asia/Kolkata'),
    ])
else:
    render_portfolio_heading('Portfolio Rebalancer',
                             'Review your allocation, compare risk, and prepare a rebalance.',
                             context='PRIVATE WORKSPACE · SYNTHETIC PREVIEW')
    st.badge('15 holdings', icon=':material/account_balance_wallet:', color='gray')
    with st.expander('Run diagnostics', expanded=False):
        st.caption('Synthetic diagnostic information')
    st.subheader('Last saved analysis')
    with st.container(horizontal=True, gap='small'):
        st.button('Update rebalance plan', icon=':material/sync:')
        st.button('Refresh portfolio evidence', icon=':material/monitoring:')
        st.button('Refresh buy / sell scenarios', icon=':material/refresh:')
    render_metric_strip([
        ('Holdings analysed', 15, 'Saved analysis'),
        ('Total invested', '₹385,179.17', 'Saved invested base'),
        ('Executable trades', 7, 'Review before execution'),
    ])
st.caption('Preview only · no database, optimizer or market requests.')
