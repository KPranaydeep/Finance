import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Retirement Corpus Calculator", layout="centered")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        h1 {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Investment Required Calculator")
st.markdown("Calculate how much you need to invest today to support inflation-adjusted withdrawals over time.")

# Formatter for currency
def format_currency(value):
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value / 1e5:.2f} Lakh"
    else:
        return f"₹{value:,.2f}"

# Inputs
col1, col2 = st.columns(2)

with col1:
    monthly_withdrawal = st.number_input("Initial Monthly Withdrawal (₹)", value=63200, step=1000)
    inflation_rate = st.number_input("Annual Inflation Rate (%)", value=6.0, step=0.1)
    duration_years = st.number_input("Investment Duration (Years)", value=100, step=1)

with col2:
    return_rate = st.number_input("Expected Annual Return (%)", value=3.0, step=0.1)
    final_balance = st.number_input("Final Balance Desired (₹)", value=0, step=100000)

if st.button("Calculate Investment Required"):
    # Convert to decimals and monthly values
    r = return_rate / 100
    g = inflation_rate / 100
    m = 12
    total_months = int(duration_years * m)

    monthly_r = (1 + r) ** (1/12) - 1
    monthly_g = (1 + g) ** (1/12) - 1

    # Calculate present value of growing annuity
    pv_annuity = monthly_withdrawal * (1 - ((1 + monthly_g) / (1 + monthly_r)) ** total_months) / (monthly_r - monthly_g)
    pv_final = final_balance / ((1 + r) ** duration_years)

    total_investment = pv_annuity + pv_final

    st.success(f"You need to invest {format_currency(total_investment)} today.")

    # Create data for graph
    months = np.arange(1, total_months + 1)
    years = months / 12
    withdrawal_series = monthly_withdrawal * ((1 + monthly_g) ** months)
    balance_series = []
    investment_series = []
    balance = total_investment

    for w in withdrawal_series:
        investment_series.append(balance)
        balance = balance * (1 + monthly_r) - w
        balance_series.append(balance)

    df = pd.DataFrame({
        "Year": years,
        "Balance": balance_series,
        "Withdrawal": withdrawal_series,
        "Investment": investment_series
    })

    # Create animated graph (in years)
    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode='lines', name='Investment Balance'),
            go.Scatter(x=[], y=[], mode='lines', name='Monthly Withdrawal')
        ],
        layout=go.Layout(
            title="Investment & Withdrawal Over Time",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Amount (₹)", range=[0, max(balance_series) * 1.1]),
            updatemenus=[dict(type="buttons", showactive=False,
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": 40, "redraw": True},
                                                         "fromcurrent": True,
                                                         "transition": {"duration": 0}}])])]
        ),
        frames=[go.Frame(
            data=[
                go.Scatter(x=years[:k], y=balance_series[:k], mode='lines', name='Investment Balance'),
                go.Scatter(x=years[:k], y=withdrawal_series[:k], mode='lines', name='Monthly Withdrawal')
            ]
        ) for k in range(1, total_months, int(total_months / 100))]
    )

    st.plotly_chart(fig, use_container_width=True)
