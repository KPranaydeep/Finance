import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re

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

st.title("💰 Retirement Corpus Calculator")
st.markdown(
    "Choose whether you want to **calculate required investment** "
    "or **check how long your current investment will last**."
)

def indian_number_format(number):
    """Format number with Indian commas, e.g., 12,34,56,789"""
    x = str(int(round(number)))  # Round and convert to int string
    pattern = re.compile(r"(\d+)(\d{3})(?!\d)")
    while pattern.match(x):
        x = pattern.sub(r"\1,\2", x)
    return x

def format_currency(value):
    """Format currency in ₹ with Cr, Lakh, or normal formatting"""
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value / 1e5:.2f} Lakh"
    else:
        return f"₹{value:,.2f}"

mode = st.radio("Select Mode", ["Calculate Required Investment", "Calculate Investment Duration"], horizontal=True)

# Common inputs
col1, col2 = st.columns(2)
with col1:
    monthly_withdrawal = st.number_input("Initial Monthly Withdrawal (₹)", value=63200, step=1000)
    inflation_rate = st.number_input("Annual Inflation Rate (%)", value=6.0, step=0.1)
with col2:
    return_rate = st.number_input("Expected Annual Return (%)", value=3.0, step=0.1)

# Convert annual rates to monthly decimal
r = return_rate / 100
g = inflation_rate / 100
monthly_r = (1 + r) ** (1 / 12) - 1
monthly_g = (1 + g) ** (1 / 12) - 1

if mode == "Calculate Required Investment":
    duration_years = st.number_input("Investment Duration (Years)", value=100, step=1)
    final_balance = st.number_input("Final Balance Desired (₹)", value=0, step=100000)

    if st.button("Calculate Investment Required"):
        total_months = int(duration_years * 12)

        # Present Value of growing annuity formula
        try:
            pv_annuity = monthly_withdrawal * (
                1 - ((1 + monthly_g) / (1 + monthly_r)) ** total_months
            ) / (monthly_r - monthly_g)
        except ZeroDivisionError:
            st.error("Monthly return rate and monthly withdrawal growth cannot be equal.")
            st.stop()

        # Present Value of final lump sum
        pv_final = final_balance / ((1 + r) ** duration_years)

        total_investment = pv_annuity + pv_final
        st.success(f"You need to invest {format_currency(total_investment)} today.")

        # Time series for plot
        months = np.arange(1, total_months + 1)
        years = months / 12
        withdrawal_series = monthly_withdrawal * ((1 + monthly_g) ** months)

        balance_series = []
        balance = total_investment

        for w in withdrawal_series:
            balance_series.append(balance)
            balance = balance * (1 + monthly_r) - w

        step = max(1, int(total_months / 100))
        
        fig = go.Figure(
            data=[
                go.Scatter(x=[], y=[], mode='lines', name='Investment Balance', yaxis="y1"),
                go.Scatter(x=[], y=[], mode='lines', name='Monthly Withdrawal', yaxis="y2")
            ],
            layout=go.Layout(
                title="Investment & Withdrawal Over Time",
                xaxis=dict(title="Year"),
                yaxis=dict(title="Investment Balance (₹ Cr)", side="left"),
                yaxis2=dict(title="Withdrawal (₹ Lakh)", side="right", overlaying="y"),
                updatemenus=[dict(type="buttons", showactive=False,
                                  buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None, {"frame": {"duration": duration_years, "redraw": True},
                                                             "fromcurrent": True,
                                                             "transition": {"duration": 0}}])])])
            ),
            frames=[go.Frame(
                data=[
                    go.Scatter(x=years[:k], y=np.array(balance_series[:k]) / 1e7, mode='lines', name='Investment Balance', yaxis="y1"),
                    go.Scatter(x=years[:k], y=np.array(withdrawal_series[:k]) / 1e5, mode='lines', name='Monthly Withdrawal', yaxis="y2")
                ]
            ) for k in range(1, total_months, step)]
        )

        st.plotly_chart(fig, use_container_width=True)

elif mode == "Calculate Investment Duration":
    current_investment_lakhs = st.number_input(
        "Current Investment (₹ in Lakhs)",
        min_value=0.0,
        value=20.0,
        step=1.0,
        format="%.2f"
    )
    current_investment_rupees = current_investment_lakhs * 100000
    st.write(f"Current Investment in Rupees: ₹{indian_number_format(current_investment_rupees)}")

    if st.button("Calculate Duration"):
        balance = current_investment_rupees
        months = 0
        while balance > 0:
            withdrawal = monthly_withdrawal * ((1 + monthly_g) ** months)
            balance = balance * (1 + monthly_r) - withdrawal
            if balance < 0:
                break
            months += 1

        years = months // 12
        rem_months = months % 12
        st.success(f"Your investment will last for {years} years and {rem_months} months.")
