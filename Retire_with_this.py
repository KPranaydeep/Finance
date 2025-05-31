import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Retirement Corpus Calculator", layout="centered")

# Custom CSS styling
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

st.title("Smart SWP Planner: Inflation-Proof")
st.markdown("Choose whether you want to **calculate required investment** or **check how long your current investment will last**.")

# Formatter for Indian currency style
def indian_number_format(number):
    num_str = str(int(round(number)))
    if len(num_str) <= 3:
        return num_str
    last_three = num_str[-3:]
    rest = num_str[:-3]
    rest_with_commas = ''
    while len(rest) > 2:
        rest_with_commas = ',' + rest[-2:] + rest_with_commas
        rest = rest[:-2]
    rest_with_commas = rest + rest_with_commas
    return rest_with_commas + ',' + last_three

# Mode selector
mode = st.radio("Select Mode", ["Calculate Required Investment", "Calculate Investment Duration"], horizontal=True)

# === MODE 1: CALCULATE REQUIRED INVESTMENT ===
if mode == "Calculate Required Investment":
    duration_years = st.number_input("Investment Duration (Years)", value=99, step=1)
    final_balance = st.number_input("Final Balance Desired (₹)", value=0, step=100000)
    monthly_withdrawal = st.number_input("Monthly Withdrawal (₹)", value=63200, step=1000)

    annual_r = st.number_input("Expected Annual Return (%)", value=11.75, step=0.1, format="%.2f")
    annual_g = st.number_input("Annual Increase in Withdrawal (%)", value=6.0, step=0.1, format="%.2f")

    # Convert annual to monthly rates
    monthly_r = (1 + annual_r / 100) ** (1 / 12) - 1
    monthly_g = (1 + annual_g / 100) ** (1 / 12) - 1

    if st.button("Calculate Investment Required"):
        total_months = int(duration_years * 12)

        # === BACKWARD CALCULATION to meet exact final balance ===
        withdrawal_schedule_rev = monthly_withdrawal * ((1 + monthly_g) ** np.arange(total_months))[::-1]

        balance = final_balance
        for w in withdrawal_schedule_rev:
            balance = (balance + w) / (1 + monthly_r)

        total_investment = balance

        st.success(f"You need to invest ₹{indian_number_format(round(total_investment))} today "
                   f"(`{total_investment / 1e7:.2f} Cr`).")

        # === FORWARD SIMULATION ===
        months = np.arange(0, total_months + 1)
        years = months / 12
        withdrawal_series = monthly_withdrawal * ((1 + monthly_g) ** months)

        balance_series = []
        balance = total_investment
        for i in range(total_months + 1):
            balance_series.append(balance)
            withdrawal = withdrawal_series[i]
            balance = balance * (1 + monthly_r) - withdrawal
            balance = max(balance, 0)  # Clamp to avoid negative drift

        # Force final value to match exactly
        balance_series[-1] = final_balance

        # Normalize to ₹ Lakhs
        withdrawal_series_lakhs = withdrawal_series / 1e5
        balance_series_lakhs = np.array(balance_series) / 1e5

        # Create animation frames
        step = max(1, int(total_months / 100))
        frames = [
            go.Frame(
                data=[
                    go.Scatter(x=years[:k], y=balance_series_lakhs[:k], mode='lines', name='Investment Balance (₹ Lakhs)'),
                    go.Scatter(x=years[:k], y=withdrawal_series_lakhs[:k], mode='lines', name='Monthly Withdrawal (₹ Lakhs)')
                ]
            )
            for k in range(2, total_months + 1, step)
        ]

        fig = go.Figure(
            data=[
                go.Scatter(x=years, y=balance_series_lakhs, mode='lines', name='Investment Balance (₹ Lakhs)'),
                go.Scatter(x=years, y=withdrawal_series_lakhs, mode='lines', name='Monthly Withdrawal (₹ Lakhs)')
            ],
            layout=go.Layout(
                title="Investment & Withdrawal Over Time",
                xaxis=dict(title="Year", range=[0, duration_years]),
                yaxis=dict(title="₹ in Lakhs"),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    )]
                )]
            ),
            frames=frames
        )

        st.plotly_chart(fig, use_container_width=True)

# === MODE 2: CALCULATE INVESTMENT DURATION ===
elif mode == "Calculate Investment Duration":
    monthly_withdrawal = st.number_input("Initial Monthly Withdrawal (₹)", value=63200, step=1000)
    inflation_rate = st.number_input("Annual Inflation Rate (%)", value=6.0, step=0.1)
    return_rate = st.number_input("Expected Annual Return (%)", value=3.0, step=0.1)

    current_investment_lakhs = st.number_input(
        "Current Investment (₹ in Lakhs)",
        min_value=0.0,
        value=20.0,
        step=1.0,
        format="%.2f"
    )

    monthly_r = (1 + return_rate / 100) ** (1 / 12) - 1
    monthly_g = (1 + inflation_rate / 100) ** (1 / 12) - 1

    current_investment_rupees = current_investment_lakhs * 100000
    formatted_investment = indian_number_format(current_investment_rupees)
    st.write(f"Current Investment in Rupees: ₹{formatted_investment}")

    if st.button("Calculate Duration"):
        balance = current_investment_rupees
        months = 0

        while balance > 0:
            withdrawal = monthly_withdrawal * ((1 + monthly_g) ** months)
            balance = balance * (1 + monthly_r) - withdrawal
            if balance < 0:
                break
            months += 1

        if months == 0:
            st.warning("⚠️ Your investment cannot support even the first month of withdrawal.")
        else:
            years = months // 12
            rem_months = months % 12
            st.success(f"Your investment will last for {years} years and {rem_months} months.")
