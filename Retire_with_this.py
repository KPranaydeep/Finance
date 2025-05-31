import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Retirement Corpus Calculator", layout="centered")

# === FORMATTERS ===
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

def format_inr_crores_lakhs(amount):
    if amount >= 1e7:
        return f"₹{indian_number_format(round(amount))} (`{amount / 1e7:.2f} Cr`)"
    else:
        return f"₹{indian_number_format(round(amount))} (`{amount / 1e5:.2f} Lakhs`)"

# === CORE SIMULATION LOGIC ===
def simulate_minimum_investment(
    total_months,
    monthly_r,
    monthly_g,
    final_balance,
    monthly_withdrawal,
    precision=1.0,
    max_iter=100
):
    months = list(range(0, total_months + 1))
    withdrawal_series = monthly_withdrawal * ((1 + monthly_g) ** np.array(months))

    low = 0
    high = 1e9
    best_investment = None

    for _ in range(max_iter):
        mid = (low + high) / 2
        balance = mid
        valid = True

        for w in withdrawal_series:
            if balance < w:
                valid = False
                break
            balance = balance * (1 + monthly_r) - w

        if valid and balance >= final_balance:
            best_investment = mid
            high = mid
        else:
            low = mid

        if high - low <= precision:
            break

    return best_investment, withdrawal_series

# === UI: MODE SELECTOR ===
st.title("Smart SWP Planner: Inflation-Proof")

mode = st.radio("Select Mode", ["Calculate Required Investment", "Calculate Investment Duration"], horizontal=True)

# === MODE 1 ===
if mode == "Calculate Required Investment":
    duration_years = st.number_input("Investment Duration (Years)", value=99, step=1)
    final_balance = st.number_input("Final Balance Desired (₹)", value=0, step=100000)
    monthly_withdrawal = st.number_input("Monthly Withdrawal (₹)", value=63200, step=1000)

    annual_r = st.number_input("Expected Annual Return (%)", value=11.75, step=0.1, format="%.2f")
    annual_g = st.number_input("Annual Increase in Withdrawal (%)", value=6.0, step=0.1, format="%.2f")

    # Monthly conversion
    monthly_r = (1 + annual_r / 100) ** (1 / 12) - 1
    monthly_g = (1 + annual_g / 100) ** (1 / 12) - 1

    if st.button("Calculate Minimum Investment"):
        total_months = int(duration_years * 12)

        # Run smart simulation
        total_investment, withdrawal_series = simulate_minimum_investment(
            total_months, monthly_r, monthly_g, final_balance, monthly_withdrawal
        )

        if total_investment is None:
            st.error("❌ Could not find a feasible investment. Try changing input parameters.")
        else:
            st.success(f"💰 Minimum Investment Required: {format_inr_crores_lakhs(total_investment)}")

            # === Forward simulate balances
            months = np.arange(0, total_months + 1)
            years = months / 12

            balance_series = []
            balance = total_investment
            for i in range(total_months + 1):
                balance_series.append(balance)
                withdrawal = withdrawal_series[i]
                balance = balance * (1 + monthly_r) - withdrawal
                balance = max(balance, 0)

            # Clamp last balance
            balance_series[-1] = max(balance_series[-1], final_balance)

            # Peak stats
            max_balance = max(balance_series)
            max_balance_index = balance_series.index(max_balance)
            max_withdrawal = max(withdrawal_series)
            max_withdrawal_index = np.argmax(withdrawal_series)

            st.info(f"📈 Peak Investment Balance: {format_inr_crores_lakhs(max_balance)} at Year {max_balance_index // 12}")
            st.info(f"📤 Peak Monthly Withdrawal: {format_inr_crores_lakhs(max_withdrawal)} at Year {max_withdrawal_index // 12}")

            # Plot
            withdrawal_lakhs = withdrawal_series / 1e5
            balance_lakhs = np.array(balance_series) / 1e5

            fig = go.Figure(
                data=[
                    go.Scatter(x=years, y=balance_lakhs, mode='lines', name='Investment Balance (₹ Lakhs)'),
                    go.Scatter(x=years, y=withdrawal_lakhs, mode='lines', name='Monthly Withdrawal (₹ Lakhs)')
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
                )
            )

            st.plotly_chart(fig, use_container_width=True)

# === MODE 2 ===
elif mode == "Calculate Investment Duration":
    monthly_withdrawal = st.number_input("Initial Monthly Withdrawal (₹)", value=63200, step=1000)
    inflation_rate = st.number_input("Annual Inflation Rate (%)", value=6.0, step=0.1)
    return_rate = st.number_input("Expected Annual Return (%)", value=3.0, step=0.1)

    current_investment_lakhs = st.number_input(
        "Current Investment (₹ in Lakhs)", min_value=0.0, value=20.0, step=1.0, format="%.2f"
    )

    monthly_r = (1 + return_rate / 100) ** (1 / 12) - 1
    monthly_g = (1 + inflation_rate / 100) ** (1 / 12) - 1

    current_investment = current_investment_lakhs * 1e5
    st.write(f"Current Investment: ₹{indian_number_format(int(current_investment))}")

    if st.button("Calculate Duration"):
        balance = current_investment
        months = 0
        while balance > 0:
            withdrawal = monthly_withdrawal * ((1 + monthly_g) ** months)
            balance = balance * (1 + monthly_r) - withdrawal
            if balance < 0:
                break
            months += 1

        if months == 0:
            st.error("⚠️ Your investment cannot support even one month.")
        else:
            st.success(f"Your investment will last for {months // 12} years and {months % 12} months.")
