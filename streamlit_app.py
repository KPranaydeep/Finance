import streamlit as st
import numpy as np

# === Sidebar: User Inputs ===
st.sidebar.header("🔧 Your Financial Details")

st.sidebar.markdown("""
Enter your current investment details and growth expectations below.
These will be used to simulate how much corpus you need to meet your financial goal.
""")

one_year_forecast = st.sidebar.number_input(
    "📈 1-Year Forecast Growth (%)",
    min_value=0.0,
    value=12.0,
    step=0.1,
    help="Expected growth in your investments over the next 1 year."
)

total_returns = st.sidebar.number_input(
    "📉 Current Total Returns (%)",
    min_value=-30.0,
    value=0.0,
    step=0.1,
    help="Recent actual return on your investments. Can be negative."
)

bonds = st.sidebar.number_input(
    "💰 Current Bonds Value (₹)",
    min_value=0.0,
    value=0.0,
    step=100.0,
    help="Total value of your fixed-income instruments like bonds."
)

tickertape = st.sidebar.number_input(
    "📊 Current Investments (e.g. Stocks, MFs) (₹)",
    min_value=0.0,
    value=0.0,
    step=1000.0,
    help="Market value of your equity/mutual fund holdings."
)

# Calculate effective growth rate and savings
effective_annual_growth = one_year_forecast + total_returns
current_savings = bonds + tickertape

# === Title ===
st.title("📈 Financial Goal Simulator")
st.markdown("""
Welcome to the **Financial Goal Simulator**!  
This app helps you understand how much money you need to invest today to reach your monthly or cumulative financial target in the coming months.
""")

# === Section: Goal Inputs ===
st.subheader("🎯 Define Your Goal")

target_type = st.selectbox(
    "Target Type",
    ["cumulative", "monthly"],
    help="Choose 'monthly' if you need a fixed income every month, or 'cumulative' if you aim to accumulate a total sum."
)

target_value = st.number_input(
    "🎯 Target ₹ Value",
    value=10000.0,
    step=1000.0,
    help="How much money you want to receive in total (cumulative) or per month (monthly)."
)

target_month = st.slider(
    "📅 Time Horizon (Months)",
    min_value=1,
    max_value=12,
    value=1,
    help="Choose a time horizon between 1 and 12 months for your financial goal."
)

annual_return_pct = st.slider(
    "📊 Expected Annual Return (%)",
    min_value=1.0,
    max_value=200.0,
    value=effective_annual_growth,
    help="How much annual return you expect from your investments."
)
st.markdown("""
💡 **Tip:** This is the annual percentage return you expect from your investments.

If you're unsure about the **1-Year Forecast Growth (%)** and **Current Total Returns (%)**, you can manually adjust this value.  
You can also modify it to simulate different growth scenarios based on your own expectations or predictions.
""")

# Monthly return rate
months = 12
r = (1 + (annual_return_pct / 100)) ** (1 / months) - 1

# === Section: Corpus Calculation ===
st.subheader("📉 Required Corpus Calculation")

if target_type == "monthly":
    required_corpus = target_value / ((1 + r) ** (target_month - 1) * r)
else:
    def simulate_cumulative(corpus):
        capital = corpus
        profits = []
        for _ in range(target_month):
            monthly_profit = capital * r
            profits.append(monthly_profit)
            capital += monthly_profit
        return sum(profits)

    low, high = 1.0, 1e9
    while high - low > 0.01:
        mid = (low + high) / 2
        if simulate_cumulative(mid) < target_value:
            low = mid
        else:
            high = mid
    required_corpus = mid

# === Output Summary ===
st.markdown("### ✅ Results Summary")

st.success(f"📌 **Total Required Corpus**: ₹{required_corpus/1e5:.2f} lakhs")
st.warning(f"📌 **Additional Corpus Needed**: ₹{(required_corpus - current_savings)/1e5:.2f} lakhs")

# === Simulation Table ===
st.subheader("📊 Monthly Profit Simulation")

capital = required_corpus
profits = []
cumulative = []

for i in range(months):
    monthly_profit = capital * r
    profits.append(monthly_profit)
    capital += monthly_profit
    cumulative.append(sum(profits))

# Display in table format
import pandas as pd

# === Format values using Indian number system and round to integers ===
def format_inr(x):
    return f"{int(round(x)):,}".replace(",", "_").replace("_", ",")

table_data = {
    "Month": list(range(1, months + 1)),
    "Profit (₹)": [format_inr(p) for p in profits],
    "Cumulative (₹)": [format_inr(c) for c in cumulative],
}

df = pd.DataFrame(table_data)

# === Display the table without vertical scroll ===
st.dataframe(df, use_container_width=True, height=400)

# === Footer Note ===
st.markdown("""
---
📌 *Disclaimer: This is a simplified financial planning tool.  
Please consult a professional financial advisor before making investment decisions.*
""")
