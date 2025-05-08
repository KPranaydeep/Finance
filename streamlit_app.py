import streamlit as st
import numpy as np

# === Constants ===
st.sidebar.header("🔧 User Inputs")

one_year_forecast = st.sidebar.number_input(
    "1-Year Forecast Growth (%)", min_value=0.0, value=12, step=0.1)

total_returns = st.sidebar.number_input(
    "Current Total Returns (%)", min_value=0.0, value=-5.74, step=0.1)

bonds = st.sidebar.number_input(
    "Current Bonds Value (₹)", min_value=0, value=1121, step=100)

tickertape = st.sidebar.number_input(
    "Current Investments (e.g. Stocks, Mutual funds) (₹)", min_value=100000, value=0, step=1000)

effective_annual_growth = one_year_forecast + total_returns
current_savings = bonds + tickertape


# === Title ===
st.title("📈 Financial Goal Simulator")

# === Inputs ===
target_type = st.selectbox("Target Type", ["cumulative", "monthly"])
target_value = st.number_input("Target ₹ Value", value=221445, step=1000)
target_month = st.slider("Target Month", min_value=1, max_value=12, value=12)
annual_return_pct = st.slider("Expected Annual Return (%)", min_value=1.0, max_value=200.0, value=effective_annual_growth)

months = 12
r = (1 + (annual_return_pct / 100)) ** (1 / months) - 1

# === Calculate Required Corpus ===
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

    low, high = 1, 1e9
    while high - low > 0.01:
        mid = (low + high) / 2
        if simulate_cumulative(mid) < target_value:
            low = mid
        else:
            high = mid
    required_corpus = mid

# === Output ===
st.markdown(f"### 📌 Total Required Corpus: ₹{required_corpus/1e5:.2f} lakhs")
st.markdown(f"### 📌 Additional Corpus Needed: ₹{(required_corpus - current_savings)/1e5:.2f} lakhs")

# === Monthly Simulation ===
capital = required_corpus
profits = []
cumulative = []

for i in range(months):
    monthly_profit = capital * r
    profits.append(monthly_profit)
    capital += monthly_profit
    cumulative.append(sum(profits))

# === Table ===
st.subheader("📊 Monthly Profit Table")
table_data = {"Month": list(range(1, months+1)),
              "Profit (₹)": profits,
              "Cumulative (₹)": cumulative}
st.dataframe(table_data)
