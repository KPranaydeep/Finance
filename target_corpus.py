import streamlit as st
import numpy as np

# === Page Config ===
st.set_page_config(
    page_title="📈 Financial Goal Simulator",
    layout="centered",
    initial_sidebar_state="auto"
)

# === Title ===
st.title("📈 Financial Goal Simulator")
st.markdown("Welcome to the **Financial Goal Simulator**! This app helps you understand how much money you need to invest today to reach your monthly or cumulative financial target in the coming months.")

st.markdown("---")
st.header("🔧 Your Financial Details")
st.markdown("Enter your current investment values and their expected 1-year growth rate:")

# === Layout ===
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Bond")
    bonds = st.number_input(
        "Current Bonds Value (₹)",
        min_value=0.0,
        value=2000.0,
        step=100.0,
        help="Total value of your fixed-income instruments like bonds."
    )
    one_year_forecast_bonds = st.number_input(
        "Expected YTM (%)",
        min_value=0.0,
        value=11.0,
        step=0.1,
        help="Expected growth in your bonds over the next 1 year."
    )

with col2:
    st.subheader("📊 Stock")
    stocks = st.number_input(
        "Current Stocks Value (₹)",
        min_value=0.0,
        value=374113.0,
        step=1000.0,
        help="Market value of your stock holdings."
    )
    one_year_forecast_stocks = st.number_input(
        "Expected Growth (%)",
        min_value=0.0,
        value=100.0,
        step=0.1,
        help="Expected growth in your stocks over the next 1 year."
    )

with col3:
    st.subheader("📈 Mutual Fund")
    mf = st.number_input(
        "Current Mutual Funds Value (₹)",
        min_value=0.0,
        value=1625000.0,
        step=1000.0,
        help="Market value of your mutual fund holdings."
    )
    one_year_forecast_mf = st.number_input(
        "Expected Growth (%)",
        min_value=0.0,
        value=12.0,
        step=0.1,
        help="Expected growth in your mutual funds over the next 1 year."
    )

# Total investment across all categories
total_investment = bonds + stocks + mf

# Handle the case when total investment is zero to avoid division by zero
if total_investment == 0:
    effective_annual_growth = 0.0
else:
    effective_annual_growth = (
        (bonds * one_year_forecast_bonds) +
        (stocks * one_year_forecast_stocks) +
        (mf * one_year_forecast_mf)
    ) / total_investment

# Total current savings
current_savings = total_investment

# === Section: Goal Inputs ===
st.subheader("🎯 Define Your Goal")
st.markdown("### ℹ️ Target Type")
st.markdown(
    "- **Monthly**: Choose this if you want to receive a fixed income every month.\n"
    "- **Cumulative**: Choose this if you want to accumulate a total target amount over time."
)

target_type = st.selectbox("Target Type", ["monthly", "cumulative"])

st.markdown("### 🎯 Target ₹ Value")
st.markdown(
    "- **Monthly Mode**: Enter how much you want to receive every month.\n"
    "- **Cumulative Mode**: Enter the total amount you aim to accumulate over your chosen period."
)

target_value = st.number_input("Target ₹ Value", value=22245.0, step=1000.0)

st.markdown("### 📅 Time Horizon (in Months)")
st.markdown(
    "- Select the number of months over which you want to achieve your financial goal.\n"
    "- Range: 1 to 12 months."
)

# Calculate months until Dasara (October 2, 2025) from today
today = date.today()
dasara_date = date(2025, 10, 2)
dasara_months = (dasara_date - today).days // 30  # Approximate months

# Ensure initial slider value is in range [1, 12]
initial_value = dasara_months if 1 <= dasara_months <= 12 else 6

# Time horizon slider
target_month = st.slider(
    "📅 Time Horizon (Months)",
    min_value=1,
    max_value=12,
    value=initial_value,
    help="Choose a time horizon between 1 and 12 months for your financial goal."
)

annual_return_pct = st.slider(
    "📊 Expected Annual Return (%)",
    min_value=1.0,
    max_value=200.0,
    value=effective_annual_growth,
    help="How much annual return you expect from your investments."
)

# Monthly return rate
months = 12
r = (1 + (annual_return_pct / 100)) ** (1 / months) - 1

# Weekly return rate
weeks = 52
weekly_r = (1 + (annual_return_pct / 100)) ** (1 / weeks) - 1

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

# === Week 1 and Month 1 Profit vs Current Savings Comparison ===
st.subheader("📊 One Week and One Month Profit vs. Your Current Savings")

# Slider input for hypothetical corpus (₹25k steps)
test_corpus_combined = st.slider(
    "🧮 Simulate Profit From This Corpus (₹)",
    min_value=0.0,
    max_value=float(required_corpus),
    value=float(current_savings),
    step=25000.0,
    help="Adjust to see profit in One Week and One Month from different corpus sizes."
)

# Compute profits
week1_profit_simulated = test_corpus_combined * weekly_r
week1_profit_required = required_corpus * weekly_r

month1_profit_simulated = test_corpus_combined * r
month1_profit_required = required_corpus * r

# === Two Column Display ===
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🗓️ **1st Week**")
    st.metric(
        label="1st Week Simulated Profit",
        value=f"₹{week1_profit_simulated:,.0f}",
        delta=f"{week1_profit_simulated - week1_profit_required:,.0f}",
        delta_color="normal" if week1_profit_simulated >= week1_profit_required else "inverse"
    )
    st.caption(f"Target: ₹{week1_profit_required:,.0f}")
    if week1_profit_simulated >= week1_profit_required:
        st.success("✅ Sufficient for 1st Week profit goal")
    else:
        st.error("⚠️ Below 1st Week profit target")

with col2:
    st.markdown("### 📅 **1st Month**")
    st.metric(
        label="1st Month Simulated Profit",
        value=f"₹{month1_profit_simulated:,.0f}",
        delta=f"{month1_profit_simulated - month1_profit_required:,.0f}",
        delta_color="normal" if month1_profit_simulated >= month1_profit_required else "inverse"
    )
    st.caption(f"Target: ₹{month1_profit_required:,.0f}")
    if month1_profit_simulated >= month1_profit_required:
        st.success("✅ Sufficient for 1st Month profit goal")
    else:
        st.error("⚠️ Below 1st Month profit target")

# === Footer Note ===
st.markdown("""
---
📌 *Disclaimer: This is a simplified financial planning tool.  
Please consult a professional financial advisor before making investment decisions.*
""")
