import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Streamlit Page Setup ---
st.set_page_config(page_title="LAMF Simulator", layout="wide")
st.title("📌 Loan Against Mutual Fund (LAMF) Simulator")

# --- Introductory Markdown ---
st.markdown("""
This tool simulates the financial outcome of taking a **Loan Against Mutual Funds (LAMF)**  
and investing that borrowed amount in the market.

It compares:
- 🟥 **Total Loan Outflow** (principal + interest + processing fee)  
- 🟩 **Investment Value** (with monthly compounding)  

---

""")

# --- Main Page Inputs with Markdown Descriptions ---
st.markdown("### 🔧 Simulation Inputs")

# Loan Amount
st.markdown("#### 🏦 **Loan Amount (₹)**")
st.markdown("Specify the loan amount you want to borrow against your mutual funds.")
loan_amount = st.slider("Loan Amount (₹)", min_value=25000, max_value=1000000, step=10000, value=100000)

# Interest Rate
st.markdown("#### 💸 **Loan Interest Rate (Annual %)**")
st.markdown("Select the annual interest rate charged on the loan.")
interest_rate = st.slider("Loan Interest Rate (Annual %)", min_value=4.0, max_value=18.0, step=0.25, value=10.5)

# Processing Fee
st.markdown("#### 💰 **Processing Fee (₹)**")
st.markdown("Enter the one-time processing fee that is charged for the loan.")
processing_fee = st.number_input("Processing Fee (₹)", min_value=0, max_value=10000, step=10, value=1179)

# Expected Annual Return
st.markdown("#### 📈 **Expected Market Return (Annual %)**")
st.markdown("Select the annual return rate you expect from investing in the market.")
expected_annual_return = st.slider("Expected Market Return (Annual %)", min_value=0.0, max_value=200.0, step=0.25, value=12.0)

# Loan Tenure
st.markdown("#### ⏳ **Loan Tenure (Months)**")
st.markdown("Select the number of months for the loan repayment period.")
tenure_months = st.slider("Loan Tenure (Months)", min_value=2, max_value=36, step=1, value=12)

# --- Financial Calculations ---
monthly_interest_rate = interest_rate / 12 / 100
monthly_return_rate = (1 + expected_annual_return / 100) ** (1 / 12) - 1

total_interest_paid = loan_amount * monthly_interest_rate * (tenure_months - 1)
total_outflow = loan_amount + total_interest_paid + processing_fee
investment_value = loan_amount * ((1 + monthly_return_rate) ** tenure_months)
net_profit_loss = investment_value - total_outflow
decision_text = "✅ YES, Take LAMF" if net_profit_loss > 0 else "❌ NO, Not Worth It"

# --- Custom Formatting Function ---
def format_currency(value):
    if isinstance(value, (int, float)):
        if abs(value) >= 1_00_000:
            return f"₹{value/1_00_000:.2f}Lakhs"
        else:
            return f"₹{value:,.2f}"
    return value

# --- Results Table ---
st.markdown("### 📊 Simulation Results")

results = {
    "Loan Amount": loan_amount,
    "Interest Rate (Annual)": f"{interest_rate:.2f}%",
    "Monthly Interest Rate": f"{monthly_interest_rate * 100:.3f}%",
    "Expected Return (Annual)": f"{expected_annual_return:.2f}%",
    "Monthly Return Rate": f"{monthly_return_rate * 100:.3f}%",
    "Loan Tenure (Months)": tenure_months,
    "Processing Fee": processing_fee,
    "Total Interest Paid": total_interest_paid,
    "Total Outflow (Principal + Interest + Fee)": total_outflow,
    "Investment Value at Maturity": investment_value,
    "Net Profit / Loss": net_profit_loss,
    "Decision": decision_text
}

# Apply currency formatting
formatted_results = {k: format_currency(v) for k, v in results.items()}

df_results = pd.DataFrame.from_dict(formatted_results, orient='index', columns=['Value'])
st.dataframe(df_results, use_container_width=True)

# --- Bar Chart: Visual Comparison ---
st.markdown("### 📈 Visual Comparison")

plt.style.use("seaborn-v0_8-muted")
fig, ax = plt.subplots(figsize=(4.5, 6.25))

# Scale values to lakhs
labels = ["Investment Value", "Total Outflow", "Net P&L"]
values = [investment_value, total_outflow, net_profit_loss]
values_in_lakhs = [v / 1_00_000 for v in values]
colors = ['green', 'red', 'green' if net_profit_loss > 0 else 'red']

bars = ax.bar(labels, values_in_lakhs, color=colors)

# Annotate bars with ₹ values in lakhs
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05,
            f"₹{abs(values[i]) / 1_00_000:.2f}L",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Investment vs Outflow vs Net Profit/Loss", fontsize=14, fontweight='bold')
ax.set_ylabel("₹ (in Lakhs)")
ax.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()

st.pyplot(fig)

# --- Final Emotional Verdict ---
st.markdown("### 🧠 Final Verdict")

def format_currency(value):
    if abs(value) >= 1_00_000:
        return f"{value/1_00_000:.2f}Lakhs"
    else:
        return f"₹{value:,.0f}"

formatted_profit = format_currency(net_profit_loss)

if net_profit_loss > 0:
    st.success(f"✅ Gain of {formatted_profit} — **Worth considering LAMF!**")
else:
    st.error(f"⚠️ Loss of {format_currency(abs(net_profit_loss))} — **Better avoid LAMF under these terms.**")


# --- Educational Guide ---
st.markdown("---")
st.markdown("### 📘 How This Works")
st.markdown("""
1. You borrow a sum against your mutual funds (collateralized, no liquidation).
2. You **pay interest monthly** and **repay principal before the last month**.
3. You **invest the borrowed amount** expecting monthly compounded returns.
4. At the end, we calculate:
    - 🟥 Total money *you paid* (outflow)
    - 🟩 Final investment value
5. If 🟩 > 🟥 → **Profit** 💰  
   If 🟥 > 🟩 → **Loss** 😓

---

> 📌 **Note:** This tool assumes reinvestment in high-return mutual funds or equity markets with no withdrawal during tenure.  
""")
