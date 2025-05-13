# lamf_simulator.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- App Title ---
st.set_page_config(page_title="LAMF Simulator", layout="centered")
st.title("📌 Loan Against Mutual Fund (LAMF) Simulator")

# --- Introduction ---
st.markdown("""
This simulator helps you decide whether to take a **Loan Against Mutual Funds (LAMF)** and reinvest the funds in the market.  
It compares the **total loan repayment (including interest and fees)** with the **investment value** generated from the borrowed amount.

---  
""")

# --- User Inputs ---
st.sidebar.header("🔧 Input Parameters")

loan_amount = st.sidebar.slider("Loan Amount (₹)", 25000, 1000000, 100000, step=10000)
interest_rate = st.sidebar.slider("Loan Interest Rate (Annual %)", 4.0, 18.0, 10.5, step=0.25)
processing_fee = st.sidebar.number_input("Processing Fee (₹)", 0, 10000, value=1179, step=10)
expected_annual_return = st.sidebar.slider("Expected Market Return (Annual %)", 0.0, 200.0, 12.0, step=0.25)
tenure_months = st.sidebar.slider("Loan Tenure (Months)", 2, 36, 12, step=1)

# --- Calculations ---
monthly_interest_rate = interest_rate / 12 / 100
monthly_return_rate = (1 + expected_annual_return / 100) ** (1 / 12) - 1

total_interest_paid = loan_amount * monthly_interest_rate * (tenure_months - 1)
total_outflow = total_interest_paid + processing_fee + loan_amount
investment_value = loan_amount * ((1 + monthly_return_rate) ** tenure_months)
net_profit_loss = investment_value - total_outflow

# --- Results Display ---
st.markdown("### 📊 Results Summary")
results = {
    "Loan Amount": f"₹{loan_amount:,.2f}",
    "Interest Rate (Annual)": f"{interest_rate:.2f}%",
    "Monthly Interest Rate": f"{monthly_interest_rate * 100:.3f}%",
    "Expected Return (Annual)": f"{expected_annual_return:.2f}%",
    "Monthly Return Rate": f"{monthly_return_rate * 100:.3f}%",
    "Tenure (months)": tenure_months,
    "Processing Fee": f"₹{processing_fee:,.2f}",
    "Total Interest Paid": f"₹{total_interest_paid:,.2f}",
    "Total Outflow (Principal + Interest + Fee)": f"₹{total_outflow:,.2f}",
    "Investment Value at End": f"₹{investment_value:,.2f}",
    "Net Profit / Loss": f"₹{net_profit_loss:,.2f}",
    "Decision": "✅ YES, Take LAMF" if net_profit_loss > 0 else "❌ NO, Not Worth It"
}
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
st.dataframe(df_results)

# --- Bar Chart ---
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Investment Value", "Total Outflow"], [investment_value, total_outflow], color=['green', 'red'])
ax.set_ylabel("₹ Amount")
ax.set_title("Investment Value vs Loan Outflow")
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + loan_amount * 0.01, f"₹{yval:,.0f}", ha='center', va='bottom')

st.pyplot(fig)

# --- Conclusion ---
st.markdown("---")
st.markdown("### 📘 How It Works")
st.markdown("""
- You take a loan against mutual funds (collateralized).
- You pay monthly **interest only**, and repay full principal **1 month before** end of tenure.
- You invest the loan amount expecting compound market returns.
- At the end of the tenure, we compare:
  - 🟥 Total amount paid (interest + principal + fee)
  - 🟩 Final investment value

If 🟩 > 🟥 → **Profit** 💰 → LAMF might be worth it  
If 🟥 > 🟩 → **Loss** 😓 → LAMF isn't worth it

---
""")

