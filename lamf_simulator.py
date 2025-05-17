import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import holidays

# --- Streamlit Page Setup ---
st.set_page_config(page_title="LAMF Simulator", layout="centered",initial_sidebar_state="auto")
st.title("📌 Loan Against Mutual Fund (LAMF) Simulator")

# -------------------------
# LAMF Simulator Section
# -------------------------

st.markdown("""
## LAMF Simulator: Financial Outcome of Taking Loan Against Mutual Funds

This tool simulates the financial outcome of taking a **Loan Against Mutual Funds (LAMF)** and investing that borrowed amount in the market.

It compares:
- 🟥 **Total Loan Outflow** (principal + interest + processing fee)  
- 🟩 **Investment Value** (with monthly compounding)  

---
""")

# --- Inputs ---
st.markdown("### 🔧 Simulation Inputs")

col_loan_start, col_loan_tenure = st.columns(2)

with col_loan_start:
    loan_start_date = st.date_input(
        "📅 Loan Start Date",
        value=dt.date(2025, 3, 27),
        help="Enter the date when the loan started."
    )

with col_loan_tenure:
    loan_tenure_months = st.number_input(
        "🏁 Loan Tenure (Months)",
        min_value=1,
        max_value=36,
        value=36,
        help="Total loan duration agreed with lender (used for foreclosure calculation)."
    )
# --- Helper functions for foreclosure date ---

def is_blackout(date):
    """Returns True if date is in blackout period: 27th to 3rd (inclusive)"""
    return date.day >= 27 or date.day <= 3

def is_valid_foreclosure_day(date, indian_holidays):
    """Returns True if date is a working day, not a holiday, and not in blackout period."""
    return (
        date.weekday() < 5 and  # Monday to Friday
        date not in indian_holidays and
        not is_blackout(date)
    )

def count_working_days(start_date, end_date, holidays_set):
    """Count working days (Mon-Fri excluding holidays) between two dates inclusive."""
    current = start_date
    count = 0
    while current <= end_date:
        if current.weekday() < 5 and current not in holidays_set:
            count += 1
        current += dt.timedelta(days=1)
    return count

# --- Foreclosure Logic ---
def get_foreclosure_date(start_date, tenure_months):
    """
    Find the latest valid foreclosure date such that:
    - Date is a valid foreclosure day (Mon-Fri, no holiday, no blackout)
    - There are at least 5 working days (Mon-Fri excluding holidays) between foreclosure date and loan end date
    - Foreclosure date is at least 7 days after start_date
    """
    end_date = start_date + pd.DateOffset(months=tenure_months)
    earliest_check_date = start_date + dt.timedelta(days=7)

    all_years = list(set([start_date.year, end_date.year]))
    indian_holidays = holidays.India(years=all_years)

    check_date = end_date.date()
    while check_date >= earliest_check_date:
        if is_valid_foreclosure_day(check_date, indian_holidays):
            working_days = count_working_days(check_date, end_date.date(), indian_holidays)
            if working_days >= 7:
                return check_date
        check_date -= dt.timedelta(days=1)

    return None

col_amounts, col_rates = st.columns(2)

from datetime import datetime, date

with col_amounts:
    # Loan and fee inputs
    loan_amount = st.number_input(
        "🏦 Loan Amount (₹)", min_value=25_000, max_value=20_00_000, step=5_000, value=1_00_000,
        help="Specify the loan amount you want to borrow against your mutual funds."
    )
    processing_fee = st.number_input(
        "💰 Processing Fee (₹)", min_value=0, max_value=10_000, step=10, value=1179,
        help="Enter the one-time processing fee charged for the loan."
    )

    # Compute foreclosure date
    foreclosure_date = get_foreclosure_date(loan_start_date, loan_tenure_months)

    # Ensure 'today' and 'foreclosure_date' are date objects
    today = datetime.today().date()
    if isinstance(foreclosure_date, datetime):
        foreclosure_date = foreclosure_date.date()

    # Determine default tenure
    if foreclosure_date:
        delta_days = (foreclosure_date - today).days
        default_tenure_months = max(2, delta_days // 30)
    else:
        default_tenure_months = 12

    # Tenure input
    tenure_months = st.number_input(
        "⏳ Investment Holding Period (Months)",
        min_value=2,
        max_value=36,
        step=1,
        value=default_tenure_months,
        help="Number of months you plan to hold the loan to generate returns."
    )

with col_rates:
    interest_rate = st.number_input(
        "💸 Loan Interest Rate (Annual %)", min_value=4.0, max_value=18.0, step=0.25, value=10.5,
        help="Select the annual interest rate charged on the loan."
    )
    expected_annual_return = st.number_input(
        "📈 Expected Market Return (Annual %)", min_value=0.0, max_value=200.0, step=0.25, value=12.0,
        help="Annual return rate you expect from investing the loaned amount."
    )

# --- Financial Calculations ---
monthly_interest_rate = interest_rate / 12 / 100
monthly_return_rate = (1 + expected_annual_return / 100) ** (1 / 12) - 1

# Total interest assumes simple interest on principal over tenure
total_interest_paid = loan_amount * monthly_interest_rate * tenure_months
total_outflow = loan_amount + total_interest_paid + processing_fee

# Investment value grows with monthly compounding
investment_value = loan_amount * ((1 + monthly_return_rate) ** tenure_months)

net_profit_loss = investment_value - total_outflow
decision_text = "✅ YES, Take LAMF" if net_profit_loss > 0 else "❌ NO, Not Worth It"
    
# --- Display Results Table ---
st.markdown("### 📊 Simulation Results")

# Format dates correctly
formatted_start_date = loan_start_date.strftime("%d-%b-%Y")
formatted_foreclosure_date = foreclosure_date.strftime("%d-%b-%Y") if foreclosure_date else "N/A"

# --- Results Dictionary with Labels ---
results = {
    "📅 Loan Start Date": formatted_start_date,
    "🔓 Foreclosure Date": formatted_foreclosure_date,
    "💵 Loan Amount (₹)": loan_amount,
    "💸 Processing Fee (₹)": processing_fee,
    "📈 Interest Rate (Annual)": f"{interest_rate:.2f}%",
    "📆 Monthly Interest Rate": f"{monthly_interest_rate * 100:.3f}%",
    "📊 Expected Return (Annual)": f"{expected_annual_return:.2f}%",
    "📅 Monthly Return Rate": f"{monthly_return_rate * 100:.3f}%",
    "⏳ Loan Tenure Left": f"{tenure_months} months",
    "💰 Total Interest Paid (₹)": total_interest_paid,
    "📤 Total Outflow (₹)": total_outflow,
    "📈 Investment Value at Maturity (₹)": investment_value,
    "📉 Net Profit / Loss (₹)": net_profit_loss,
    "✅ Decision": decision_text
}

# --- Formatting Function ---
def format_currency(value):
    if isinstance(value, (int, float)):
        abs_val = abs(value)
        if abs_val >= 1_00_00_000:
            return f"₹{value/1_00_00_000:.2f} Crores"
        elif abs_val >= 1_00_000:
            return f"₹{value/1_00_000:.2f} Lakhs"
        else:
            return f"₹{value:,.2f}"
    return value

# Format currency fields selectively
formatted_results = {
    k: format_currency(v) if not isinstance(v, str) or "₹" in k else v
    for k, v in results.items()
}

# --- Create and Display DataFrame ---
df_results = pd.DataFrame.from_dict(formatted_results, orient='index', columns=['Value'])
st.dataframe(df_results, use_container_width=True)

# --- Final Emotional Verdict ---
st.markdown("### 🧠 Final Verdict")

formatted_profit = format_currency(net_profit_loss)

if net_profit_loss > 0:
    st.success(f"✅ Gain of {formatted_profit} — **Worth considering LAMF!**")
else:
    st.error(f"⚠️ Loss of {format_currency(abs(net_profit_loss))} — **Better avoid LAMF under these terms.**")
    
summary_text = f"""
Let's break this down:

You plan to borrow ₹{loan_amount:,.0f} starting from {formatted_start_date}, with a loan tenure of {loan_tenure_months} months.  
At an annual interest rate of {interest_rate:.2f}%. Over the loan period, you will pay a total interest of approximately {format_currency(total_interest_paid)}, plus a processing fee of {format_currency(processing_fee)}.  
This means your total outflow (principal + interest + fees) will be around {format_currency(total_outflow)}.  

Assuming you reinvest the loan amount, expecting an annual return of {expected_annual_return:.2f}%, your investment could grow to {format_currency(investment_value)} by the end of the tenure.  

This results in a net {"profit" if net_profit_loss >= 0 else "loss"} of {format_currency(abs(net_profit_loss))}.  

So, is it financially sound to proceed? {decision_text}

Remember, these are projections based on current assumptions. Real markets fluctuate, but this gives you a realistic outlook to help you make an informed decision.

{f"Foreclosure is scheduled for {formatted_foreclosure_date}." if foreclosure_date else ""}
"""

st.markdown("### 📋 Summary of Your Loan & Investment Simulation")
st.markdown(summary_text)

# --- Foreclosure Date Output ---
if foreclosure_date:
    st.success(f"📅 The foreclosure date is {foreclosure_date.strftime('%A, %d %B %Y')}")
else:
    st.error("Could not find a valid foreclosure date within the loan tenure period.")

# --- Bar Chart: Visual Comparison ---
st.markdown("### 📈 Visual Comparison")

plt.style.use("seaborn-v0_8-muted")
fig, ax = plt.subplots(figsize=(4.5, 6.25))

labels = ["Investment Value", "Total Outflow", "Net P&L"]
values = [investment_value, total_outflow, net_profit_loss]
values_in_lakhs = [v / 1_00_000 for v in values]
colors = ['green', 'red', 'green' if net_profit_loss > 0 else 'red']

bars = ax.bar(labels, values_in_lakhs, color=colors)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.001,
            f"₹{abs(values[i]) / 1_00_000:.2f}L",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Investment vs Outflow vs Net Profit/Loss", fontsize=14, fontweight='bold')
ax.set_ylabel("₹ (in Lakhs)")
ax.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
st.pyplot(fig)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Sensitivity Plot: Net P&L vs Loan Amount ---
st.markdown("### 🔍 Sensitivity: Net P&L vs Loan Amount")

# --- Input Variables ---
loan_amount = 500000
interest_rate = 10.0
expected_annual_return = 12.0
tenure_months = 12
processing_fee = 1000

monthly_interest_rate = interest_rate / 12 / 100
monthly_return_rate = expected_annual_return / 12 / 100

loan_range = np.concatenate([
    np.arange(25000, 100001, 25000),
    np.arange(150000, 350001, 50000),
    np.arange(400000, 1000001, 100000),
    np.arange(1250000, 2000001, 250000)
])

net_pnl_list = []
for loan in loan_range:
    total_interest = loan * monthly_interest_rate * tenure_months
    total_cost = loan + total_interest + processing_fee
    investment = loan * ((1 + monthly_return_rate) ** tenure_months)
    net_pnl = investment - total_cost
    net_pnl_list.append(net_pnl)

def format_rupee(x, _):
    abs_x = abs(x)
    if abs_x < 100000:
        return f"₹{x/1000:.1f}k".rstrip('0').rstrip('.')
    else:
        return f"₹{x/100000:.1f}L".rstrip('0').rstrip('.')

# Plot using indices for equidistant bars
indices = np.arange(len(loan_range))

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    indices,
    net_pnl_list,
    width=0.6,
    color=['green' if val >= 0 else 'red' for val in net_pnl_list]
)

# Annotate each bar
for i, (bar, value) in enumerate(zip(bars, net_pnl_list)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (2500 if value >= 0 else -2500),
        format_rupee(value, None),
        ha='center', va='bottom' if value >= 0 else 'top',
        fontsize=8, fontweight='bold'
    )

# Highlight user's loan
user_index = np.argmin(np.abs(loan_range - loan_amount))
bars[user_index].set_edgecolor("orange")
bars[user_index].set_linewidth(2)

# Axis labels and title
ax.set_title("Net Profit / Loss vs Loan Amount", fontsize=14, fontweight='bold')
ax.set_ylabel("Net P&L", fontsize=12)
ax.set_xlabel("Loan Amount", fontsize=12)
y_max = max(net_pnl_list) * 1.1
ax.set_ylim(0, y_max)

# Custom ticks and formatting
ax.set_xticks(indices)
ax.set_xticklabels([format_rupee(x, None) for x in loan_range], rotation=45, ha='right')
ax.yaxis.set_major_formatter(FuncFormatter(format_rupee))

# Input summary
input_summary = (
    f"Input Variables:\n"
    f"Loan: ₹{loan_amount:,.0f}\n"
    f"Interest: {interest_rate:.1f}% p.a.\n"
    f"Tenure: {tenure_months} months\n"
    f"Return: {expected_annual_return:.1f}% p.a.\n"
    f"Fee: ₹{processing_fee}"
)

ax.text(
    0.01, 0.98, input_summary,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8)
)

ax.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.tight_layout()

st.pyplot(fig)
# --- Educational Guide ---
st.markdown("---")
st.markdown("""
### 📘 How This Works
1. You borrow a sum against your mutual funds (collateralized, no liquidation).
2. You **pay interest monthly** and **repay principal before the last month**.
3. You **invest the borrowed amount** expecting monthly compounded returns.
4. At the end, we calculate:
    - 🟥 Total money *you paid* (outflow)
    - 🟩 Final investment value
5. If 🟩 > 🟥 → **Profit** 💰  
   If 🟥 > 🟩 → **Loss** 😓

---

> 📌 **Note:** This tool assumes reinvestment in funds or equity markets you trust—ones that you believe have a track record of consistent, strong returns, not solely based on past performance. Remember, investing always carries risk, so please invest thoughtfully and within your comfort zone.
""")
