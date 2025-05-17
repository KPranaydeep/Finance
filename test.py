import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import holidays

# --- Streamlit Page Setup ---
st.set_page_config(page_title="LAMF Simulator & Foreclosure Estimator", layout="wide")
st.title("📌 Loan Against Mutual Fund (LAMF) Simulator & Foreclosure Date Estimator")

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

loan_amount = st.number_input(
    "🏦 Loan Amount (₹)", min_value=25000, max_value=2000000, step=5000, value=100000,
    help="Specify the loan amount you want to borrow against your mutual funds."
)

interest_rate = st.number_input(
    "💸 Loan Interest Rate (Annual %)", min_value=4.0, max_value=18.0, step=0.25, value=10.5,
    help="Select the annual interest rate charged on the loan."
)

processing_fee = st.number_input(
    "💰 Processing Fee (₹)", min_value=0, max_value=10000, step=10, value=1179,
    help="Enter the one-time processing fee that is charged for the loan."
)

expected_annual_return = st.number_input(
    "📈 Expected Market Return (Annual %)", min_value=0.0, max_value=200.0, step=0.25, value=12.0,
    help="Select the annual return rate you expect from investing in the market."
)

tenure_months = st.number_input(
    "⏳ Loan Tenure (Months)", min_value=2, max_value=36, step=1, value=35,
    help="Select the number of months for the loan repayment period."
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

# --- Display Results Table ---
st.markdown("### 📊 Simulation Results")

results = {
    "Loan Amount": loan_amount,
    "Interest Rate (Annual)": f"{interest_rate:.2f}%",
    "Monthly Interest Rate": f"{monthly_interest_rate * 100:.3f}%",
    "Expected Return (Annual)": f"{expected_annual_return:.2f}%",
    "Monthly Return Rate": f"{monthly_return_rate * 100:.3f}%",
    "Loan Tenure (Months)": f"{tenure_months} Months",
    "Processing Fee": processing_fee,
    "Total Interest Paid": total_interest_paid,
    "Total Outflow (Principal + Interest + Fee)": total_outflow,
    "Investment Value at Maturity": investment_value,
    "Net Profit / Loss": net_profit_loss,
    "Decision": decision_text
}

# Format currency fields except for percentages, months, and decision text
formatted_results = {
    k: format_currency(v) if k not in [
        "Loan Tenure (Months)", "Decision",
        "Interest Rate (Annual)", "Monthly Interest Rate",
        "Expected Return (Annual)", "Monthly Return Rate"
    ] else v for k, v in results.items()
}

df_results = pd.DataFrame.from_dict(formatted_results, orient='index', columns=['Value'])
st.dataframe(df_results, use_container_width=True)

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

# --- Final Emotional Verdict ---
st.markdown("### 🧠 Final Verdict")

def format_currency_simple(value):
    abs_val = abs(value)
    if abs_val >= 1_00_000:
        return f"{value/1_00_000:.2f} Lakhs"
    else:
        return f"₹{value:,.0f}"

formatted_profit = format_currency_simple(net_profit_loss)

if net_profit_loss > 0:
    st.success(f"✅ Gain of {formatted_profit} — **Worth considering LAMF!**")
else:
    st.error(f"⚠️ Loss of {format_currency_simple(abs(net_profit_loss))} — **Better avoid LAMF under these terms.**")

# --- Sensitivity Plot: Net P&L vs Loan Amount ---
st.markdown("### 🔍 Sensitivity: Net P&L vs Loan Amount")

loan_range = np.concatenate([
    np.arange(25000, 100001, 25000),
    np.arange(150000, 350001, 50000),
    np.arange(400000, 1000001, 100000)
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
        return f"₹{x/1000:.1f}k"
    else:
        return f"₹{x/100000:.2f}L"

fig3, ax3 = plt.subplots(figsize=(10, 6))
bars = ax3.bar(
    loan_range.astype(str), net_pnl_list,
    color=['green' if val >= 0 else 'red' for val in net_pnl_list]
)

for bar, value in zip(bars, net_pnl_list):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        value + (25 if value >= 0 else -25),
        format_rupee(value, None),
        ha='center', va='bottom' if value >= 0 else 'top',
        fontsize=8, fontweight='bold'
    )

user_index = np.argmin(np.abs(loan_range - loan_amount))
bars[user_index].set_edgecolor("orange")
bars[user_index].set_linewidth(2)

ax3.set_title("Net Profit / Loss vs Loan Amount", fontsize=14, fontweight='bold')
ax3.set_ylabel("Net P&L", fontsize=12)
ax3.set_xlabel("Loan Amount (₹)", fontsize=12)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_rupee))
ax3.grid(True, linestyle='--', alpha=0.5, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()

input_summary = (
    f"Input Variables:\n"
    f"Loan: ₹{loan_amount:,.0f}\n"
    f"Interest: {interest_rate:.1f}% p.a.\n"
    f"Tenure: {tenure_months} months\n"
    f"Return: {expected_annual_return:.1f}% p.a.\n"
    f"Fee: ₹{processing_fee}"
)

ax3.text(
    0.01, 0.98, input_summary,
    transform=ax3.transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.8)
)

st.pyplot(fig3)

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

> 📌 **Note:** This tool assumes reinvestment in high-return mutual funds or equity markets with no withdrawal during tenure.
""")

# -------------------------
# Foreclosure Date Estimator Section
# -------------------------

st.markdown("---")
st.markdown("## 📆 Mutual Fund Loan Foreclosure Estimator (India)")

st.markdown("""
This tool helps you estimate the earliest valid foreclosure date for a mutual fund loan, accounting for:
- Working days (Mon–Fri)
- Indian public holidays
- Monthly blackout period (27th to 3rd)
""")

# --- Helper functions for foreclosure date ---

def is_blackout(date):
    """Returns True if date is in blackout period: 27th to 3rd (inclusive)"""
    return (date.day >= 27 or date.day <= 3)

def is_valid_foreclosure_day(date, indian_holidays):
    """Returns True if date is a working day, not a holiday, and not in blackout period."""
    return (
        date.weekday() < 5 and  # Monday to Friday
        date not in indian_holidays and
        not is_blackout(date)
    )

def get_foreclosure_date(start_date, tenure_months):
    """Find the earliest valid foreclosure date based on start date and tenure."""
    end_date = start_date + pd.DateOffset(months=tenure_months)
    today = dt.date.today()

    # Get Indian holidays for all relevant years
    all_years = list(set([today.year, end_date.year, start_date.year]))
    indian_holidays = holidays.India(years=all_years)

    check_date = today
    while check_date < end_date.date():
        if is_valid_foreclosure_day(check_date, indian_holidays):
            # Only consider dates at least 7 days after today for realistic processing
            if (check_date - today).days >= 7:
                return check_date
        check_date += dt.timedelta(days=1)
    return None

# --- Inputs for Foreclosure Estimator ---
st.markdown("### 📅 Inputs for Foreclosure Estimation")

loan_start_date = st.date_input(
"Loan Start Date",
value=dt.date.today() - pd.DateOffset(months=1),
help="Enter the date when the loan started."
)

loan_tenure_months = st.number_input(
"Loan Tenure (Months)",
min_value=1,
max_value=36,
value=tenure_months,
help="Enter the loan tenure in months."
)

if st.button("Estimate Foreclosure Date"):
  foreclosure_date = get_foreclosure_date(loan_start_date, loan_tenure_months)
if foreclosure_date:
  st.success(f"📅 Earliest valid foreclosure date is {foreclosure_date.strftime('%A, %d %B %Y')}")
else:
  st.error("Could not find a valid foreclosure date within the loan tenure period.")

st.markdown("---")
st.markdown("""

About Blackout Period:
The blackout period is from the 27th of a month to the 3rd of the next month, inclusive.

During this time, loan foreclosure or redemptions are generally restricted.

This calculator accounts for all public holidays and weekends in India, and only returns the earliest valid foreclosure date outside blackout.
""")
