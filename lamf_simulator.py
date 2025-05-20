import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import datetime as dt
import holidays
from datetime import datetime, date
import io
import base64

# --- Streamlit Page Setup ---
st.set_page_config(page_title="LAMF Simulator", layout="centered", initial_sidebar_state="auto")
st.title("📌 Loan Against Mutual Fund (LAMF) Simulator")

# --- Intro ---
st.markdown("""
## LAMF Simulator: Financial Outcome of Taking Loan Against Mutual Funds

This tool simulates the financial outcome of taking a **Loan Against Mutual Funds (LAMF)** and investing that borrowed amount in the market.

It compares:
- 🟥 **Total Loan Outflow** (principal + interest + processing fee)  
- 🟩 **Investment Value** (with monthly compounding)  

---
""")

# --- Input Header ---
st.markdown("### 🔧 Simulation Inputs")

# --- Loan Date and Tenure ---
col1, col2 = st.columns(2)
with col1:
    loan_start_date = st.date_input(
        "📅 Loan Start Date",
        value=dt.date(2025, 3, 27),
        help="Enter the date when the loan started."
    )
with col2:
    loan_tenure_months = st.number_input(
        "🏁 Loan Tenure (Months)",
        min_value=1,
        max_value=36,
        value=36,
        help="Total loan duration agreed with lender (used for foreclosure calculation)."
    )

# --- Foreclosure Logic ---
def is_blackout(date):
    return date.day >= 27 or date.day <= 3

def is_valid_foreclosure_day(date, indian_holidays):
    return (
        date.weekday() < 5 and
        date not in indian_holidays and
        not is_blackout(date)
    )

def count_working_days(start_date, end_date, holidays_set):
    current = start_date
    count = 0
    while current <= end_date:
        if current.weekday() < 5 and current not in holidays_set:
            count += 1
        current += dt.timedelta(days=1)
    return count

def get_foreclosure_date(start_date, tenure_months):
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

# --- Calculate Foreclosure Date ---
foreclosure_date = get_foreclosure_date(loan_start_date, loan_tenure_months)
today = datetime.today().date()
if isinstance(foreclosure_date, datetime):
    foreclosure_date = foreclosure_date.date()

# --- Default Tenure ---
if foreclosure_date:
    delta_days = (foreclosure_date - today).days
    default_tenure_months = max(2, delta_days // 30)
else:
    default_tenure_months = 12

# --- Dasara Date Logic ---
dasara_date = date(2025, 10, 12)
dasara_months = (dasara_date - today).days // 30
dasara_months = min(max(dasara_months, 2), 36)

if "tenure_months" not in st.session_state:
    st.session_state.tenure_months = default_tenure_months

# --- Loan Details: Amount, Fee, Dasara Button ---
fee_col, button_col = st.columns([2, 3])
with fee_col:
    loan_amount = st.number_input(
        "🏦 Loan Amount (₹)", min_value=25_000, max_value=40_00_000, step=5_000, value=3_00_000,
        help="Specify the loan amount you want to borrow against your mutual funds."
    )
    processing_fee = st.number_input(
        "💰 Processing Fee (₹)", min_value=0, max_value=10_000, step=10, value=1179,
        help="Enter the one-time processing fee charged for the loan."
    )
with button_col:
    st.markdown("### ")  # Vertical space
    if st.button(f"🎯 Set Tenure to Dasara ({dasara_months} months left)"):
        st.session_state.tenure_months = dasara_months

# --- Investment Tenure ---
st.number_input(
    "⏳ Investment Holding Period (Months)",
    min_value=2,
    max_value=36,
    step=1,
    value=st.session_state.tenure_months,
    key="tenure_months",
    help="Number of months you plan to hold the loan to generate returns."
)

# --- Interest and Return Rates ---
col3, col4 = st.columns(2)
with col3:
    interest_rate = st.number_input(
        "💸 Loan Interest Rate (Annual %)", min_value=4.0, max_value=18.0, step=0.25, value=10.5,
        help="Select the annual interest rate charged on the loan."
    )
with col4:
    expected_annual_return = st.number_input(
        "📈 Expected Market Return (Annual %)", min_value=0.0, max_value=200.0, step=0.25, value=100.0,
        help="Annual return rate you expect from investing the loaned amount."
    )
# After user inputs section ends
tenure_months = st.session_state.get("tenure_months", default_tenure_months)

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
fig, ax = plt.subplots(figsize=(4.5, 5.0),dpi=600)

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
# Save the figure to a BytesIO buffer
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)

# Encode buffer to base64
b64 = base64.b64encode(buf.read()).decode()
href = f'<a href="data:image/png;base64,{b64}" download="visual_comparison.png">📥 Download Chart as PNG</a>'

# Show the download link
st.markdown(href, unsafe_allow_html=True)

# --- Sensitivity Plot: Net P&L vs Loan Amount ---
st.markdown("### 🔍 Sensitivity: Net P&L vs Loan Amount")

# --- Input Variables ---

loan_range = np.concatenate([
    np.arange(25000, 100001, 25000),
    np.arange(150000, 350001, 50000),
    np.arange(400000, 1000001, 100000),
    np.arange(1250000, 2000001, 250000),
    np.arange(2500000, 3500001, 500000)
])

loan_range = np.unique(np.append(loan_range, loan_amount))
loan_range.sort()

net_pnl_list = []
for loan in loan_range:
    total_interest = loan * monthly_interest_rate * tenure_months
    total_cost = loan + total_interest + processing_fee
    investment = loan * ((1 + monthly_return_rate) ** tenure_months)
    net_pnl = investment - total_cost
    net_pnl_list.append(net_pnl)

def format_rupee(x, _=None):
    abs_x = abs(x)
    if abs_x >= 10000000:
        return f"₹{x / 10000000:.1f}Cr".rstrip('0').rstrip('.')
    elif abs_x >= 100000:
        return f"₹{x / 100000:.1f}L".rstrip('0').rstrip('.')
    else:
        return f"₹{x / 1000:.1f}k".rstrip('0').rstrip('.')

# Plot using indices for equidistant bars
indices = np.arange(len(loan_range))

fig, ax = plt.subplots(figsize=(12, 6),dpi=1000)
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
        value + (3000 if value >= 0 else -3000),
        format_rupee(value, None),
        ha='center', va='bottom' if value >= 0 else 'top',
        fontsize=8, fontweight='bold'
    )

# Highlight user's loan bar
user_index = np.argmin(np.abs(loan_range - loan_amount))
highlight_bar = bars[user_index]

highlight_bar.set_width(0.6)            
highlight_bar.set_edgecolor((1.0, 0.843, 0.0, 1.0))  # RGBA glitter gold
highlight_bar.set_linewidth(3)
highlight_bar.set_hatch(".*")           # Simulates a glittery pattern

# Axis labels and title
ax.set_title("Net Profit / Loss vs Loan Amount", fontsize=14, fontweight='bold')
ax.set_ylabel("Net P&L", fontsize=12)
ax.set_xlabel("Loan Amount", fontsize=12)
y_max = max(net_pnl_list) * 1.05
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
# Save figure to buffer
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=1000, bbox_inches="tight")
buf.seek(0)

# Encode to base64
b64 = base64.b64encode(buf.read()).decode()
href = f'<a href="data:image/png;base64,{b64}" download="net_pnl_vs_loan.png">📥 Download Plot as PNG</a>'

# Show download link
st.markdown(href, unsafe_allow_html=True)

from scipy.optimize import fsolve

st.markdown("### 🔥 FIRE Target Simulation")
fire_target = st.number_input(
    "🎯 Enter Desired Net P&L (FIRE Goal in ₹)",
    min_value=-1_00_00_000, max_value=5_00_00_000,
    step=50_000, value=1_90_00_000, # ₹1,87,78,023
    help="This is your desired gain from taking a loan and investing it. We will estimate the required loan amount to achieve this."
)

# Define the net P&L function to zero-in on
def net_pnl_solver(loan_guess):
    loan = loan_guess
    total_interest = loan * monthly_interest_rate * tenure_months
    total_cost = loan + total_interest + processing_fee
    investment = loan * ((1 + monthly_return_rate) ** tenure_months)
    return investment - total_cost - fire_target

# Use fsolve to find the loan amount that gives the desired net P&L
initial_guess = 1_00_000
try:
    fire_loan_amount = float(fsolve(net_pnl_solver, initial_guess)[0])
    fire_loan_amount = round(fire_loan_amount, -2)  # round to nearest ₹100
except Exception:
    fire_loan_amount = None

if fire_loan_amount > 0:
    st.success(
        f"💡 To achieve a Net P&L of {format_currency(fire_target)}, "
        f"you need to borrow approximately **{format_currency(fire_loan_amount)}**."
    )
    st.caption("Note: This assumes fixed processing fee and your current input rates.")
else:
    st.error(
        f"❌ Unable to compute a valid loan amount for a target Net P&L of {format_currency(fire_target)} "
        f"with the current parameters."
    )

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
