import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import calplot
import matplotlib as mpl
import warnings
import logging

# --- 🧽 Suppress font warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams['font.family'] = 'DejaVu Sans'

# --- 📌 Format Indian currency ---
def format_indian_currency(value):
    try:
        value = float(value)
        if value >= 1e7:
            return f"₹{value / 1e7:.2f} Cr"
        elif value >= 1e5:
            return f"₹{value / 1e5:.2f} Lakhs"
        else:
            return f"₹{value:,.0f}"
    except:
        return value

# --- 📈 Linear Regression for P&L Forecast ---
def get_regression_prediction(df, deadline):
    X = (df["Sell date"] - df["Sell date"].min()).dt.days.values.reshape(-1, 1)
    y = df["Cumulative P&L"].values
    model = LinearRegression().fit(X, y)

    days_to_goal = (deadline - df["Sell date"].min()).days
    predicted_value = model.predict(np.array([[days_to_goal]]))[0]

    future_dates = pd.date_range(start=df["Sell date"].min(), end=deadline)
    future_X = (future_dates - df["Sell date"].min()).days.values.reshape(-1, 1)
    future_y = model.predict(future_X)

    return predicted_value, future_dates, future_y, model

# --- 🧭 App Configuration ---
st.set_page_config(layout="centered", page_title="📈 P&L Tracker")
st.title("📈 Stock P&L Tracker & Projection")

# --- 📁 File Upload Section ---
with st.expander("📁 Upload Excel File", expanded=False):
    uploaded_file = st.file_uploader("Upload your 'Stocks_PnL_Report.xlsx'", type=["xlsx"])

    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            df = xls.parse("Trade Level", skiprows=30)
            df.columns = [
                "Stock name", "ISIN", "Quantity", "Buy date", "Buy price", "Buy value",
                "Sell date", "Sell price", "Sell value", "Realised P&L", "Remark"
            ]
            df["Sell date"] = pd.to_datetime(df["Sell date"], dayfirst=True, errors='coerce')
            df["Realised P&L"] = pd.to_numeric(df["Realised P&L"], errors='coerce')
            df = df.dropna(subset=["Sell date", "Realised P&L"])
            df = df.sort_values("Sell date")
            df["Cumulative P&L"] = df["Realised P&L"].cumsum()

            # Save only on fresh upload
            st.session_state["df"] = df
            st.success("✅ File uploaded and saved in session!")

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")

# --- 📊 Main Visualisation Block ---
# Use stored dataframe if available
if "df" in st.session_state:
    df = st.session_state["df"]

    # 🗓️ Daily aggregation
    daily_pnl = df.groupby("Sell date")["Realised P&L"].sum()
    daily_pnl[daily_pnl == 0] = np.nan

    # Show Calendar Heatmap
    with st.expander("📆 Calendar Heatmap of Daily P&L", expanded=True):
        fig1, ax1 = calplot.calplot(
            daily_pnl,
            cmap='RdYlGn',
            suptitle='Realised P&L Calendar Heatmap',
            colorbar=True,
            linewidth=1,
            edgecolor='black',
            how='sum',
            figsize=(16, 3)
        )
        st.pyplot(fig1)

    # --- 📈 Cumulative P&L ---
    with st.expander("📈 Cumulative Realised P&L Over Time", expanded=True):
        date_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max())
        daily_cumsum = daily_pnl.reindex(date_range, fill_value=0).cumsum()

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(daily_cumsum.index, daily_cumsum.values, color='blue')
        ax2.set_title("Cumulative Realised P&L Over Time")
        ax2.set_ylabel("₹")
        ax2.grid(True)
        st.pyplot(fig2)

    # --- 🎯 Goal Inputs ---
    st.subheader("🎯 Set Your Realised P&L Goal")
    col1, col2 = st.columns(2)
    with col1:
        goal_amount = st.number_input("Enter Goal Amount (₹)", min_value=0, value=200000, step=10000)
    with col2:
        goal_deadline = st.date_input("Enter Deadline Date", value=pd.to_datetime("2025-12-31"))

    if goal_amount and goal_deadline:
        predicted_pnl, future_dates, future_y, model = get_regression_prediction(df, pd.to_datetime(goal_deadline))
        progress = df[df["Sell date"] <= pd.to_datetime(goal_deadline)]["Realised P&L"].sum()

        st.info(f"""
        ✅ Realised P&L till **{goal_deadline.strftime("%a, %d %b %Y")}**: {format_indian_currency(progress)}  
        🎯 Goal: {format_indian_currency(goal_amount)}  
        📈 Progress: {progress / goal_amount * 100:.1f}%  
        📊 Predicted P&L by Deadline: {format_indian_currency(predicted_pnl)}
        """)

        # --- ⏱ Goal Achievement Estimation ---
        if model.coef_[0] != 0:
            days_to_goal_achieve = (goal_amount - model.intercept_) / model.coef_[0]
            goal_achieve_date = df["Sell date"].min() + pd.Timedelta(days=int(days_to_goal_achieve))
        else:
            goal_achieve_date = None

        # --- 📉 Cumulative P&L vs Goal Plot ---
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(df["Sell date"], df["Cumulative P&L"], marker='o', label="Actual P&L", linewidth=2)
        ax3.axhline(progress, color='blue', linestyle='--', label=f"Progress {format_indian_currency(progress)}")
        ax3.axhline(goal_amount, color='green', linestyle='--', label=f"Goal {format_indian_currency(goal_amount)}")
        formatted_deadline = pd.to_datetime(goal_deadline).strftime("%a, %d %b %Y")
        ax3.axvline(pd.to_datetime(goal_deadline), color='red', linestyle='--', label=f"Deadline: {formatted_deadline}")
        ax3.scatter(pd.to_datetime(goal_deadline), predicted_pnl, color='orange', s=100, label="Predicted P&L")
        ax3.plot(future_dates, future_y, color='gray', linestyle=':', label="Linear Projection")

        if (
            goal_achieve_date is not None and
            df["Sell date"].min() <= goal_achieve_date <= pd.to_datetime(goal_deadline)
        ):
            formatted_goal_date = pd.to_datetime(goal_achieve_date).strftime("%a, %d %b %Y")
            ax3.axvline(goal_achieve_date, color='black', linestyle='--', label=f"Goal Hit: {formatted_goal_date}")
            ax3.scatter(goal_achieve_date, goal_amount, color='black', s=80)

        ax3.set_title("Cumulative Realised P&L vs Goal")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("₹ P&L")
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        fig3.autofmt_xdate()
        st.pyplot(fig3)
else:
    st.info("📂 Please upload your Stocks_PnL_Report.xlsx file to begin.")
