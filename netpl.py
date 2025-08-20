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
from io import BytesIO
import os

# --- 🧽 Suppress font warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams['font.family'] = 'DejaVu Sans'

# --- 📌 Format Indian currency ---
def format_indian_currency(value):
    try:
        value = float(value)
        if value >= 1e7:
            return f"₹{value / 1e7:.1f} Cr"
        elif value >= 1e5:
            return f"₹{value / 1e5:.1f} Lakhs"
        elif value >= 1e3:
            return f"₹{value / 1e3:.1f} K"
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
st.set_page_config(layout="wide", page_title="📈 P&L Tracker")
st.markdown("#### 📈 Stock P&L Tracker & Projection")  # Smaller than subheader

# --- 📁 File Handling for Cross-Device ---
STORAGE_FILENAME = "stored_pnl_data.xlsx"

@st.cache_data
def load_excel_data(file_bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
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
    return df

# Load from storage if available
if os.path.exists(STORAGE_FILENAME) and "uploaded_data" not in st.session_state:
    with open(STORAGE_FILENAME, "rb") as f:
        st.session_state["uploaded_data"] = f.read()
        st.session_state["file_name"] = STORAGE_FILENAME

# Upload File
with st.expander("📁 Upload Excel File", expanded=False):
    uploaded_file = st.file_uploader("Upload your 'Stocks_PnL_Report.xlsx'", type=["xlsx"])
    if uploaded_file:
        file_content = uploaded_file.read()
        st.session_state["uploaded_data"] = file_content
        st.session_state["file_name"] = uploaded_file.name
        with open(STORAGE_FILENAME, "wb") as f:
            f.write(file_content)
        try:
            st.session_state["df"] = load_excel_data(file_content)
            st.success(f"✅ File '{uploaded_file.name}' successfully loaded and data updated.")
        except Exception as e:
            st.error(f"❌ Failed to parse uploaded file: {e}")
    elif "uploaded_data" in st.session_state:
        st.success(f"✅ {st.session_state['file_name']} already loaded.")

# Load DataFrame
if "uploaded_data" in st.session_state:
    if "df" not in st.session_state:
        try:
            st.session_state["df"] = load_excel_data(st.session_state["uploaded_data"])
        except Exception as e:
            st.error(f"❌ Failed to parse Excel file: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]

    # --- 🗓️ Daily aggregation ---
    daily_pnl = df.groupby("Sell date")["Realised P&L"].sum()
    daily_pnl[daily_pnl == 0] = np.nan

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Normalize daily_pnl between -1 and 1 (preserving sign)
    normalized_pnl = daily_pnl.copy()
    max_abs = max(abs(daily_pnl.min()), daily_pnl.max())
    normalized_pnl = daily_pnl / max_abs
    
    # Custom red-to-green colormap (no yellow)
    cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "white", "green"], N=256)
    
    # Plot
    with st.expander("📆 Calendar Heatmap of Daily P&L", expanded=True):
        fig1, ax1 = calplot.calplot(
            normalized_pnl,
            cmap=cmap,
            suptitle='Daily Realised P&L (Normalized)',
            colorbar=True,
            linewidth=1,
            edgecolor='black',
            how='sum',
            figsize=(16, 2)
        )
        st.pyplot(fig1)

    with st.expander("📈 Cumulative Realised P&L Over Time", expanded=True):
        date_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max())
        daily_cumsum = daily_pnl.reindex(date_range, fill_value=0).cumsum()

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(daily_cumsum.index, daily_cumsum.values, color='blue')
        ax2.set_title("Cumulative Realised P&L Over Time")
        ax2.set_ylabel("₹")
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        st.pyplot(fig2)

    # --- 🎯 Goal Tracking ---
    from datetime import date
    
    # Dates (normalize to midnight to avoid off-by-one)
    today = pd.to_datetime("today").normalize()
    month_end = (today.replace(day=1) + pd.offsets.MonthEnd(1)).date()  # st.date_input likes date objects
    
    start_goal = 52621.0  # float for consistent types
    start_date = pd.Timestamp(year=today.year if today.month >= 4 else today.year - 1, month=4, day=1)
    
    # Days between start_date and today (clip at >=0 just in case)
    days_diff = max((today - start_date).days, 0)
    
    # Compounding factor (1% daily growth)
    compounding_factor = 1.01 ** days_diff
    
    # Current goal (float)
    current_goal = float(start_goal * compounding_factor)
    
    st.markdown("#### 🎯 Set Your Net Profit Goal")
    col1, col2 = st.columns(2)
    with col1:
        # ✅ Make all numeric args floats; add a currency-like format
        goal_amount = st.number_input(
            "Enter Goal Amount (₹)",
            min_value=0.0,
            value=float(round(current_goal, 2)),
            step=1000.0,
            format="%.2f",
        )
    with col2:
        # Use a date object as default
        goal_deadline = st.date_input("Enter Deadline Date", value=month_end)
    
    if goal_amount and goal_deadline:
        # Need at least 2 points for regression; otherwise, skip projection safely
        if len(df["Sell date"].unique()) < 2:
            st.warning("Not enough data points to project P&L yet. Add more trades for a projection.")
            progress = df[df["Sell date"] <= pd.to_datetime(goal_deadline)]["Realised P&L"].sum()
            st.info(f"✅ Realised P&L till **{pd.to_datetime(goal_deadline).strftime('%a, %d %b %Y')}**: {format_indian_currency(progress)}")
        else:
            predicted_pnl, future_dates, future_y, model = get_regression_prediction(df, pd.to_datetime(goal_deadline))
            progress = df[df["Sell date"] <= pd.to_datetime(goal_deadline)]["Realised P&L"].sum()
            remaining = predicted_pnl - progress
    
            # Avoid divide-by-zero when goal_amount == 0
            progress_pct = (progress / goal_amount * 100.0) if goal_amount > 0 else 0.0
    
            st.info(f"""
            ✅ Realised P&L till **{pd.to_datetime(goal_deadline).strftime("%a, %d %b %Y")}**: {format_indian_currency(progress)}
            
            🎯 Goal: {format_indian_currency(goal_amount)}
            
            📈 Progress: {progress_pct:.1f}%
            
            📊 Predicted P&L by Deadline: {format_indian_currency(predicted_pnl)}
            
            🧭 Expected Earnings from Now till Deadline: {format_indian_currency(remaining)}
            """)

            # --- 🔄 Animated Progress Bar ---
            import time
            
            progress_bar = st.progress(0)
            
            # Animate smoothly up to progress_pct
            for i in range(int(progress_pct) + 1):
                progress_bar.progress(i / 100.0)
                time.sleep(0.25)
            
            # Final precise percentage (handles decimals like 36.5%)
            progress_bar.progress(progress_pct / 100.0)       
                
            # Compute date when model hits the goal (if slope != 0)
            if model.coef_[0] != 0:
                days_to_goal_achieve = (goal_amount - model.intercept_) / model.coef_[0]
                goal_achieve_date = df["Sell date"].min() + pd.Timedelta(days=int(days_to_goal_achieve))
            else:
                goal_achieve_date = None
    
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            ax3.plot(df["Sell date"], df["Cumulative P&L"], marker='o', label="Actual P&L", linewidth=2)
            ax3.axhline(progress, linestyle='--', label=f"Progress {format_indian_currency(progress)}")
            ax3.axhline(
                goal_amount,
                color="black",     # ✅ make goal line black
                linestyle="--",
                label=f"Goal {format_indian_currency(goal_amount)}"
            )
            deadline_label = pd.to_datetime(goal_deadline).strftime("%A, %d %B %Y")
            ax3.axvline(
                pd.to_datetime(goal_deadline),
                color="green",             # ✅ set to green
                linestyle="--",
                label=f"Deadline: {deadline_label}"
            )
            ax3.scatter(
                pd.to_datetime(goal_deadline),
                predicted_pnl,
                color="green",   # ✅ change from 'orange' to 'green'
                s=100,
                label="Predicted P&L"
            )
            ax3.plot(future_dates, future_y, linestyle=':', label="Linear Projection")
    
            if goal_achieve_date is not None and df["Sell date"].min() <= goal_achieve_date <= pd.to_datetime(goal_deadline):
                goal_label = goal_achieve_date.strftime("%A, %d %B %Y")
                ax3.axvline(
                    goal_achieve_date,
                    color="black",     # ✅ explicitly black
                    linestyle="--",
                    label=f"Goal Hit: {goal_label}"
                )
                ax3.scatter(goal_achieve_date, goal_amount, s=80)
    
            ax3.set_title("Cumulative Realised P&L vs Goal")
            ax3.set_xlabel("Date")
            ax3.set_ylabel("₹ P&L")
            ax3.grid(True, linestyle='--', alpha=0.3)
            ax3.legend()
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            st.pyplot(fig3)
    
            if goal_achieve_date is None or goal_achieve_date > pd.to_datetime(goal_deadline):
                st.caption("💡 *Be patient and consistent — you might hit your profit goal next month!* 💪")
