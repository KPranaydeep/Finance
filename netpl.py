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
from datetime import date

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
    except Exception:
        return value

# --- 📈 Linear Regression for P&L Forecast (with guards) ---
def get_regression_prediction(df, deadline):
    # need at least 2 distinct dates to fit a line
    if len(df["Sell date"].dropna().unique()) < 2:
        return None, None, None, None

    origin = df["Sell date"].min()
    X = (df["Sell date"] - origin).dt.days.values.reshape(-1, 1)
    y = df["Cumulative P&L"].values

    # guard: all y equal -> zero slope; return flat projection
    if np.allclose(y, y[0]):
        days_to_goal = (deadline - origin).days
        predicted_value = float(y[0])
        future_dates = pd.date_range(start=origin, end=deadline)
        future_y = np.full_like(future_dates, fill_value=predicted_value, dtype=float)
        class Dummy:
            coef_ = np.array([0.0])
            intercept_ = float(y[0])
        return predicted_value, future_dates, future_y, Dummy()

    model = LinearRegression().fit(X, y)
    days_to_goal = (deadline - origin).days
    predicted_value = float(model.predict(np.array([[days_to_goal]]))[0])

    future_dates = pd.date_range(start=origin, end=deadline)
    future_X = (future_dates - origin).days.values.reshape(-1, 1)
    future_y = model.predict(future_X)

    return predicted_value, future_dates, future_y, model

# --- 🧭 App Configuration ---
st.set_page_config(layout="wide", page_title="📈 P&L Tracker")
st.markdown("#### 📈 Stock P&L Tracker & Projection")  # Smaller than subheader

# --- 📁 File Handling for Cross-Device ---
STORAGE_FILENAME = "stored_pnl_data.xlsx"

@st.cache_data
def load_excel_data(file_bytes):
    # Robust sheet picking: try exact, then partial, else first sheet
    xls = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    sheet_name = None
    if "Trade Level" in xls.sheet_names:
        sheet_name = "Trade Level"
    else:
        candidates = [s for s in xls.sheet_names if "trade" in s.lower()]
        sheet_name = candidates[0] if candidates else xls.sheet_names[0]

    # Many brokers add headers; try a couple of skiprows patterns
    tried = []
    for skip in (30, 0, 1, 2, 3):
        try:
            tried.append(skip)
            df = xls.parse(sheet_name, skiprows=skip)
            # Expect at least these columns to be present in some form
            # If the layout is known, directly rename after slicing
            df = df.iloc[:, :11]
            df.columns = [
                "Stock name", "ISIN", "Quantity", "Buy date", "Buy price", "Buy value",
                "Sell date", "Sell price", "Sell value", "Realised P&L", "Remark"
            ]
            break
        except Exception:
            df = None
    if df is None:
        raise ValueError(f"Could not parse sheet '{sheet_name}'. Tried skiprows={tried}")

    df["Sell date"] = pd.to_datetime(df["Sell date"], dayfirst=True, errors='coerce')
    df["Realised P&L"] = pd.to_numeric(df["Realised P&L"], errors='coerce')
    df = df.dropna(subset=["Sell date", "Realised P&L"])
    if df.empty:
        raise ValueError("No valid rows after parsing dates and P&L.")

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
if "uploaded_data" in st.session_state and "df" not in st.session_state:
    try:
        st.session_state["df"] = load_excel_data(st.session_state["uploaded_data"])
    except Exception as e:
        st.error(f"❌ Failed to parse Excel file: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
    if df.empty:
        st.warning("Uploaded file contains no usable rows after cleaning.")
        st.stop()

    # --- KPIs ---
    total_pnl = df["Realised P&L"].sum()
    best_day = df.groupby("Sell date")["Realised P&L"].sum().max()
    worst_day = df.groupby("Sell date")["Realised P&L"].sum().min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Realised P&L", format_indian_currency(total_pnl))
    c2.metric("Best Day", format_indian_currency(best_day if pd.notna(best_day) else 0))
    c3.metric("Worst Day", format_indian_currency(worst_day if pd.notna(worst_day) else 0))

    # --- 🗓️ Daily aggregation ---
    daily_pnl = df.groupby("Sell date")["Realised P&L"].sum()
    
    # Build a full calendar range
    full_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq="D")
    
    # Reindex → insert non-trading days as NaN, keep actual trade days intact
    daily_pnl = daily_pnl.reindex(full_range)
    
    # --- Calendar Heatmap ---
    with st.expander("📆 Calendar Heatmap of Daily P&L", expanded=True):
        if daily_pnl.dropna().empty:
            st.info("No daily P&L to display on heatmap yet.")
        else:
            from matplotlib.colors import LinearSegmentedColormap
    
            max_abs = max(abs(daily_pnl.min(skipna=True)), daily_pnl.max(skipna=True))
            denom = max_abs if (pd.notna(max_abs) and max_abs > 0) else 1.0
            normalized = daily_pnl / denom
    
            cmap = LinearSegmentedColormap.from_list(
                "RedWhiteGreen", ["red", "white", "green"], N=256
            )
    
            fig1, ax1 = calplot.calplot(
                normalized,
                cmap=cmap,
                suptitle="Daily Realised P&L (Normalized)",
                colorbar=True,
                linewidth=1,
                edgecolor="black",
                how="sum",
                figsize=(16, 2),
            )
            st.pyplot(fig1)

    # --- 📈 Cumulative Realised P&L Over Time ---
    with st.expander("📈 Cumulative Realised P&L Over Time", expanded=True):
        if daily_pnl.index.size == 0:
            st.info("No dates to plot yet.")
        else:
            date_range = pd.date_range(start=daily_pnl.index.min(), end=daily_pnl.index.max())
            daily_cumsum = daily_pnl.fillna(0).reindex(date_range, fill_value=0).cumsum()

            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(daily_cumsum.index, daily_cumsum.values)
            ax2.set_title("Cumulative Realised P&L Over Time")
            ax2.set_ylabel("₹")
            ax2.grid(True)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            st.pyplot(fig2)

    # --- 🎯 Goal Tracking ---
    # Dates (normalize to midnight to avoid off-by-one)
    today = pd.to_datetime("today").normalize()
    month_end = (today.replace(day=1) + pd.offsets.MonthEnd(1)).date()

    start_goal = 73200.0  # base
    start_date = pd.Timestamp(year=today.year if today.month >= 4 else today.year - 1, month=4, day=1)
    days_diff = max((today - start_date).days, 0)
    compounding_factor = 1.01 ** days_diff
    current_goal_default = float(start_goal * compounding_factor)

    st.markdown("#### 🎯 Set Your Net Profit Goal")
    col1, col2 = st.columns(2)
    with col1:
        goal_amount = st.number_input(
            "Enter Goal Amount (₹)",
            min_value=0.0,
            value=float(round(current_goal_default, 2)),
            step=1000.0,
            format="%.2f",
        )
    with col2:
        goal_deadline = st.date_input("Enter Deadline Date", value=month_end)

    if goal_amount and goal_deadline:
        deadline_ts = pd.to_datetime(goal_deadline)

        # Progress until deadline
        progress = df[df["Sell date"] <= deadline_ts]["Realised P&L"].sum()
        # % bounded to [0, 100] for nicer UX even if over/under
        progress_pct = 0.0 if goal_amount <= 0 else float(np.clip((progress / goal_amount) * 100.0, 0, 100))

        # Projection (safe)
        predicted = None
        future_dates, future_y, model = None, None, None
        pred_tuple = get_regression_prediction(df, deadline_ts)
        if all(v is not None for v in pred_tuple):
            predicted, future_dates, future_y, model = pred_tuple

        remaining = (predicted - progress) if predicted is not None else None

        st.info(
            f"✅ Realised P&L till **{deadline_ts.strftime('%a, %d %b %Y')}**: {format_indian_currency(progress)}\n\n"
            f"🎯 Goal: {format_indian_currency(goal_amount)}\n\n"
            f"📈 Progress: {progress_pct:.1f}%\n\n"
            + (f"📊 Predicted P&L by Deadline: {format_indian_currency(predicted)}\n\n" if predicted is not None else "")
            + (f"🧭 Expected Earnings from Now till Deadline: {format_indian_currency(remaining)}\n" if remaining is not None else "")
        )

        # --- Faster, bounded progress animation ---
        import time

        steps = 50
        target = progress_pct / 100.0
        bar = st.progress(0)
        for i in range(steps + 1):
            bar.progress(min(i / steps, target))
            time.sleep(0.3)  # ⏳ adjust speed: smaller = faster, larger = slower

        # --- Goal hit date (guarded) & plot ---
        goal_achieve_date = None
        if model is not None and hasattr(model, "coef_"):
            slope = float(model.coef_[0])
            intercept = float(getattr(model, "intercept_", 0.0))
            origin = df["Sell date"].min()
            if slope != 0:
                days_to_goal_hit = (goal_amount - intercept) / slope
                # compute date and ensure it's a valid timestamp
                try:
                    goal_achieve_date = origin + pd.Timedelta(days=int(days_to_goal_hit))
                except Exception:
                    goal_achieve_date = None

        # Plot only if we have something meaningful
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(df["Sell date"], df["Cumulative P&L"], marker='o', label="Actual P&L", linewidth=2)
        ax3.axhline(progress, linestyle='--', label=f"Progress {format_indian_currency(progress)}")
        ax3.axhline(goal_amount, color="black", linestyle="--", label=f"Goal {format_indian_currency(goal_amount)}")
        deadline_label = deadline_ts.strftime("%A, %d %B %Y")
        ax3.axvline(deadline_ts, color="green", linestyle="--", label=f"Deadline: {deadline_label}")

        if predicted is not None and future_dates is not None and future_y is not None:
            ax3.scatter(deadline_ts, predicted, s=100, label="Predicted P&L")
            ax3.plot(future_dates, future_y, linestyle=':', label="Linear Projection")

        if (goal_achieve_date is not None) and (df["Sell date"].min() <= goal_achieve_date <= deadline_ts):
            goal_label = goal_achieve_date.strftime("%A, %d %B %Y")
            ax3.axvline(goal_achieve_date, color="black", linestyle="--", label=f"Goal Hit: {goal_label}")
            ax3.scatter(goal_achieve_date, goal_amount, s=80)

        ax3.set_title("Cumulative Realised P&L vs Goal")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("₹ P&L")
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        st.pyplot(fig3)

        if (goal_achieve_date is None) or (goal_achieve_date > pd.to_datetime(goal_deadline)):
            st.caption("💡 *Be patient and consistent — you might hit your profit goal next month!* 💪")
