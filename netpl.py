import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# ----------------------------
# Utility Functions
# ----------------------------
def format_indian_currency(amount):
    """Format number in Indian currency style."""
    return f"₹{amount:,.0f}"


def get_regression_prediction(df):
    """Fit linear regression to cumulative profit and return next-day prediction."""
    df = df.reset_index(drop=True)
    df["Day"] = df.index + 1

    X = df[["Day"]]
    y = df["Cumulative Profit"]

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[len(df) + 1]])
    return model.predict(next_day)[0]


# ----------------------------
# Data Loading
# ----------------------------
def load_excel_data(uploaded_file):
    """Load and preprocess Excel data."""
    try:
        df = pd.read_excel(uploaded_file)
        required_cols = ["Date", "Buy", "Sell"]

        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns. Found: {list(df.columns)}")
            return None

        df = df[required_cols]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Profit"] = df["Sell"] - df["Buy"]
        df["Cumulative Profit"] = df["Profit"].cumsum()
        return df

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# ----------------------------
# KPI Display
# ----------------------------
def show_kpis(df, bike_price):
    total_investment = df["Buy"].sum()
    total_profit = df["Profit"].sum()
    cumulative_profit = df["Cumulative Profit"].iloc[-1]
    regression_pred = get_regression_prediction(df)

    st.subheader("📊 Key Metrics")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Investment", format_indian_currency(total_investment))
    kpi_cols[1].metric("Total Profit", format_indian_currency(total_profit))
    kpi_cols[2].metric("Cumulative Profit", format_indian_currency(cumulative_profit))
    kpi_cols[3].metric("Regression Predicted Profit", format_indian_currency(regression_pred))

    return total_profit, regression_pred


# ----------------------------
# Plotting Functions
# ----------------------------
def plot_heatmap(df):
    """Show daily profit heatmap with white for no trades."""
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.strftime("%b")

    pivot_df = df.pivot("Month", "Day", "Profit")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="RdYlGn",
                linewidths=.5, linecolor="gray", cbar=True,
                mask=pivot_df.isna(),  # White cells for no trades
                )
    plt.title("Daily Profit Heatmap", fontsize=16)
    st.pyplot(plt)


def plot_cumulative(df):
    """Plot cumulative profit with regression trend."""
    df = df.reset_index(drop=True)
    df["Day"] = df.index + 1

    X = df[["Day"]]
    y = df["Cumulative Profit"]

    model = LinearRegression()
    model.fit(X, y)
    df["Trendline"] = model.predict(X)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Cumulative Profit"], label="Cumulative Profit", linewidth=2)
    plt.plot(df["Date"], df["Trendline"], linestyle="--", label="Trendline")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit")
    plt.title("Cumulative Profit with Regression Trend")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


# ----------------------------
# Goal Tracking
# ----------------------------
def goal_tracking(total_profit, regression_pred, bike_price):
    st.subheader("🎯 Goal Tracking")

    st.write(f"Target: {format_indian_currency(bike_price)}")
    st.write(f"Current Profit: {format_indian_currency(total_profit)}")
    st.write(f"Predicted Next-Day Profit: {format_indian_currency(regression_pred)}")

    if total_profit >= bike_price:
        st.success("Congratulations! You’ve achieved your bike purchase goal! 🎉")
    else:
        remaining = bike_price - total_profit
        st.info(f"You need {format_indian_currency(remaining)} more to reach your goal.")


# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="P&L Tracker", layout="wide")
    st.title("💰 Trading P&L Tracker")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = load_excel_data(uploaded_file)
        if df is not None:
            st.dataframe(df, use_container_width=True)

            bike_price = 221000  # Example target

            total_profit, regression_pred = show_kpis(df, bike_price)

            plot_heatmap(df)
            plot_cumulative(df)

            goal_tracking(total_profit, regression_pred, bike_price)


if __name__ == "__main__":
    main()
