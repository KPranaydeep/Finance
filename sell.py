import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import yfinance as yf
import requests
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📊 Stock Holdings Analysis & Sell Plan with MMI-based Recommendations")

# === Part 1: Load & preprocess MMI dataset ===
@st.cache_data(show_spinner=False)
def load_mmi_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Date', 'MMI', 'Nifty']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df = df[df['MMI'] <= 100].copy()

    for lag in range(1, 8):
        df[f'Lag{lag}'] = df['MMI'].shift(lag)
    for lag in range(1, 4):
        df[f'Nifty_Lag{lag}'] = df['Nifty'].shift(lag)
    df['MMI_Rolling3'] = df['MMI'].rolling(window=3).mean()
    df['MMI_Rolling5'] = df['MMI'].rolling(window=5).mean()
    df['Nifty_Rolling3'] = df['Nifty'].rolling(window=3).mean()
    df['Nifty_Rolling5'] = df['Nifty'].rolling(window=5).mean()

    df.dropna(inplace=True)
    return df

uploaded_mmi_file = st.file_uploader("📂 Upload MMI CSV file", type=['csv'])

if uploaded_mmi_file:
    df_filtered = load_mmi_data(uploaded_mmi_file)

    st.subheader("📈 Enter Today's Market Mood Index (MMI) and Auto-Fetch Nifty")
    try:
        nifty_data = yf.download("^NSEI", period="1d", interval="1m", progress=False)
        latest_close = nifty_data['Close'].dropna().iloc[-1]
        today_nifty = round(float(latest_close), 2)
        st.success(f"📈 Auto-fetched Today's Nifty LTP: **{today_nifty}**")
    except Exception as e:
        today_nifty = st.number_input("Today's Nifty (Auto-fetch failed)", min_value=0.0, step=1.0)
        st.warning("⚠️ Couldn't fetch live Nifty data. Please enter manually.")

    today_mmi = st.number_input("Today's MMI", min_value=0.0, max_value=100.0, step=0.1)
    today = pd.to_datetime('today').normalize()

    if today not in df_filtered.index:
        new_row = pd.DataFrame({'MMI': [today_mmi], 'Nifty': [today_nifty]}, index=[today])
        df_combined = pd.concat([df_filtered[['MMI', 'Nifty']], new_row])
        df_combined.sort_index(inplace=True)

        for lag in range(1, 8):
            df_combined[f'Lag{lag}'] = df_combined['MMI'].shift(lag)
        for lag in range(1, 4):
            df_combined[f'Nifty_Lag{lag}'] = df_combined['Nifty'].shift(lag)
        df_combined['MMI_Rolling3'] = df_combined['MMI'].rolling(window=3).mean()
        df_combined['MMI_Rolling5'] = df_combined['MMI'].rolling(window=5).mean()
        df_combined['Nifty_Rolling3'] = df_combined['Nifty'].rolling(window=3).mean()
        df_combined['Nifty_Rolling5'] = df_combined['Nifty'].rolling(window=5).mean()

        df_combined.dropna(inplace=True)
        df_filtered = df_combined.copy()

    X = df_filtered.drop(columns='MMI')
    y = df_filtered['MMI']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    def nse(obs, sim): return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    def pbias(obs, sim): return 100 * np.sum(obs - sim) / np.sum(obs)
    def rsr(obs, sim): return np.sqrt(np.sum((obs - sim) ** 2)) / np.sqrt(np.sum((obs - np.mean(obs)) ** 2))

    st.markdown("### 📊 Model Evaluation Metrics")
    st.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    st.write(f"NSE: {nse(y_test, y_pred):.4f}, PBIAS: {pbias(y_test, y_pred):.2f}%, RSR: {rsr(y_test, y_pred):.4f}")

    # Forecasting omitted for brevity

# === Part 2: Groww Holdings + Sell Plan ===
st.header("💼 Upload Your Groww Holdings File (.xlsx)")

@st.cache_data
def load_equity_mapping():
    url = "https://raw.githubusercontent.com/KPranaydeep/Finance/refs/heads/main/EQUITY_L.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df[['ISIN NUMBER', 'SYMBOL', 'NAME OF COMPANY']].rename(columns={
        'ISIN NUMBER': 'ISIN',
        'SYMBOL': 'Symbol',
        'NAME OF COMPANY': 'Company Name'
    })

equity_mapping = load_equity_mapping()

uploaded_file = st.file_uploader("📂 Upload your Groww Holdings Excel file", type=['xlsx'])

if uploaded_file:
    try:
        holdings_df = pd.read_excel(uploaded_file, sheet_name='Sheet1', skiprows=9)
        holdings_df = holdings_df.rename(columns={
            'Unnamed: 0': 'Stock Name',
            'Unnamed: 1': 'ISIN',
            'Unnamed: 2': 'Quantity',
            'Unnamed: 3': 'Average Price',
            'Unnamed: 4': 'Buy Value',
            'Unnamed: 5': 'LTP',
            'Unnamed: 6': 'Current Value',
            'Unnamed: 7': 'P&L'
        })
        holdings_df = holdings_df.dropna(subset=['ISIN'])

        # Merge with equity mapping
        merged_df = holdings_df.merge(equity_mapping, on='ISIN', how='left')
        merged_df.dropna(subset=['Symbol'], inplace=True)

        # Fetch latest prices
        st.subheader("🔄 Fetching Latest Market Prices")
        ltp_list = []
        for symbol in merged_df['Symbol']:
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                ltp = ticker.history(period='1d')['Close'].iloc[-1]
                ltp_list.append(ltp)
            except:
                ltp_list.append(None)

        merged_df['LTP'] = ltp_list
        merged_df['Invested Amount'] = merged_df['Quantity'] * merged_df['Average Price']
        merged_df['Current Value'] = merged_df['Quantity'] * merged_df['LTP']
        merged_df['Profit/Loss'] = merged_df['Current Value'] - merged_df['Invested Amount']
        merged_df['Profit/Loss (%)'] = (merged_df['Profit/Loss'] / merged_df['Invested Amount']) * 100

        st.subheader("📈 Holdings Overview with Profit/Loss")
        st.dataframe(merged_df[['Symbol', 'Company Name', 'Quantity', 'Average Price', 'LTP', 'Invested Amount', 'Current Value', 'Profit/Loss', 'Profit/Loss (%)']])

        target_profit = st.number_input("🎯 Enter your target profit percentage (%)", min_value=0.0, step=0.1)
        sell_candidates = merged_df[merged_df['Profit/Loss (%)'] >= target_profit]

        st.subheader("📤 Sell Plan Based on Target Profit")
        if not sell_candidates.empty:
            st.success(f"The following holdings have met or exceeded your target profit of {target_profit}%:")
            st.dataframe(sell_candidates[['Symbol', 'Company Name', 'Quantity', 'Average Price', 'LTP', 'Profit/Loss (%)']])
        else:
            st.info("No holdings have met your target profit yet.")

    except Exception as e:
        st.error(f"❌ An error occurred while processing the file: {e}")
