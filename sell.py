import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import requests
from io import BytesIO

st.set_page_config(layout="wide", page_icon=":moneybag:")
st.title("📊 Stock Holdings Analysis & Sell Plan")

# === Page Setup ===
st.header("💼 Upload Your Groww Holdings File (.xlsx)")

# === Load ISIN to Symbol Mapping ===
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

# === Upload Holdings File ===
uploaded_holdings = st.file_uploader("📂 Upload your Groww holdings file", type=['xlsx'])

if uploaded_holdings:
    try:
        # === Read and Clean Holdings File ===
        df = pd.read_excel(uploaded_holdings, sheet_name='Sheet1', skiprows=9)
        # Drop the first row if it contains header-like strings (double header issue)
        if df.iloc[0].astype(str).str.contains("Stock Name", case=False).any():
            df = df.iloc[1:]
        df = df.rename(columns={
            'Unnamed: 0': 'Stock Name',
            'Unnamed: 1': 'ISIN',
            'Unnamed: 2': 'Quantity',
            'Unnamed: 3': 'Average Price',
            'Unnamed: 4': 'Buy Value',
            'Unnamed: 5': 'LTP',
            'Unnamed: 6': 'Current Value',
            'Unnamed: 7': 'P&L'
        })
        df = df.dropna(subset=['Stock Name', 'ISIN'])

        st.markdown("### 🧾 Your Holdings (Groww Upload)")
        st.dataframe(df[['Stock Name', 'ISIN', 'Quantity', 'Average Price', 'Buy Value', 'LTP', 'Current Value', 'P&L']])

        # === Merge with Equity Mapping ===
        merged_df = df.merge(equity_mapping, on='ISIN', how='left')
        merged_df.dropna(subset=['Symbol'], inplace=True)

        # === Fetch Live LTP ===
        st.subheader("🔄 Fetching Live Prices from NSE/BSE...")
        ltp_list = []
        for symbol in merged_df['Symbol']:
            ltp = None
            for suffix in ['.NS', '.BO']:
                try:
                    ticker = yf.Ticker(symbol + suffix)
                    ltp_data = ticker.history(period='1d')
                    if not ltp_data.empty:
                        ltp = ltp_data['Close'].iloc[-1]
                        break
                except:
                    continue
            ltp_list.append(ltp)

        merged_df['Live LTP'] = ltp_list
        merged_df.dropna(subset=['Live LTP'], inplace=True)

        # === Calculate Metrics ===
        merged_df['Invested Amount'] = merged_df['Quantity'] * merged_df['Average Price']
        merged_df['Current Value'] = merged_df['Quantity'] * merged_df['Live LTP']
        merged_df['Profit/Loss'] = merged_df['Current Value'] - merged_df['Invested Amount']
        merged_df['Profit/Loss (%)'] = (merged_df['Profit/Loss'] / merged_df['Invested Amount']) * 100

        total_invested = merged_df['Invested Amount'].sum()
        total_current_value = merged_df['Current Value'].sum()
        total_pl = merged_df['Profit/Loss'].sum()

        # === Display Summary ===
        st.markdown("### 📊 Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
        col2.metric("📈 Total Current Value", f"₹{total_current_value:,.2f}")
        col3.metric("📊 Overall P&L", f"₹{total_pl:,.2f}", delta=f"{(total_pl/total_invested)*100:.2f}%")

        # === Explain Target ===
        st.markdown("💡 *To achieve 100% CAGR (doubling in 1 year), you need ~0.34% daily gross return. After 15% trading charges, your net return is ~0.28% per day.*")
        st.markdown("📌 *This tool helps you plan daily profit booking using that logic.*")

        # === Enter Daily Target Profit ===
        default_target = round(total_invested * 0.0034, 2)
        target_rupees = st.number_input("🎯 Enter today's target booking profit (₹)", value=default_target, min_value=0.0, step=100.0)

        # === Sell Plan Logic ===
        merged_df = merged_df.sort_values(by='Profit/Loss (%)', ascending=False).reset_index(drop=True)

        cumulative_profit = 0
        selected_rows = []

        for _, row in merged_df.iterrows():
            if cumulative_profit >= target_rupees:
                break
            cumulative_profit += row['Profit/Loss']
            selected_rows.append(row)

        if cumulative_profit >= target_rupees and selected_rows:
            sell_plan = pd.DataFrame(selected_rows).copy()
            sell_plan['Sell Limit (₹)'] = (sell_plan['Live LTP'] * 1.0034).round(2)

            sell_plan = sell_plan[[
                'Symbol', 'Company Name', 'Quantity', 'Average Price',
                'Live LTP', 'Sell Limit (₹)', 'Profit/Loss', 'Profit/Loss (%)'
            ]]
            sell_plan[['Average Price', 'Live LTP', 'Sell Limit (₹)', 'Profit/Loss']] = sell_plan[[
                'Average Price', 'Live LTP', 'Sell Limit (₹)', 'Profit/Loss'
            ]].round(2)
            sell_plan['Profit/Loss (%)'] = sell_plan['Profit/Loss (%)'].round(2)

            st.subheader("📤 Suggested Sell Plan to Book Target Profit")
            st.success(f"✅ To book ₹{target_rupees:.2f}, sell the following holdings:")
            st.dataframe(sell_plan, use_container_width=True)

        else:
            st.warning("📉 No sufficient profitable stocks available to book the target profit.")
            st.info("⏳ Come back tomorrow — the market may rise and help you hit your target!")

    except Exception as e:
        st.error(f"❌ Could not process file: {e}")

        
        
