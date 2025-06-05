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

    future_dates = pd.date_range(start=df_filtered.index[-1] + pd.Timedelta(days=1), periods=120, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    last_row = df_filtered.iloc[-1]

    current_lag_values = {f'Lag{i}': last_row[f'Lag{i-1}'] for i in range(2, 8)}
    current_lag_values['Lag1'] = last_row['MMI']
    current_nifty_lag_values = {f'Nifty_Lag{i}': last_row[f'Nifty_Lag{i}'] for i in range(1, 4)}
    last_nifty = last_row['Nifty']
    m3, m5 = [last_row['MMI_Rolling3']] * 2, [last_row['MMI_Rolling5']] * 4

    predictions = []
    for date in future_dates:
        features = {
            **current_lag_values,
            **current_nifty_lag_values,
            'MMI_Rolling3': np.mean(m3[-2:] + [current_lag_values['Lag1']]),
            'MMI_Rolling5': np.mean(m5[-4:] + [current_lag_values['Lag1']]),
            'Nifty_Rolling3': last_row['Nifty_Rolling3'],
            'Nifty_Rolling5': last_row['Nifty_Rolling5']
        }
        features_df = pd.DataFrame([features])
        features_df = features_df.reindex(columns=X.columns, fill_value=0)

        pred = model.predict(features_df)[0]
        predictions.append(pred)

        for i in range(7, 1, -1): current_lag_values[f'Lag{i}'] = current_lag_values[f'Lag{i-1}']
        current_lag_values['Lag1'] = pred

        for i in range(3, 1, -1): current_nifty_lag_values[f'Nifty_Lag{i}'] = current_nifty_lag_values[f'Nifty_Lag{i-1}']
        current_nifty_lag_values['Nifty_Lag1'] = last_nifty

        m3.append(features['MMI_Rolling3']); m3.pop(0)
        m5.append(features['MMI_Rolling5']); m5.pop(0)

    future_df['Predicted_MMI'] = predictions
    residual_std = np.std(y - model.predict(X))

    lowest_mmi_date = future_df['Predicted_MMI'].idxmin()
    highest_mmi_date = future_df['Predicted_MMI'].idxmax()
    ci_low = (future_df.loc[lowest_mmi_date, 'Predicted_MMI'] - 1.96 * residual_std,
              future_df.loc[lowest_mmi_date, 'Predicted_MMI'] + 1.96 * residual_std)
    ci_high = (future_df.loc[highest_mmi_date, 'Predicted_MMI'] - 1.96 * residual_std,
               future_df.loc[highest_mmi_date, 'Predicted_MMI'] + 1.96 * residual_std)

    st.markdown("### 🔮 Forecast Summary (Next 120 Days)")
    st.write(f"🔻 **Lowest MMI:** {future_df.loc[lowest_mmi_date, 'Predicted_MMI']:.2f} on {lowest_mmi_date.strftime('%d %b %Y')} (CI: {ci_low[0]:.2f}–{ci_low[1]:.2f})")
    st.write(f"🔺 **Highest MMI:** {future_df.loc[highest_mmi_date, 'Predicted_MMI']:.2f} on {highest_mmi_date.strftime('%d %b %Y')} (CI: {ci_high[0]:.2f}–{ci_high[1]:.2f})")

    st.markdown("### 💡 MMI-Based Trading Recommendations")

    # Add today's MMI to the forecast dataframe
    today_row = pd.DataFrame({
        'Predicted_MMI': [today_mmi]
    }, index=[pd.Timestamp.today().normalize()])  # today at 00:00
    
    # Combine today + forecast
    combined_df = pd.concat([today_row, future_df])
    
    # Find lowest and highest MMI (from today onward)
    min_mmi_date = combined_df['Predicted_MMI'].idxmin()
    min_mmi_value = combined_df.loc[min_mmi_date, 'Predicted_MMI']
    
    max_mmi_date = combined_df['Predicted_MMI'].idxmax()
    max_mmi_value = combined_df.loc[max_mmi_date, 'Predicted_MMI']
    
    # === BUY Signal ===
    if min_mmi_value < 50:
        st.success(
            f"📥 **BUY on {min_mmi_date.strftime('%d %b %Y')}** – MMI {min_mmi_value:.2f} < 50"
        )
    else:
        st.warning(
            f"❌ No BUY Signal – Minimum MMI is {min_mmi_value:.2f} on {min_mmi_date.strftime('%d %b %Y')}"
        )
    
    # === SELL Signal ===
    if today_mmi > 50 and max_mmi_value > 50:
        st.success(
            f"📤 **SELL on {max_mmi_date.strftime('%d %b %Y')}** – MMI {max_mmi_value:.2f} > 50"
        )
    else:
        st.warning(
            f"❌ No SELL Signal – Max MMI is {max_mmi_value:.2f} on {max_mmi_date.strftime('%d %b %Y')}, Today's MMI: {today_mmi:.2f}"
        )

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

        
        
