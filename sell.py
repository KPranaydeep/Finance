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
    if future_df.loc[lowest_mmi_date, 'Predicted_MMI'] < 50:
        st.success(f"📥 **BUY on {lowest_mmi_date.strftime('%d %b %Y')}** – Forecast MMI {future_df.loc[lowest_mmi_date, 'Predicted_MMI']:.2f} < 50")
    else:
        st.warning("❌ No BUY signal – forecast MMI stays above 50")

    if future_df.loc[highest_mmi_date, 'Predicted_MMI'] > 50:
        st.success(f"📤 **SELL on {highest_mmi_date.strftime('%d %b %Y')}** – Forecast MMI {future_df.loc[highest_mmi_date, 'Predicted_MMI']:.2f} > 50")
    else:
        st.warning("❌ No SELL signal – forecast MMI stays below 50")

# === Part 2: Groww Holdings with LTP and Sell Plan ===
st.header("💼 Upload Your Groww Holdings File (.xlsx)")

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

uploaded_holdings = st.file_uploader("📂 Upload your Groww holdings file", type=['xlsx'])

if uploaded_holdings:
    try:
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

        st.markdown("### 🧾 Your Holdings (Groww)")
        st.dataframe(df[['Stock Name', 'ISIN', 'Quantity', 'Average Price', 'Buy Value', 'LTP', 'Current Value', 'P&L']])

        # Enhance: Join with symbol mapping
        merged_df = df.merge(equity_mapping, on='ISIN', how='left')
        merged_df.dropna(subset=['Symbol'], inplace=True)

        st.subheader("🔄 Fetching Live Prices from NSE")
        ltp_list = []
        for symbol in merged_df['Symbol']:
            ltp = None
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                ltp_data = ticker.history(period='1d')
                if not ltp_data.empty:
                    ltp = ltp_data['Close'].iloc[-1]
            except:
                pass
            if ltp is None:
                try:
                    ticker = yf.Ticker(f"{symbol}.BO")
                    ltp_data = ticker.history(period='1d')
                    if not ltp_data.empty:
                        ltp = ltp_data['Close'].iloc[-1]
                except:
                    pass
            ltp_list.append(ltp)

        merged_df['Live LTP'] = ltp_list

        merged_df['Invested Amount'] = merged_df['Quantity'] * merged_df['Average Price']
        merged_df['Current Value'] = merged_df['Quantity'] * merged_df['Live LTP']
        merged_df['Profit/Loss'] = merged_df['Current Value'] - merged_df['Invested Amount']
        merged_df['Profit/Loss (%)'] = (merged_df['Profit/Loss'] / merged_df['Invested Amount']) * 100

        total_invested = merged_df['Invested Amount'].sum()
        default_target = round(total_invested * 0.06, 2)
        target_rupees = st.number_input("🎯 Enter target booking profit (₹)", value=default_target, min_value=0.0, step=100.0)

        merged_df = merged_df.sort_values('Profit/Loss', ascending=False).copy()
        merged_df['Cumulative P&L'] = merged_df['Profit/Loss'].cumsum()
        sell_plan = merged_df[merged_df['Cumulative P&L'] <= target_rupees]
        if not sell_plan.empty:
            final_row = merged_df[merged_df['Cumulative P&L'] >= target_rupees].head(1)
            sell_plan = pd.concat([sell_plan, final_row])

        st.subheader("📤 Suggested Sell Plan to Book Target Profit")
        if not sell_plan.empty:
            st.success(f"To book ₹{target_rupees}, sell these holdings:")
            st.dataframe(sell_plan[['Symbol', 'Company Name', 'Quantity', 'Average Price', 'Live LTP', 'Profit/Loss', 'Cumulative P&L']])
        else:
            st.info("No holdings available to meet the target profit.")

    except Exception as e:
        st.error(f"❌ Could not process file: {e}")
