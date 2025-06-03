import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
import yfinance as yf

st.title("Stock Holdings Analysis & Sell Plan with MMI-based Recommendations")

# === Part 1: Load & preprocess MMI dataset ===
@st.cache_data(show_spinner=False)
def load_mmi_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Date', 'MMI', 'Nifty']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df_filtered = df[df['MMI'] <= 100].copy()
    
    # Lag features for MMI
    for lag in range(1, 8):
        df_filtered[f'Lag{lag}'] = df_filtered['MMI'].shift(lag)
    # Lag features for Nifty
    for lag in range(1, 4):
        df_filtered[f'Nifty_Lag{lag}'] = df_filtered['Nifty'].shift(lag)
    # Rolling means
    df_filtered['MMI_Rolling3'] = df_filtered['MMI'].rolling(window=3).mean()
    df_filtered['MMI_Rolling5'] = df_filtered['MMI'].rolling(window=5).mean()
    df_filtered['Nifty_Rolling3'] = df_filtered['Nifty'].rolling(window=3).mean()
    df_filtered['Nifty_Rolling5'] = df_filtered['Nifty'].rolling(window=5).mean()
    df_filtered.dropna(inplace=True)
    return df_filtered

# Replace this with your actual file upload or local path
uploaded_mmi_file = st.file_uploader("Upload MMI CSV file", type=['csv'])
if uploaded_mmi_file:
    df_filtered = load_mmi_data(uploaded_mmi_file)

    # Features and target
    feature_cols = [col for col in df_filtered.columns if col != 'MMI']
    X = df_filtered[feature_cols]
    y = df_filtered['MMI']

    # Train-test split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)

    # Predictions & metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    def nse(obs, sim):
        return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    def pbias(obs, sim):
        return 100 * np.sum(obs - sim) / np.sum(obs)

    def rsr(obs, sim):
        return np.sqrt(np.sum((obs - sim) ** 2)) / np.sqrt(np.sum((obs - np.mean(obs)) ** 2))

    nse_score = nse(y_test.values, y_pred)
    pbias_score = pbias(y_test.values, y_pred)
    rsr_score = rsr(y_test.values, y_pred)

    st.write("### Model Evaluation Metrics")
    st.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    st.write(f"NSE: {nse_score:.4f}, PBIAS: {pbias_score:.2f}%, RSR: {rsr_score:.4f}")

    # Forecast next 120 days
    future_dates = pd.date_range(start=df_filtered.index[-1] + pd.Timedelta(days=1), periods=120, freq='D')
    future_df = pd.DataFrame(index=future_dates)

    last_row = df_filtered.iloc[-1]
    current_lag_values = {f'Lag{i}': last_row[f'Lag{i-1}'] for i in range(2, 8)}
    current_lag_values['Lag1'] = last_row['MMI']
    current_nifty_lag_values = {f'Nifty_Lag{i}': last_row[f'Nifty_Lag{i}'] for i in range(1, 4)}
    last_nifty = last_row['Nifty']
    future_df['Nifty'] = last_nifty

    m_pred_window_3 = [last_row['MMI_Rolling3']] * 2
    m_pred_window_5 = [last_row['MMI_Rolling5']] * 4
    nifty_rolling_3 = last_row['Nifty_Rolling3']
    nifty_rolling_5 = last_row['Nifty_Rolling5']

    future_predictions = []
    for date in future_dates:
        feature_dict = {}
        feature_dict.update(current_lag_values)
        feature_dict.update(current_nifty_lag_values)

        rolling3_vals = m_pred_window_3[-2:] + [current_lag_values['Lag1']]
        mmi_rolling3 = np.mean(rolling3_vals)
        m_pred_window_3.append(mmi_rolling3)
        m_pred_window_3.pop(0)

        rolling5_vals = m_pred_window_5[-4:] + [current_lag_values['Lag1']]
        mmi_rolling5 = np.mean(rolling5_vals)
        m_pred_window_5.append(mmi_rolling5)
        m_pred_window_5.pop(0)

        feature_dict['MMI_Rolling3'] = mmi_rolling3
        feature_dict['MMI_Rolling5'] = mmi_rolling5
        feature_dict['Nifty_Rolling3'] = nifty_rolling_3
        feature_dict['Nifty_Rolling5'] = nifty_rolling_5
        feature_dict['Nifty'] = last_nifty

        future_row_df = pd.DataFrame([feature_dict], index=[date])[X.columns]
        predicted_mmi = model.predict(future_row_df)[0]
        future_predictions.append(predicted_mmi)

        for i in range(7, 1, -1):
            current_lag_values[f'Lag{i}'] = current_lag_values[f'Lag{i-1}']
        current_lag_values['Lag1'] = predicted_mmi

        for i in range(3, 1, -1):
            current_nifty_lag_values[f'Nifty_Lag{i}'] = current_nifty_lag_values[f'Nifty_Lag{i-1}']
        current_nifty_lag_values['Nifty_Lag1'] = last_nifty

    future_df['Predicted_MMI'] = future_predictions

    # Residual std for confidence interval
    residuals = df_filtered['MMI'] - model.predict(X)
    resid_std = np.std(residuals)

    # Lowest and highest predicted MMI
    lowest_mmi_value = future_df['Predicted_MMI'].min()
    lowest_mmi_date = future_df['Predicted_MMI'].idxmin()
    ci_lower = lowest_mmi_value - 1.96 * resid_std
    ci_upper = lowest_mmi_value + 1.96 * resid_std

    highest_mmi_value = future_df['Predicted_MMI'].max()
    highest_mmi_date = future_df['Predicted_MMI'].idxmax()
    ci_upper_high = highest_mmi_value + 1.96 * resid_std
    ci_lower_high = highest_mmi_value - 1.96 * resid_std

    st.write(f"### Forecast Summary for next 120 days")
    st.write(f"Lowest predicted MMI: {lowest_mmi_value:.2f} on {lowest_mmi_date.date()}")
    st.write(f"95% CI for lowest MMI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    st.write(f"Highest predicted MMI: {highest_mmi_value:.2f} on {highest_mmi_date.date()}")
    st.write(f"95% CI for highest MMI: [{ci_lower_high:.2f}, {ci_upper_high:.2f}]")

    # Recommendation logic
    buy_recommendation = ""
    sell_recommendation = ""
    if lowest_mmi_value < 50:
        buy_recommendation = f"BUY recommendation on {lowest_mmi_date.date()} with forecast MMI {lowest_mmi_value:.2f} (below 50 threshold)."
    else:
        buy_recommendation = "No BUY recommendation (lowest forecast MMI not below 50)."

    if highest_mmi_value > 50:
        sell_recommendation = f"SELL recommendation on {highest_mmi_date.date()} with forecast MMI {highest_mmi_value:.2f} (above 50 threshold)."
    else:
        sell_recommendation = "No SELL recommendation (highest forecast MMI not above 50)."

    st.write("### MMI Based Trading Recommendations")
    st.write(f"**{buy_recommendation}**")
    st.write(f"**{sell_recommendation}**")

# === Part 2: Holdings analysis and sell plan ===
st.header("Upload Your Stock Holdings")

mapping_url = "https://raw.githubusercontent.com/KPranaydeep/Finance/main/EQUITY_L.csv"

@st.cache_data(show_spinner=False)
def load_mapping(url):
    df_map = pd.read_csv(url)
    df_map.columns = df_map.columns.str.strip()
    return df_map

mapping_df = load_mapping(mapping_url)
st.success(f"Loaded ISIN-Symbol mapping with {len(mapping_df)} rows.")

holdings_file = st.file_uploader("Upload your Holdings Excel file (with ISIN and quantities)", type=["xlsx"])
if holdings_file:
    df_holdings = pd.read_excel(holdings_file, skiprows=10)
    st.success(f"Loaded holdings file with {len(df_holdings)} rows.")

    df_holdings['ISIN'] = df_holdings.iloc[:, 1].astype(str).str.strip()
    isin_to_symbol = dict(zip(mapping_df['ISIN NUMBER'].astype(str).str.strip(),
                              mapping_df['SYMBOL'].astype(str).str.strip()))

    # Map symbols
    df_holdings['Symbol'] = df_holdings['ISIN'].map(isin_to_symbol)

    # Drop rows with no symbol mapping
    missing_symbols = df_holdings['Symbol'].isna().sum()
    if missing_symbols > 0:
        st.warning(f"{missing_symbols} holdings could not be mapped to a Symbol.")

    df_holdings = df_holdings.dropna(subset=['Symbol'])
    df_holdings = df_holdings[df_holdings['Quantity'] > 0]

    st.write("### Holdings after mapping ISIN to Symbol")
    st.dataframe(df_holdings[['ISIN', 'Symbol', 'Quantity']])

    # Fetch latest market prices from Yahoo Finance
    unique_symbols = df_holdings['Symbol'].unique().tolist()
    yf_symbols = [sym + ".NS" for sym in unique_symbols]

    st.write("Fetching latest market prices from Yahoo Finance...")
    prices = {}
    for sym, yf_sym in zip(unique_symbols, yf_symbols):
        try:
            ticker = yf.Ticker(yf_sym)
            data = ticker.history(period="1d")
            if not data.empty:
                prices[sym] = data['Close'][0]
            else:
                prices[sym] = np.nan
        except Exception as e:
            prices[sym] = np.nan

    price_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
    df_holdings = df_holdings.merge(price_df, on='Symbol', how='left')

    df_holdings['Market Value'] = df_holdings['Quantity'] * df_holdings['Price']
    total_portfolio_value = df_holdings['Market Value'].sum()

    st.write(f"### Total Portfolio Market Value: ₹{total_portfolio_value:,.2f}")
    st.dataframe(df_holdings[['Symbol', 'Quantity', 'Price', 'Market Value']].sort_values(by='Market Value', ascending=False))

    # Sell plan calculation based on user's input sell percentage
    sell_percent = st.slider("Select the % of holdings to sell (applied uniformly)", 0, 100, 25)

    df_holdings['Qty_to_Sell'] = (df_holdings['Quantity'] * sell_percent / 100).round().astype(int)
    df_holdings['Sell_Value'] = df_holdings['Qty_to_Sell'] * df_holdings['Price']

    st.write(f"### Sell Plan at {sell_percent}% of holdings")
    st.dataframe(df_holdings[['Symbol', 'Quantity', 'Qty_to_Sell', 'Price', 'Sell_Value']])

    total_sell_value = df_holdings['Sell_Value'].sum()
    st.write(f"Total estimated sell value: ₹{total_sell_value:,.2f}")

    # Recommendation overlay from MMI forecast if available
    if uploaded_mmi_file:
        st.write("---")
        st.write("### Combine MMI Forecast Recommendations with Sell Plan")

        st.write("You can plan to SELL more aggressively on or near predicted high MMI dates (when market mood is pessimistic).")
        st.write(f"Highest predicted MMI date: {highest_mmi_date.date()} with value {highest_mmi_value:.2f}")

        # Suggest delaying sells if sell date is far from predicted high MMI
        st.write("You might want to consider holding until the predicted high MMI date for better sell timing.")

else:
    st.info("Please upload your holdings Excel file to analyze your portfolio.")
