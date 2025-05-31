import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Stock Holdings Analysis & Sell Plan")

# GitHub raw URL for ISIN-Symbol mapping CSV
mapping_url = "https://raw.githubusercontent.com/KPranaydeep/Finance/main/EQUITY_L.csv"

@st.cache_data(show_spinner=False)
def load_mapping(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

mapping_df = load_mapping(mapping_url)
st.success(f"Loaded ISIN-Symbol mapping with {len(mapping_df)} rows.")

# Upload Holdings Excel file
holdings_file = st.file_uploader("Upload your Holdings Excel file (with ISIN and quantities)", type=["xlsx"])
if holdings_file:
    # Adjust skiprows if your Excel has header rows to skip
    df = pd.read_excel(holdings_file, skiprows=10)
    st.success(f"Loaded holdings file with {len(df)} rows.")

    # Clean ISIN column - assuming ISIN is 2nd column (index 1)
    df['ISIN'] = df.iloc[:, 1].astype(str).str.strip()

    # Map ISIN to SYMBOL using mapping_df
    isin_to_symbol = dict(zip(mapping_df['ISIN NUMBER'].astype(str).str.strip(),
                              mapping_df['SYMBOL'].astype(str).str.strip()))

    df['SYMBOL'] = df['ISIN'].map(isin_to_symbol)

    # Drop rows where mapping fails
    df = df.dropna(subset=['SYMBOL']).reset_index(drop=True)
    df['SYMBOL'] = df['SYMBOL'].str.upper().str.strip()
    df['TICKER'] = df['SYMBOL'] + '.NS'

    st.write("Mapped ISIN to SYMBOL:")
    st.dataframe(df[['ISIN', 'SYMBOL', 'TICKER']])

    # Fetch live prices from yfinance
    unique_tickers = df['TICKER'].unique()

    @st.cache_data(show_spinner=False)
    def fetch_price(ticker):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            return info.get('currentPrice', None)
        except Exception:
            return None

    with st.spinner("Fetching live prices..."):
        prices = {ticker: fetch_price(ticker) for ticker in unique_tickers}

    df['LTP'] = df['TICKER'].map(prices)

    # Fill missing LTP with Closing Price if available
    if 'Closing price' in df.columns:
        df['LTP'] = df['LTP'].fillna(df['Closing price'])

    # Rename Quantity and AvgBuyPrice columns - adjust indexes if your file differs
    df.rename(columns={
        df.columns[2]: 'Quantity',
        df.columns[3]: 'AvgBuyPrice'
    }, inplace=True)

    # Calculate Unrealized P&L and Profit per share
    df['Unrealized_PnL'] = (df['LTP'] - df['AvgBuyPrice']) * df['Quantity']
    df['Profit_per_share'] = df['LTP'] - df['AvgBuyPrice']

    st.subheader("Holdings with LTP and Unrealized P&L")
    st.dataframe(df[['SYMBOL', 'Quantity', 'AvgBuyPrice', 'LTP', 'Unrealized_PnL']])

# Calculate default target profit as (0.069957 / 20) * SUMPRODUCT(Quantity * AvgBuyPrice)
    scaling_factor = 0.069957 / 20
    sumproduct = (df['Quantity'] * df['AvgBuyPrice']).sum()
    default_profit = int(scaling_factor * sumproduct)

    # Target profit input (default is computed)
    target_profit = st.number_input(
        "Enter target profit to book (₹)",
        min_value=1,
        value=default_profit,
        step=100
    )

    def get_sell_plan(df, target_profit):
        df_profitable = df[df['Unrealized_PnL'] > 0].sort_values(by='Unrealized_PnL', ascending=False).reset_index(drop=True)
        total_profit = 0
        plan = []

        for _, row in df_profitable.iterrows():
            if total_profit >= target_profit:
                break

            sell_price = round(row['LTP'] * 1.00324, 2)  # 0.324% markup on LTP
            profit_per_share = sell_price - row['AvgBuyPrice']
            max_profit = profit_per_share * row['Quantity']

            if total_profit + max_profit <= target_profit:
                sell_qty = row['Quantity']
            else:
                profit_needed = target_profit - total_profit
                sell_qty = int(profit_needed / profit_per_share)
                sell_qty = min(sell_qty, row['Quantity'])

            if sell_qty > 0:
                realized_profit = sell_qty * profit_per_share
                sell_value = round(sell_qty * sell_price, 2)
                plan.append({
                    'Stock': row['SYMBOL'],
                    'Sell Quantity': sell_qty,
                    'Sell Price (₹)': sell_price,
                    'Sell Value (₹)': sell_value,
                    'Realized Profit (₹)': round(realized_profit, 2)
                })
                total_profit += realized_profit

        return pd.DataFrame(plan), total_profit

    plan_df, total_booked = get_sell_plan(df, target_profit)
    plan_df = pd.DataFrame(plan).sort_values(by="Sell Value (₹)", ascending=False).reset_index(drop=True)
    st.subheader(f"Sell Plan to Book ₹{target_profit} Profit")
    if not plan_df.empty:
        st.dataframe(plan_df)
        st.write(f"### Total Estimated Profit Booked: ₹{total_booked:.2f}")
    else:
        st.info("No profitable sell plan possible with current holdings.")
        
