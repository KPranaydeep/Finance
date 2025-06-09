import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import requests
from io import BytesIO
from itertools import groupby
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_icon=":moneybag:")
st.title("📊 Stock Holdings Analysis & Market Mood Dashboard")

# ==================== MARKET MOOD ANALYSIS ====================
class MarketMoodAnalyzer:
    def __init__(self, mmi_data):
        self.df = self._prepare_mmi_data(mmi_data)
        self.run_lengths = self._identify_mood_streaks()
        self.today_date = self.df['Date'].iloc[-1]
        self.current_mmi = self.df['MMI'].iloc[-1]
        self.current_mood = 'Fear' if self.current_mmi <= 50 else 'Greed'
        self.current_streak = self._get_current_streak_length()
        
    def _prepare_mmi_data(self, mmi_data):
        """Prepare MMI data from CSV content"""
        df = pd.read_csv(BytesIO(mmi_data))
        df.columns = ['Date', 'MMI', 'Nifty']
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df.sort_values('Date', inplace=True)
        df['Mood'] = df['MMI'].apply(lambda x: 'Fear' if x < 50 else 'Greed')
        return df
    
    def _identify_mood_streaks(self):
        """Identify consecutive runs of Fear/Greed moods"""
        mood_series = self.df['Mood'].values
        streaks = [(mood, sum(1 for _ in group)) for mood, group in groupby(mood_series)]
        run_lengths = {'Fear': [], 'Greed': []}
        for mood, length in streaks:
            run_lengths[mood].append(length)
        return run_lengths
    
    def _get_current_streak_length(self):
        """Calculate length of current mood streak"""
        current_streak = 1
        for i in range(len(self.df) - 2, -1, -1):
            if self.df['Mood'].iloc[i] == self.current_mood:
                current_streak += 1
            else:
                break
        return current_streak
    
    @staticmethod
    def _empirical_survival_hazard(data):
        """Calculate empirical survival function"""
        kmf = KaplanMeierFitter()
        kmf.fit(data, event_observed=np.ones_like(data))
        surv = kmf.survival_function_.reset_index()
        return surv['timeline'].values, surv['KM_estimate'].values
    
    def _analyze_mood(self, mood):
        """Analyze historical patterns for a specific mood"""
        data = np.array(self.run_lengths[mood])
        days, S = self._empirical_survival_hazard(data)
        return {'runs': data, 'survival_days': days, 'survival_prob': S}
    
    def _get_confidence_flip_date(self, survival_days, survival_prob, confidence=0.05):
        """Get flip date at specified confidence level"""
        for d, s in zip(survival_days, survival_prob):
            if s <= confidence:
                return d
        return None
    
    def display_mood_analysis(self):
        """Display market mood analysis in Streamlit"""
        # Analyze historical patterns
        fear_res = self._analyze_mood('Fear')
        greed_res = self._analyze_mood('Greed')
        res = fear_res if self.current_mood == 'Fear' else greed_res
        
        # Find confidence-based flip point
        confidence_flip_day = self._get_confidence_flip_date(
            res['survival_days'], 
            res['survival_prob']
        )
        
        # Create mood display container
        mood_container = st.container()
        with mood_container:
            st.subheader("📈 Current Market Mood Analysis")
            
            # Current status columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Current MMI", f"{self.current_mmi:.2f}", 
                       "Fear" if self.current_mmi <= 50 else "Greed")
            col2.metric("Current Streak", f"{self.current_streak} days")
            
            # Mood prediction
            if confidence_flip_day:
                days_until_flip = confidence_flip_day - self.current_streak
                confidence_date = self.today_date + timedelta(days=days_until_flip)
                col3.metric("Expected Flip Date", 
                           confidence_date.strftime('%d %b %Y'), 
                           f"in {days_until_flip} days")
            
            # Historical patterns
            st.markdown("**Historical Patterns**")
            hist_col1, hist_col2 = st.columns(2)
            hist_col1.metric("Fear Streaks", 
                            f"{len(fear_res['runs'])}", 
                            f"Avg: {np.mean(fear_res['runs']):.1f} days")
            hist_col2.metric("Greed Streaks", 
                           f"{len(greed_res['runs'])}", 
                           f"Avg: {np.mean(greed_res['runs']):.1f} days")
            
            # Recommendation
            if confidence_flip_day:
                if self.current_mood == 'Greed':
                    st.warning("🛑 Market in Greed Phase - Consider profit booking")
                else:
                    st.success("🟢 Market in Fear Phase - Look for entry opportunities")

# ==================== STOCK HOLDINGS ANALYSIS ====================
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

def analyze_holdings(uploaded_holdings):
    """Analyze stock holdings and generate sell plan"""
    try:
        # Read and clean holdings file
        df = pd.read_excel(uploaded_holdings, sheet_name='Sheet1', skiprows=9)
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

        # Merge with equity mapping
        equity_mapping = load_equity_mapping()
        merged_df = df.merge(equity_mapping, on='ISIN', how='left')
        merged_df.dropna(subset=['Symbol'], inplace=True)

        # Fetch live prices
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

        # Calculate metrics
        merged_df['Invested Amount'] = merged_df['Quantity'] * merged_df['Average Price']
        merged_df['Current Value'] = merged_df['Quantity'] * merged_df['Live LTP']
        merged_df['Profit/Loss'] = merged_df['Current Value'] - merged_df['Invested Amount']
        merged_df['Profit/Loss (%)'] = (merged_df['Profit/Loss'] / merged_df['Invested Amount']) * 100

        return merged_df
    except Exception as e:
        st.error(f"❌ Could not process file: {e}")
        return None

# ==================== STREAMLIT UI ====================
st.header("📤 Upload Your Holdings")
uploaded_holdings = st.file_uploader("Upload your stock holdings Excel file (.xlsx)", type=['xlsx'])

if uploaded_holdings:
    st.header("💼 Your Portfolio Analysis")
    ...

# Add these functions to your existing code
def get_april_first_current_year():
    today = datetime.now()
    april_first = datetime(today.year, 4, 1)
    return april_first.strftime("%Y-%m-%d")

def trading_days_elapsed(start_date_str, end_date_str=None):
    if end_date_str is None:
        end_date = datetime.now().date()
    else:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

    all_days = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = all_days[all_days.dayofweek < 5]  # Monday=0 ... Friday=4
    return len(trading_days)

def calculate_dynamic_sell_limit(net_pl, charges, target_net_daily_pct=0.28):
    """Calculate dynamic sell limit based on P&L and charges"""
    start_date_str = get_april_first_current_year()
    days_elapsed = trading_days_elapsed(start_date_str)
    gross_pl = net_pl + charges
    effective_cost_pct = (charges / gross_pl) * 100
    daily_charges_pct = effective_cost_pct / days_elapsed
    sell_limit_pct = target_net_daily_pct + daily_charges_pct
    return round(1 + (sell_limit_pct/100), 6)  # Convert to multiplier (e.g., 1.005298)

# Modify the profit booking section in your Streamlit app
if uploaded_holdings:
    st.header("💼 Your Portfolio Analysis")
    merged_df = analyze_holdings(uploaded_holdings)
    
    if merged_df is not None:
        # Display holdings
        st.subheader("🧾 Current Holdings")
        st.dataframe(merged_df[['Symbol', 'Company Name', 'Quantity', 'Average Price', 
                              'Live LTP', 'Current Value', 'Profit/Loss', 'Profit/Loss (%)']])
        
        # Portfolio summary
        total_invested = merged_df['Invested Amount'].sum()
        total_current_value = merged_df['Current Value'].sum()
        total_pl = merged_df['Profit/Loss'].sum()

        st.subheader("📊 Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
        col2.metric("📈 Current Value", f"₹{total_current_value:,.2f}")
        col3.metric("📊 Overall P&L", f"₹{total_pl:,.2f}", delta=f"{(total_pl/total_invested)*100:.2f}%")

        # Sell plan logic - MODIFIED SECTION
        st.subheader("🎯 Profit Booking Strategy")
        
        with st.expander("🔧 Adjust Profit Booking Parameters"):
            net_pl = st.number_input("Enter net P&L (INR)", 
                                     value=float(total_pl), 
                                     min_value=0.0, 
                                     step=1000.0)
            charges = st.number_input("Enter charges (INR)", 
                                      value=6135.0, 
                                      min_value=0.0, 
                                      step=100.0)
            target_net_daily_pct = st.number_input("Target net daily return (%)", 
                                                   value=0.28, 
                                                   min_value=0.01, 
                                                   max_value=5.0, 
                                                   step=0.01)
        
            if net_pl > 0:
                sell_limit_multiplier = calculate_dynamic_sell_limit(net_pl, charges, target_net_daily_pct)
                daily_return_pct = round((sell_limit_multiplier - 1) * 100, 4)
                st.markdown(f"💡 *Dynamic sell limit calculated at {daily_return_pct}% above buy price*")
                
                # Default target profit
                default_target = round(total_invested * (sell_limit_multiplier - 1), 2)
                target_rupees = st.number_input("Enter today's target profit (₹)", 
                                                value=default_target, 
                                                min_value=0.0, 
                                                step=100.0)
        
                # --- Move entire sell plan logic here ---
                # Filter profitable stocks
                profitable_df = merged_df[merged_df['Profit/Loss'] > 0].copy()
                profitable_df = profitable_df.sort_values(by='Profit/Loss (%)', ascending=False)
                profitable_df['Sell Limit (₹)'] = (profitable_df['Live LTP'] * sell_limit_multiplier).round(2)
        
                cumulative_profit = 0.0
                sell_plan_rows = []
        
                for _, row in profitable_df.iterrows():
                    if cumulative_profit >= target_rupees:
                        break
        
                    per_share_profit = row['Sell Limit (₹)'] - row['Average Price']
                    if per_share_profit <= 0:
                        continue
        
                    max_possible_shares = row['Quantity']
                    needed_profit = target_rupees - cumulative_profit
                    shares_to_sell = min(max_possible_shares, int(needed_profit // per_share_profit))
        
                    if shares_to_sell <= 0:
                        continue
        
                    actual_profit = shares_to_sell * per_share_profit
                    cumulative_profit += actual_profit
        
                    sell_plan_rows.append({
                        'Symbol': row['Symbol'],
                        'Company Name': row['Company Name'],
                        'Quantity to Sell': shares_to_sell,
                        'Average Price': row['Average Price'],
                        'Current Price': row['Live LTP'],
                        'Sell Limit (₹)': row['Sell Limit (₹)'],
                        'Expected Profit': actual_profit,
                        'Profit (%)': (row['Sell Limit (₹)'] - row['Average Price']) / row['Average Price'] * 100
                    })
        
                if sell_plan_rows:
                    sell_plan_df = pd.DataFrame(sell_plan_rows)
                    st.success(f"✅ Suggested Sell Plan to Book ₹{cumulative_profit:.2f} Profit")
                    st.dataframe(sell_plan_df)
                    csv = sell_plan_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Sell Plan",
                        csv,
                        "sell_plan.csv",
                        "text/csv",
                        key='download-sell-plan'
                    )
                else:
                    st.warning("📉 Not enough profitable stocks to meet target")
                    st.info("⏳ Check back tomorrow when market conditions may improve")
            else:
                st.error("❌ Cannot calculate sell limit with zero or negative P&L")
