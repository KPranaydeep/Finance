import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import requests
from io import BytesIO
from itertools import groupby
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter
from datetime import timedelta

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
        with st.expander("📈 Market Mood Analysis (Expand for Details)"):
            # Analyze historical patterns
            fear_res = self._analyze_mood('Fear')
            greed_res = self._analyze_mood('Greed')
            res = fear_res if self.current_mood == 'Fear' else greed_res
            
            # Find confidence-based flip point
            confidence_flip_day = self._get_confidence_flip_date(
                res['survival_days'], 
                res['survival_prob']
            )
            
            # Display current status
            st.subheader("🔍 Current Market Mood Status")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current MMI", f"{self.current_mmi:.2f}")
            col2.metric("Market Mood", self.current_mood)
            col3.metric("Current Streak", f"{self.current_streak} days")
            
            # Display historical patterns
            st.subheader("📊 Historical Mood Patterns")
            hist_col1, hist_col2 = st.columns(2)
            hist_col1.metric("Fear Streaks", f"{len(fear_res['runs'])}", f"Avg: {np.mean(fear_res['runs']):.1f} days")
            hist_col2.metric("Greed Streaks", f"{len(greed_res['runs'])}", f"Avg: {np.mean(greed_res['runs']):.1f} days")
            
            # Display prediction
            if confidence_flip_day:
                days_until_flip = confidence_flip_day - self.current_streak
                confidence_date = self.today_date + timedelta(days=days_until_flip)
                
                st.subheader("⚠️ High Confidence Prediction")
                st.warning(f"95% probability the current {self.current_mood} streak will end by:")
                st.metric("Expected Flip Date", confidence_date.strftime('%d %b %Y'), f"in {days_until_flip} days")
                
                if self.current_mood == 'Greed':
                    st.info("🛑 Recommended Action: Prepare for potential Fear entry opportunity")
                else:
                    st.info("🛑 Recommended Action: Prepare for potential Greed exit opportunity")
            else:
                st.info("ℹ️ No high-confidence flip prediction available within observed data range")

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
# Sidebar for MMI data upload
with st.sidebar:
    st.header("📊 Market Mood Data")
    uploaded_mmi = st.file_uploader("Upload MMI Data (CSV)", type=['csv'])
    if uploaded_mmi:
        mood_analyzer = MarketMoodAnalyzer(uploaded_mmi.read())
        mood_analyzer.display_mood_analysis()

# Main content area for holdings analysis
st.header("💼 Upload Your Groww Holdings File (.xlsx)")
uploaded_holdings = st.file_uploader("📂 Upload your Groww holdings file", type=['xlsx'])

if uploaded_holdings:
    merged_df = analyze_holdings(uploaded_holdings)
    if merged_df is not None:
        # Display holdings
        st.markdown("### 🧾 Your Holdings")
        st.dataframe(merged_df[['Symbol', 'Company Name', 'Quantity', 'Average Price', 
                              'Live LTP', 'Current Value', 'Profit/Loss', 'Profit/Loss (%)']])
        
        # Portfolio summary
        total_invested = merged_df['Invested Amount'].sum()
        total_current_value = merged_df['Current Value'].sum()
        total_pl = merged_df['Profit/Loss'].sum()

        st.markdown("### 📊 Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
        col2.metric("📈 Current Value", f"₹{total_current_value:,.2f}")
        col3.metric("📊 Overall P&L", f"₹{total_pl:,.2f}", delta=f"{(total_pl/total_invested)*100:.2f}%")

        # Sell plan logic
        st.markdown("💡 *To achieve 100% CAGR (doubling in 1 year), you need ~0.34% daily gross return.*")
        default_target = round(total_invested * 0.0034, 2)
        target_rupees = st.number_input("🎯 Enter today's target booking profit (₹)", 
                                      value=default_target, min_value=0.0, step=100.0)

        merged_df = merged_df.sort_values(by='Profit/Loss (%)', ascending=False)
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

            st.subheader("📤 Suggested Sell Plan")
            st.success(f"✅ To book ₹{target_rupees:.2f}, sell these holdings:")
            st.dataframe(sell_plan[['Symbol', 'Company Name', 'Quantity', 'Average Price',
                                  'Live LTP', 'Sell Limit (₹)', 'Profit/Loss', 'Profit/Loss (%)']])
        else:
            st.warning("📉 Not enough profitable stocks to meet target.")
            st.info("⏳ Check back tomorrow when market conditions may improve.")
