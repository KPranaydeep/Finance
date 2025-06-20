import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import requests
import os
from io import BytesIO
from itertools import groupby
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Replace <db_password> with your actual MongoDB password
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"

# Create a new client and connect to the server using Server API v1
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("✅ Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("❌ Error connecting to MongoDB:", e)

db = client['finance_db']
collection = db['sell_plan_params']
mmi_collection = db['mmi_data']
allocation_collection = db['allocation_plans']

def save_allocation_plan(user_id, plan_df, total_amount, days, mmi_snapshot):
    record = {
        'user_id': user_id,
        'timestamp': datetime.utcnow(),
        'investable_amount': total_amount,
        'total_days': days,
        'mmi_snapshot': mmi_snapshot,
        'plan': plan_df.to_dict(orient='records')  # convert DataFrame to dict list
    }
    allocation_collection.insert_one(record)

def is_market_closed():
    try:
        nifty = yf.Ticker("^NSEI")
        df = nifty.history(period="1d", interval="5m")
        if len(df) < 2:
            return True  # Not enough data to say it's open
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        return abs(last_close - prev_close) < 0.5  # Minimal movement → likely closed
    except:
        return False  # Fail-safe: assume market open if error

def get_next_trading_day(from_date):
    # Skip Saturday/Sunday
    next_day = from_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_day += timedelta(days=1)
    return next_day

def save_input_params(user_id, net_pl, charges, target_pct):
    record = {
        'user_id': user_id,
        'timestamp': datetime.utcnow(),
        'net_pl': net_pl,
        'charges': charges,
        'target_pct': target_pct
    }
    collection.insert_one(record)

def get_latest_input_params(user_id):
    latest = collection.find_one(
        {'user_id': user_id},
        sort=[('timestamp', -1)]
    )
    return latest if latest else {'net_pl': 0.0, 'charges': 0.0, 'target_pct': 0.28}

st.set_page_config(layout="wide", page_icon=":moneybag:")
st.title("📊 Stock Holdings Analysis & Market Mood Dashboard")

# ==================== MARKET MOOD ANALYSIS ====================
from streamlit.runtime.uploaded_file_manager import UploadedFile

class MarketMoodAnalyzer:
    def __init__(self, mmi_data):
        if isinstance(mmi_data, bytes):
            self.df = self._prepare_mmi_data_from_bytes(mmi_data)
        elif isinstance(mmi_data, pd.DataFrame):
            self.df = self._prepare_mmi_data_from_df(mmi_data)
        elif hasattr(mmi_data, "read"):  # Handles Streamlit UploadedFile
            self.df = self._prepare_mmi_data_from_bytes(mmi_data.read())
        else:
            raise ValueError(f"Unsupported input type for MMI data: {type(mmi_data)}")

        self.df = self.df.sort_values('Date').reset_index(drop=True)
        self.run_lengths = self._identify_mood_streaks()
        
        self.mmi_last_date = self.df['Date'].iloc[-1].date()  # Last date in MMI data
        self.today_date = datetime.today().date()             # Actual today
        self.current_mmi = self.df['MMI'].iloc[-1]
        self.current_mood = 'Fear' if self.current_mmi <= 50 else 'Greed'
        self.current_streak = self._get_current_streak_length()

    def _prepare_mmi_data_from_bytes(self, mmi_bytes):
        if not mmi_bytes:
            raise ValueError("Empty file provided.")
        df = pd.read_csv(BytesIO(mmi_bytes))
        return self._process_dataframe(df)

    def _prepare_mmi_data_from_df(self, df):
        return self._process_dataframe(df)

    def _process_dataframe(self, df):
        df.columns = ['Date', 'MMI', 'Nifty']
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Mood'] = df['MMI'].apply(lambda x: 'Fear' if x <= 50 else 'Greed')
        return df

    def _identify_mood_streaks(self):
        mood_series = self.df['Mood'].values
        streaks = [(mood, sum(1 for _ in group)) for mood, group in groupby(mood_series)]
        run_lengths = {'Fear': [], 'Greed': []}
        for mood, length in streaks:
            run_lengths[mood].append(length)
        return run_lengths

    def _get_current_streak_length(self):
        current_streak = 1
        for i in range(len(self.df) - 2, -1, -1):
            if self.df['Mood'].iloc[i] == self.current_mood:
                current_streak += 1
            else:
                break
        return current_streak

    @staticmethod
    def _empirical_survival_hazard(data):
        kmf = KaplanMeierFitter()
        kmf.fit(data, event_observed=np.ones_like(data))
        surv = kmf.survival_function_.reset_index()
        return surv['timeline'].values, surv['KM_estimate'].values

    def _analyze_mood(self, mood):
        data = np.array(self.run_lengths[mood])
        days, S = self._empirical_survival_hazard(data)
        return {'runs': data, 'survival_days': days, 'survival_prob': S}

    def _get_confidence_flip_date(self, survival_days, survival_prob, confidence=0.05):
        for d, s in zip(survival_days, survival_prob):
            if s <= confidence:
                return d
        return None
    def generate_allocation_plan(self, investable_amount, total_days):
        mmi_today = self.current_mmi
        streak_days = self.current_streak
        mmi_step = (50 - mmi_today) / (total_days - 1)
    
        date = self.today_date + timedelta(days=1)
        mmi = mmi_today
        day_count = 0
        allocation_rows = []
    
        while day_count < total_days:
            if date.weekday() < 5:  # Only weekdays (Mon-Fri) are market days
                gap = max(0, 50 - mmi)
                weight = gap * (1 / streak_days) * (1 / total_days)
                allocation = weight * investable_amount
                percent = int(round((allocation / investable_amount) * 100))
    
                if percent > 0:  # Avoid showing 0% allocations
                    allocation_rows.append({
                        "Day": day_count + 1,
                        "Date": date.strftime('%a, %d %b %Y'),
                        "Est. MMI": round(mmi, 2),
                        "MMI Gap": round(gap, 2),
                        "Weight": round(weight, 6),
                        "Allocation (%)": f"{percent}%",
                        "Allocation (₹)": f"₹{allocation:.2f}"
                    })
    
                mmi += mmi_step
                day_count += 1
            date += timedelta(days=1)
    
        return pd.DataFrame(allocation_rows)

    def display_mood_analysis(self):
        fear_res = self._analyze_mood('Fear')
        greed_res = self._analyze_mood('Greed')
        res = fear_res if self.current_mood == 'Fear' else greed_res
    
        confidence_flip_day = self._get_confidence_flip_date(
            res['survival_days'], res['survival_prob']
        )
    
        mood_container = st.container()
        with mood_container:
            st.subheader("📈 Current Market Mood Analysis")
    
            col1, col2, col3 = st.columns(3)
            col1.metric("Current MMI", f"{self.current_mmi:.2f}", 
                        "Fear" if self.current_mmi <= 50 else "Greed")
            col2.metric("Current Streak", f"{self.current_streak} days")
    
            if confidence_flip_day is not None:
                days_until_flip = confidence_flip_day - self.current_streak
                raw_confidence_date = self.mmi_last_date + timedelta(days=days_until_flip)
                confidence_date = raw_confidence_date
                days_left = (confidence_date - self.today_date).days
    
                # ✅ Final flip display logic
                if days_left <= 0:
                    if is_market_closed() or raw_confidence_date < self.today_date:
                        confidence_date = get_next_trading_day(self.today_date)
                        flip_status = f"on {confidence_date.strftime('%A')}"
                    else:
                        flip_status = "today"
                else:
                    flip_status = f"in {days_left} days"
    
                col3.metric("Expected Flip Date", 
                            confidence_date.strftime('%d %b %Y'), 
                            flip_status)
    
            st.markdown("**Historical Patterns**")
            hist_col1, hist_col2 = st.columns(2)
            hist_col1.metric("Fear Streaks", 
                             f"{len(fear_res['runs'])}", 
                             f"Avg: {np.mean(fear_res['runs']):.1f} days")
            hist_col2.metric("Greed Streaks", 
                             f"{len(greed_res['runs'])}", 
                             f"Avg: {np.mean(greed_res['runs']):.1f} days")
    
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
    """Analyze stock holdings from Groww or Kite XLSX files"""
    try:
        xls = pd.ExcelFile(uploaded_holdings)
        sheet_names = xls.sheet_names

        if "Equity" in sheet_names:
            # Kite XLSX format (your uploaded file)
            df = pd.read_excel(uploaded_holdings, sheet_name="Equity", skiprows=22)
            df = df.dropna(subset=["Symbol"])
            df = df.rename(columns={
                "Quantity Available": "Quantity",
                "Average Price": "Average Price",
                "Unrealized P&L": "P&L",
                "Unrealized P&L Pct.": "P&L (%)",
                "Previous Closing Price": "Live LTP"
            })
            df["Company Name"] = df["Symbol"]

        else:
            # Assume Groww XLSX format
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
            df = df.merge(equity_mapping, on='ISIN', how='left')
            df.dropna(subset=['Symbol'], inplace=True)
            df['Company Name'] = df['Company Name'].fillna(df['Stock Name'])
            df['Live LTP'] = df['LTP']

        # Ensure numeric types
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Average Price'] = pd.to_numeric(df['Average Price'], errors='coerce')
        df['Live LTP'] = pd.to_numeric(df['Live LTP'], errors='coerce')

        # Clean and calculate derived fields
        df = df.dropna(subset=['Symbol', 'Quantity', 'Average Price', 'Live LTP'])
        df['Invested Amount'] = df['Quantity'] * df['Average Price']
        df['Current Value'] = df['Quantity'] * df['Live LTP']
        df['Profit/Loss'] = df['Current Value'] - df['Invested Amount']
        df['Profit/Loss (%)'] = (df['Profit/Loss'] / df['Invested Amount']) * 100

        return df[['Symbol', 'Company Name', 'Quantity', 'Average Price', 
                   'Live LTP', 'Invested Amount', 'Current Value', 
                   'Profit/Loss', 'Profit/Loss (%)']]

    except Exception as e:
        st.error(f"❌ Could not process XLSX file: {e}")
        return None

# ==================== STREAMLIT UI ====================

# ========== MMI SECTION (Place this AFTER the class definition) ==========
# ========== Load Analyzer from MongoDB or Uploaded File ==========
# ========== Define helper to read MMI from MongoDB ==========
def read_mmi_from_mongodb():
    try:
        records = list(mmi_collection.find({}, {'_id': 0}))
        if records:
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to read from MongoDB: {e}")
        return pd.DataFrame()

# ========== Upload full MMI dataset ==========
st.subheader("📂 Upload full MMI dataset (optional)")
with st.expander("📂 Upload Full MMI Dataset (CSV Format)"):
    uploaded_mmi_csv = st.file_uploader(
        "Upload full MMI dataset (CSV format)",
        type=["csv"],
        key="upload_mmi_db"
    )

uploaded_bytes = None
if uploaded_mmi_csv is not None and uploaded_mmi_csv.size > 0:
    uploaded_bytes = uploaded_mmi_csv.read()
    try:
        mmi_df = pd.read_csv(BytesIO(uploaded_bytes))
        mmi_df.columns = ['Date', 'MMI', 'Nifty']
        mmi_df['Date'] = pd.to_datetime(mmi_df['Date'], format='%d/%m/%Y')

        # Store to MongoDB
        mmi_collection.delete_many({})
        mmi_collection.insert_many(mmi_df.to_dict(orient='records'))

        st.success("✅ MMI data uploaded and saved to MongoDB")
    except Exception as e:
        st.error(f"❌ Error processing MMI CSV: {e}")
        uploaded_bytes = None  # reset on failure

# ========== MMI SECTION (Load Analyzer from MongoDB or Uploaded File) ==========
try:
    if uploaded_bytes:
        st.info("📄 Using uploaded MMI CSV file")
        analyzer = MarketMoodAnalyzer(uploaded_bytes)
    else:
        df_from_db = read_mmi_from_mongodb()
        if not df_from_db.empty:
            st.info("☁️ Using MMI data from MongoDB")
            analyzer = MarketMoodAnalyzer(df_from_db)
        else:
            st.warning("⚠️ No valid MMI data found in MongoDB. Please upload or add today’s MMI.")
            analyzer = None
except Exception as e:
    analyzer = None
    st.error(f"❌ Error loading MMI data: {str(e)}")

# ========== Add Today’s MMI Entry ==========
st.subheader("📝 Add Today's MMI")

with st.form("add_today_mmi"):
    today_mmi = st.number_input("Enter Today's MMI", min_value=0.0, max_value=100.0, step=0.1)
    today = datetime.today().date()
    submitted = st.form_submit_button("Add to MongoDB")

    if submitted:
        try:
            # Fetch today's Nifty value
            nifty_ticker = yf.Ticker("^NSEI")
            nifty_today = nifty_ticker.history(period='1d')['Close'].iloc[-1]

            # Insert into MongoDB
            mmi_collection.insert_one({
                "Date": datetime.combine(today, datetime.min.time()),
                "MMI": today_mmi,
                "Nifty": nifty_today
            })

            st.success(f"✅ Added today's MMI ({today_mmi}) and Nifty ({nifty_today:.2f}) to MongoDB")

            # Refresh analyzer with updated data
            df_from_db = read_mmi_from_mongodb()
            if not df_from_db.empty:
                analyzer = MarketMoodAnalyzer(df_from_db)
                st.success("🔄 Analysis updated with latest data")
            else:
                analyzer = None
                st.warning("⚠️ MongoDB returned no data. Please check your upload.")

        except Exception as e:
            analyzer = None
            st.error(f"❌ Failed to fetch Nifty or save to DB: {e}")

if analyzer:
    analyzer.display_mood_analysis()
    if analyzer.current_mood == "Fear":
        st.success("🟢 MMI indicates *Fear* – market may offer entry opportunities")

        # Existing logic
        from buy import show_buy_plan
        show_buy_plan(analyzer)

        # 🔥 NEW: Allocation Plan UI
        st.subheader("📊 Generate Smart Allocation Plan")

        with st.form("allocation_plan_form"):
            investable_amount = st.number_input("Enter total investable amount (₹)", min_value=1000.0, step=1000.0)
            total_days = st.slider("Number of days to spread investment", min_value=5, max_value=30, value=15)
            submitted = st.form_submit_button("Generate Allocation Plan")

        if submitted and investable_amount > 0:
            allocation_df = analyzer.generate_allocation_plan(investable_amount, total_days)

            if not allocation_df.empty:
                st.success("✅ Allocation plan generated")
                st.dataframe(allocation_df)

                USER_ID = "default_user"  # Can replace with login-based ID
                from datetime import datetime
                save_allocation_plan(USER_ID, allocation_df, investable_amount, total_days, {
                    "mmi": analyzer.current_mmi,
                    "mood": analyzer.current_mood,
                    "streak": analyzer.current_streak,
                    "date": analyzer.mmi_last_date.strftime("%Y-%m-%d")
                })

                csv = allocation_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Allocation Plan", csv, "allocation_plan.csv", "text/csv")

        with st.expander("📂 View Last Saved Allocation Plan"):
            last_plan = get_latest_allocation_plan("default_user")
            if last_plan is not None:
                st.dataframe(last_plan)
            else:
                st.info("No saved allocation plan found.")

st.header("📤 Upload Your Holdings")

with st.expander("📤 Upload Your Stock Holdings File"):
    uploaded_holdings = st.file_uploader(
        "Upload your stock holdings file (.xlsx format only — Groww or Kite)",
        type=['xlsx']
    )

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
        # Set a default user ID (could later be tied to login/email)
        USER_ID = "default_user"
        
        # Fetch latest saved values from MongoDB
        latest_params = get_latest_input_params(USER_ID)
        
        st.subheader("🎯 Profit Booking Strategy")
        st.subheader("🔧 Adjust Profit Booking Parameters")
        
        # Load previous values as defaults
        net_pl = st.number_input("Enter net P&L (INR)", 
                                 value=float(latest_params.get('net_pl', 0.0)), 
                                 min_value=0.0, 
                                 step=1000.0)
        
        charges = st.number_input("Enter charges (INR)", 
                                  value=float(latest_params.get('charges', 0.0)), 
                                  min_value=0.0, 
                                  step=100.0)
        
        target_net_daily_pct = st.number_input("Target net daily return (%)", 
                                               value=float(latest_params.get('target_pct', 0.28)), 
                                               min_value=0.01, 
                                               max_value=5.0, 
                                               step=0.01)
        
        # Save values when form submitted or app rerun
        save_input_params(USER_ID, net_pl, charges, target_net_daily_pct)

        # ✅ Fix starts here — remove one indentation level from this line onward
        if net_pl > 0:
            sell_limit_multiplier = calculate_dynamic_sell_limit(net_pl, charges, target_net_daily_pct)
            daily_return_pct = round((sell_limit_multiplier - 1) * 100, 4)
            st.markdown(f"💡 *Dynamic sell limit calculated at {daily_return_pct}% above buy price*")

            # Choose rotation strategy
            rotation_option = st.radio(
                "Select rotation strategy for calculating target profit:",
                ["Daily", "Weekly", "Monthly"],
                index=2,
                horizontal=True
            )
            
            # Calculate target based on selected rotation
            if rotation_option == "Daily":
                default_target = round(total_invested * (sell_limit_multiplier - 1), 2)
            elif rotation_option == "Weekly":
                default_target = round(total_invested * (sell_limit_multiplier - 1) * 4.84615385, 2)
            else:  # Monthly Rotation
                default_target = round(total_invested * (sell_limit_multiplier - 1) * 21, 2)

                st.title("Monthly Rotation Checklist – 21 Market Days Strategy (Jun–Dec 2025)")
                
                st.markdown("Each rotation happens after ~21 market days from the previous one. Check off when done!")
                
                st.checkbox("July 18, 2025 (Fri) – 1st Monthly Rotation")
                
                st.checkbox("August 22, 2025 (Fri) – 2nd Monthly Rotation (Adjusted for Aug 15 Holiday)")
                
                st.checkbox("September 19, 2025 (Fri) – 3rd Monthly Rotation")
                
                st.checkbox("October 17, 2025 (Fri) – 4th Monthly Rotation (Oct 18 is Saturday)")
                
                st.checkbox("November 21, 2025 (Fri) – 5th Monthly Rotation (Post-Diwali Holiday Window)")
                
                st.checkbox("December 19, 2025 (Fri) – 6th Monthly Rotation")
            
            target_rupees = st.number_input("Enter target profit (₹)", 
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
