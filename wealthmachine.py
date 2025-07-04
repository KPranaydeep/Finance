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
import json
def get_max_roi_from_file():
    try:
        with open("max_roi.json", "r") as f:
            data = json.load(f)
            return data.get("max_roi", 0.0)
    except FileNotFoundError:
        return 0.0

min_threshold = get_max_roi_from_file()
# st.caption(f"🧪 Debug: Loaded min_threshold = {min_threshold:.2f}% from max_roi.json")

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

    def _count_trading_days(self, start_date, calendar_days):
        """Counts trading days (Mon–Fri) starting from given date for next X calendar days"""
        end_date = start_date + timedelta(days=calendar_days)
        trading_days = pd.date_range(start=start_date + timedelta(days=1), end=end_date, freq='B')
        return len(trading_days)

    def _get_days_until_confidence_flip(self, confidence=0.05):
        """Returns number of calendar days until the current mood is expected to flip"""
        mood = self.current_mood
        res = self._analyze_mood(mood)
        confidence_flip_day = self._get_confidence_flip_date(res['survival_days'], res['survival_prob'], confidence)
    
        if confidence_flip_day is not None:
            return max(1, confidence_flip_day - self.current_streak)
        else:
            return None

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
    def _generate_survival_based_forecast(self, forecast_days=30, confidence=0.05):
        remaining_days = self._get_forecast_horizon(confidence)
        forecast = []
        current_mmi = self.current_mmi
        mood = self.current_mood
        flip_triggered = False
    
        for day in range(1, forecast_days + 1):
            forecast_date = self.today_date + timedelta(days=day)
    
            if day <= remaining_days:
                mmi = current_mmi + np.random.uniform(-2, 2)
            else:
                if not flip_triggered:
                    flip_triggered = True
                    mmi = 49.0 if mood == 'Greed' else 51.0
                    mood = 'Fear' if mood == 'Greed' else 'Greed'
                else:
                    mmi += np.random.uniform(-3, 3)
    
            current_mmi = mmi
            forecast.append((forecast_date, mmi))
    
        return pd.DataFrame(forecast, columns=['Date', 'Forecasted_MMI'])
    def _get_forecast_horizon(self, confidence=0.05):
        streak_data = self.run_lengths[self.current_mood]
        if len(streak_data) < 2:
            return 5  # default small fallback
    
        shape, loc, scale = weibull_min.fit(streak_data, floc=0)
        current = self.current_streak
        x = current
        while weibull_min.sf(x, shape, loc, scale) > confidence:
            x += 1
            if x > 10 * max(streak_data):
                break
        return max(1, x - current)

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

    def generate_allocation_plan(self, investable_amount):
        """Generate MMI-aware staggered allocation plan where last day is the confidence flip date"""
        days_until_flip = self._get_days_until_confidence_flip()
    
        if days_until_flip is None:
            st.warning("⚠️ Could not forecast flip — defaulting to 15 market days.")
            target_flip_date = self.today_date + timedelta(days=21)
        else:
            target_flip_date = self.mmi_last_date + timedelta(days=days_until_flip)
    
        # Generate trading days between tomorrow and flip date (inclusive)
        all_days = pd.date_range(start=self.today_date + timedelta(days=1), end=target_flip_date, freq='B')
    
        # If flip is far enough, restrict to Mondays only
        if days_until_flip is not None and days_until_flip >= 14:
            all_days = [day for day in all_days if day.weekday() == 0]  # Monday = 0
    
        if len(all_days) < 5:
            st.warning("⚠️ Not enough trading days before flip — using minimum of 5")
            all_days = pd.date_range(start=self.today_date + timedelta(days=1), periods=5, freq='B')
    
        mmi_today = self.current_mmi
        streak_days = self.current_streak
        mmi_step = (50 - mmi_today) / max(1, (len(all_days) - 1))
    
        allocation_rows = []
        total_weight = 0
        mmi = mmi_today
        temp_rows = []
    
        for i, date in enumerate(all_days):
            gap = max(0, 50 - mmi)
            weight = gap * (1 / streak_days)
            temp_rows.append((i + 1, date, mmi, gap, weight))
            total_weight += weight
            mmi += mmi_step
    
        # Use last date as confidence_date
        confidence_date = all_days[-1].date()
    
        allocation_total = 0
        for i, (day_num, date, mmi, gap, weight) in enumerate(temp_rows):
            weight_norm = weight / total_weight
            allocation = investable_amount * weight_norm
            allocation_total += allocation
    
            allocation_rows.append({
                "Day": day_num,
                "Date": date.strftime('%a, %d %b %Y'),
                "Est. MMI": round(mmi, 2),
                "MMI Gap": round(gap, 2),
                "Weight": round(weight_norm, 6),
                "Allocation (%)": f"{round(weight_norm * 100, 2)}%",
                "Allocation (₹)": f"₹{allocation:.2f}"
            })
    
        total_alloc = sum(float(row['Allocation (₹)'].replace('₹', '')) for row in allocation_rows)
        diff = investable_amount - total_alloc
        if abs(diff) > 0.01:
            allocation_rows[-1]['Allocation (₹)'] = f"₹{(float(allocation_rows[-1]['Allocation (₹)'].replace('₹', '')) + diff):.2f}"
    
        return pd.DataFrame(allocation_rows), confidence_date

    def display_mood_analysis(self):
        from scipy import stats  # Only needed here
    
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
    
            with st.expander("📊 Show Historical Streak Patterns", expanded=False):
                # 🧮 Streak Stats
                fear_runs = fear_res['runs']
                greed_runs = greed_res['runs']
            
                fear_min = np.min(fear_runs) if len(fear_runs) else None
                fear_max = np.max(fear_runs) if len(fear_runs) else None
                fear_mean = np.mean(fear_runs) if len(fear_runs) else None
                fear_median = np.median(fear_runs) if len(fear_runs) else None
                fear_mode = int(stats.mode(fear_runs, keepdims=False).mode) if len(fear_runs) else None
            
                greed_min = np.min(greed_runs) if len(greed_runs) else None
                greed_max = np.max(greed_runs) if len(greed_runs) else None
                greed_mean = np.mean(greed_runs) if len(greed_runs) else None
                greed_median = np.median(greed_runs) if len(greed_runs) else None
                greed_mode = int(stats.mode(greed_runs, keepdims=False).mode) if len(greed_runs) else None
            
                # 📘 Table Summary
                st.markdown("**📘 Historical Streak Statistics (Days)**")
                st.table(pd.DataFrame({
                    "Mood": ["Fear", "Greed"],
                    "Min": [fear_min, greed_min],
                    "Median": [fear_median, greed_median],
                    "Mean": [round(fear_mean, 1), round(greed_mean, 1)],
                    "Mode": [fear_mode, greed_mode],
                    "Max": [fear_max, greed_max]
                }))
            
                hist_col1, hist_col2 = st.columns(2)
                hist_col1.metric("Fear Streaks",
                                 f"{len(fear_runs)}",
                                 f"Avg: {fear_mean:.1f} days")
                hist_col2.metric("Greed Streaks",
                                 f"{len(greed_runs)}",
                                 f"Avg: {greed_mean:.1f} days")
            # 🔁 Capital Allocation Suggestion Based on MMI
            st.warning("### 💰 Capital Allocation")

            if self.current_mmi < 50:
                invest_pct = (50 - self.current_mmi) * 2
                st.info(f"""
                😊  **Fear in Market (MMI = {self.current_mmi:.2f})**  
                👉 **Invest `{invest_pct:.1f}%`** of your deployable cash.  
                🪙 Fear offers value buys, consider accumulating high-quality assets at lower valuations.
                """)
            elif self.current_mmi > 50:
                liquid_hold_pct = (self.current_mmi - 50) * 2
                st.info(f"""
                😬  **Greed in Market (MMI = {self.current_mmi:.2f})**
                
                👉  Hold at least `{liquid_hold_pct:.1f}%` of total capital in liquid, low-risk instruments.  
                    Ideally, keep this amount not invested and easily liquidable in your account.
                
                🧠  Wait for better valuations to re-enter.  
                    Greed phases often precede corrections.
                """)

            else:
                st.info("""
                ⚖️ **MMI at Neutral (50)**  
                No strong directional bias — consider balanced allocation between equity and liquid assets.
                """)

            # 🧠 Dynamic Mood Suggestion
            if self.current_mood == 'Greed':
                threshold = (greed_max - self.current_streak) * 0.277
                active_threshold = min_threshold if min_threshold > 0 else threshold
            
                if self.current_streak < greed_mean:
                    st.warning(f"""
            📉 **Market in Greed** – but still early in the cycle.  
            This phase is ideal for:
            
            - 🏦 **Booking profits** on outperformers  
            - 🔁 **Rotating into safer assets**  
            - 💵 **Holding cash** to prepare for possible pullbacks  
            
            📊 **Action Tip**  
            If your portfolio has gained over **{threshold:.1f}%**, it’s wise to secure some gains.  
            For more active strategies, start rotating once returns cross **{active_threshold:.1f}%** to stay agile and reduce downside risk.
                    """)
                else:
                    st.warning(f"""
            🛑 **Market in Greed** – Current streak: `{self.current_streak}` days  
            **Above average**: `{greed_mean:.1f}` days
            
            📉 Now is an optimal time to **book profits** and shift your gains into:
            - 💵 Cash or liquid funds  
            - 🧱 Short-duration bonds  
            - 🟡 Commodities like **Gold**, **Silver**, or other non-equity hedges  
            
            💡 **Why?**
            - Greed doesn't last forever — **extended streaks often precede market corrections**
            - Selling pressure typically builds as investors lock in gains
            - A shift from Greed to Fear may increase volatility and downside risk
            
            📊 **Suggested Action**  
            Book profits if returns exceed **{min_threshold:.0f}%**, and rotate into **capital-preserving strategies**
                    """)
            
            elif self.current_mood == 'Fear':
                if self.current_streak < fear_mean:
                    st.success("""
            🟢 **Market in Fear** but early in the cycle –  
            Great opportunity to **accumulate stocks** with fresh capital.
                    """)
                else:
                    st.success("""
            📘 **Market in Fear** –  
            Be cautious and only accumulate **selectively**,  
            as the fear phase may be maturing.
                    """)

            
# 🧩 Finally — show allocation planner
allocation_collection = db['allocation_plans']

def save_allocation_plan(user_id, plan_df, total_amount, days, mmi_snapshot):
    allocation_collection.insert_one({
        'user_id': user_id,
        'timestamp': datetime.utcnow(),
        'investable_amount': total_amount,
        'total_days': days,
        'mmi_snapshot': mmi_snapshot,
        'confidence_date': mmi_snapshot.get('confidence_date'),  # ✅ save it directly
        'plan': plan_df.to_dict(orient='records')
    })

def get_latest_allocation_plan(user_id):
    rec = allocation_collection.find_one({'user_id': user_id}, sort=[('timestamp', -1)])
    return pd.DataFrame(rec['plan']) if rec else None
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
with st.expander("📂 Upload Full MMI Dataset", expanded=False):
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
st.markdown("📊 [Visit Tickertape Market Mood Index](https://www.tickertape.in/market-mood-index)")

with st.expander("📝 Add Today's MMI", expanded=False):
    with st.form("add_today_mmi"):
        today_mmi = st.number_input("Today's MMI", min_value=0.0, max_value=100.0, step=0.1)
        today = datetime.today().date()
        submitted = st.form_submit_button("📥 Save to MongoDB")

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

            st.success(f"✅ Saved MMI: {today_mmi} | Nifty: {nifty_today:.2f}")

            # Refresh analyzer with updated data
            df_from_db = read_mmi_from_mongodb()
            if not df_from_db.empty:
                analyzer = MarketMoodAnalyzer(df_from_db)
                st.success("🔄 Analyzer updated with new data")
            else:
                analyzer = None
                st.warning("⚠️ MongoDB returned no data")

        except Exception as e:
            analyzer = None
            st.error(f"❌ Error: {e}")

# ========== Display Mood Analysis ==========
# 🧩 Hook into Streamlit logic after analyzer.display_mood_analysis()
if analyzer:
    analyzer.display_mood_analysis()

    if analyzer.current_mood == "Fear":
        st.success("🟢 MMI indicates Fear – You may plan allocation")

        st.markdown("""
            💡 **Planning Your Investment**
            
            Since the market is currently in a *Fear* phase (MMI < 50), it may be a good time to start deploying capital gradually.
            Enter the total amount you'd like to invest.  
            The tool will generate a staggered buy allocation plan across the next few market days based on historical mood patterns.
            
            This approach helps reduce timing risk and allows disciplined entry during uncertain times.
            """)


        with st.form("allocation_plan_form"):
            amt = st.number_input("Investable Amount (₹)", min_value=100.0, step=1000.0)
            submit_alloc = st.form_submit_button("Generate Allocation Plan")

        if submit_alloc:
            alloc_df, confidence_date = analyzer.generate_allocation_plan(amt)
            st.dataframe(alloc_df)
        
            # Extract total days from the plan
            total_days = len(alloc_df)
        
            save_allocation_plan("default_user", alloc_df, amt, total_days, {
                'mmi': analyzer.current_mmi,
                'mood': analyzer.current_mood,
                'streak': analyzer.current_streak,
                'date': analyzer.mmi_last_date.strftime('%Y-%m-%d'),
                'confidence_date': confidence_date.strftime('%Y-%m-%d')  # ✅ New field
            })
        
            st.download_button("Download Allocation CSV", alloc_df.to_csv(index=False), file_name="allocation_plan.csv")

        with st.expander("🗂 View Last Saved Allocation Plan"):
            prev = get_latest_allocation_plan("default_user")
            if prev is not None:
                st.dataframe(prev)
            else:
                st.info("No saved plan yet.")

# ==================== STOCK RECOMMENDATIONS FROM GOOGLE SHEET ====================
st.markdown("## 📌 Recommended Stocks to Explore")

# Define Google Sheet info
sheet_id = "1f1N_2T9xvifzf4BjeiwVgpAcak8_AVaEEbae_NXua8c"
sheet_name = "Sheet1"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"

# Show link to open Google Sheet
st.markdown(f"🔗 [Open Google Sheet]({sheet_url})")

try:
    # Load and clean data
    df_reco = pd.read_csv(csv_url)
    column_a = df_reco.iloc[0:, 0].dropna().astype(str).str.strip()  # A2:A
    column_a = column_a[column_a != ""]  # Filter empty strings

    if not column_a.empty:
        st.success("✅ Loaded stock recommendations from the sheet")
        st.markdown("These are **community-sourced stock ideas**. Use them as a starting point, not financial advice.")
        st.write("🔍 **Suggested Stocks:**")
        st.markdown(
            f"<ul style='list-style-type: square; padding-left: 1.5em;'>"
            + "".join([f"<li>{s}</li>" for s in column_a])
            + "</ul>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ No non-empty stock entries found in A2:A")

except Exception as e:
    st.error("❌ Failed to load Google Sheet data.")
    st.code(str(e), language='text')

uploaded_holdings = None  # ✅ Initialize at top (before condition)

if analyzer and analyzer.current_mood == "Greed":
    st.header("📤 Upload Your Holdings")

    uploaded_holdings = st.file_uploader(
        "Upload your stock holdings file (.xlsx format only — Groww or Kite)",
        type=['xlsx'],
        key="upload_holdings_greed"
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
