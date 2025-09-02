import numpy as np
import yfinance as yf
import requests
import os
from io import BytesIO
from itertools import groupby
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json

import streamlit as st
# --- CONFIG ---
GROWTH_RATE = 0.456  # 4% per market day
import datetime
import pytz
import pandas as pd

#---------------
import streamlit as st
import pandas as pd
import plotly.express as px

# Google Sheet CSV export link
allocation_url = "https://docs.google.com/spreadsheets/d/1tpxU2_BEopIMRBF1cvMZXvMF3GCUM3xpcKthw_BYZZw/gviz/tq?tqx=out:csv&sheet=Savings"

# Read the sheet into Pandas
df = pd.read_csv(allocation_url, usecols=["G", "H"], skiprows=17, nrows=6)
df.columns = ["Asset", "Allocation"]

# Clean Allocation column
df["Allocation"] = (
    df["Allocation"].astype(str).str.replace("%", "", regex=False).str.strip()
)

# Convert to numeric
df["Allocation"] = pd.to_numeric(df["Allocation"], errors="coerce")

# Drop missing rows
df = df.dropna(subset=["Allocation"])

# ✅ Ensure this is a real Pandas DataFrame
df = pd.DataFrame(df.copy())

# --- Treemap ---
fig = px.treemap(
    df,
    path=["Asset"],
    values="Allocation",
    title="Portfolio Allocation Treemap",
)

fig.update_traces(
    texttemplate="<b>%{label}</b><br>%{value:.2f}%",
    textposition="middle center"
)

st.plotly_chart(fig, use_container_width=True)

#---------------
# --- HELPERS ---
def get_market_status(now: datetime.datetime) -> str:
    """Return market status based on IST time and weekday."""
    weekday = now.weekday() # Monday=0 ... Sunday=6
    hour, minute = now.hour , now.minute

    if weekday == 5:  # Saturday
        return "after_market_close"
    if weekday == 6:  # Sunday
        return "pre_market"

    if hour < 9 or (hour == 9 and minute < 15):
        return "pre_market"
    elif (hour > 15) or (hour == 15 and minute >= 30):
        return "after_market_close"
    else:
        return "market_hours"

# --- MAIN APP ---
def main():
    st.header("📈 Daily Profit Booking Assistant")

    # Input: Last 30 days Net P&L
    last_30_days_netpl = st.number_input(
        "Enter last 30 days Net P&L (₹)", value=0.0, step=100.0
    )

    # Baseline = average daily profit of last 30 days
    baseline = last_30_days_netpl / 30 if last_30_days_netpl > 0 else 0

    # Today’s target = baseline × (1 + growth_rate)
    today_target = baseline * (1 + GROWTH_RATE) if baseline > 0 else 0

    # Current IST time
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    status = get_market_status(now)

    # --- Display Guidance ---
    st.subheader("🎯 Daily Guidance")
    st.write(f"🗓️ {now.strftime('%A, %d %B %Y')}")
    st.write(f"⏰ Current Time (IST): {now.strftime('%I:%M %p')}")

    if status == "pre_market":
        if now.weekday() == 6:  # Sunday
            st.success(
                "📊 Market is closed today (Sunday).\n\n"
                "Here’s your **profit booking plan for next week** 👇"
            )

            # Generate Monday–Friday targets
            targets = []
            for i in range(1, 6):
                target = baseline * ((1 + GROWTH_RATE) ** i)
                day_name = (now + datetime.timedelta(days=i)).strftime("%A")
                targets.append({"Day": day_name, "Target (₹)": f"{target:,.0f}"})

            df = pd.DataFrame(targets)
            st.table(df)

            st.info("✅ Stick to these daily targets and avoid greed. 🌱")

        else:
            st.success(
                f"✅ Book **₹{today_target:,.0f}** profit when market opens.\n\n"
            )

    elif status == "market_hours":
        st.warning(
            f"🎯 Target for today: **₹{today_target:,.0f}**.\n\n"
            f"If you’ve already booked it: Why are you still here? 🚪 "
            f"Come back tomorrow. Life is more than money. 🌱"
        )

    elif status == "after_market_close":
        if now.weekday() == 5:  # Saturday
            st.info(
                f"📉 Market is closed for the weekend. 🌃 \n\n"
                f"Come back on **Monday at 9:15 AM** to book \n"
                f"**₹{today_target:,.0f}** profit."
            )
        else:
            st.info(
                f"📉 Market is closed. 🌃 \n\n Relax and enjoy your evening. \n"
                f"Come back tomorrow at 9:15 AM to book \n"
                f"**₹{today_target:,.0f}** profit."
            )

    else:
        st.error("⚠️ Unknown status. Please check system time.")

if __name__ == "__main__":
    main()

from datetime import datetime, timedelta
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

# --- Cached Leverage Decision Function ---
@st.cache_data
def should_use_leverage(ticker="^NSEI", days=200, cap=3.0):
    try:
        data = yf.download(ticker, period="400d")
        if 'Close' not in data.columns:
            raise ValueError("Missing 'Close' column in downloaded data.")

        close_series = data['Close'].copy()
        ma_series = close_series.rolling(window=days).mean()

        df = pd.DataFrame(index=close_series.index)
        df['Close'] = close_series
        df['200_MA'] = ma_series
        df['pct_above_ma'] = (df['Close'] - df['200_MA']) / df['200_MA']

        valid_rows = df.dropna(subset=['200_MA'])

        if valid_rows.empty:
            raise ValueError("No valid rows to calculate leverage — insufficient data.")

        latest_row = valid_rows.iloc[-1]
        latest_close = float(latest_row['Close'])
        latest_ma = float(latest_row['200_MA'])
        current_pct_above = float(latest_row['pct_above_ma'])

        # Max observed % above MA (only positive cases)
        positive_pct = valid_rows[valid_rows['pct_above_ma'] > 0]['pct_above_ma']
        max_pct_above_ma = float(positive_pct.max()) if not positive_pct.empty else 0.10

        alpha = cap / max_pct_above_ma if max_pct_above_ma > 0 else 0.0
        leverage_flag = latest_close > latest_ma

        return {
            "should_leverage": leverage_flag,
            "latest_close": round(latest_close, 2),
            "ma_value": round(latest_ma, 2),
            "pct_above_ma": current_pct_above,
            "max_pct_above_ma": max_pct_above_ma,
            "alpha": alpha
        }

    except Exception as e:
        return {
            "should_leverage": False,
            "latest_close": None,
            "ma_value": None,
            "pct_above_ma": None,
            "max_pct_above_ma": None,
            "alpha": 0.0,
            "error": str(e)
        }

# --- LAMF Allocation Calculator ---
def compute_lamf_pct(pct_above_ma, mmi, alpha, cap=3.0):
    fear_factor = 1 - (mmi / 100)  # MMI=0 → max fear, MMI=100 → max greed
    lamf_pct = alpha * pct_above_ma * fear_factor
    return min(max(lamf_pct, 0.0), cap)


st.set_page_config(layout="wide", page_icon=":moneybag:")
st.header("📊 Stock Holdings Analysis & Market Mood Dashboard")

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
                # 📈 Expected Streak with 95% Confidence Interval
                def compute_95_ci(data):
                    n = len(data)
                    if n < 2:
                        return None, None  # Not enough data
                    mean = np.mean(data)
                    std = np.std(data, ddof=1)
                    stderr = std / np.sqrt(n)
                    margin = 1.96 * stderr  # 95% confidence
                    return mean, (mean - margin, mean + margin)

                fear_mean_ci, fear_ci_range = compute_95_ci(fear_runs)
                greed_mean_ci, greed_ci_range = compute_95_ci(greed_runs)

                st.markdown("**📐 Expected Streak Duration (95% Confidence Interval)**")
                ci_df = pd.DataFrame({
                    "Mood": ["Fear", "Greed"],
                    "Mean Streak (days)": [f"{fear_mean_ci:.1f}", f"{greed_mean_ci:.1f}"],
                    "95% CI (days)": [
                        f"{fear_ci_range[0]:.1f} – {fear_ci_range[1]:.1f}" if fear_ci_range else "N/A",
                        f"{greed_ci_range[0]:.1f} – {greed_ci_range[1]:.1f}" if greed_ci_range else "N/A"
                    ]
                })
                st.table(ci_df)

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
            st.info("### 💰 Capital Allocation")

            if self.current_mmi < 50:
                invest_pct = (50 - self.current_mmi) * 2
                st.info(f"""
                😊  **Fear in Market MMI = {self.current_mmi:.2f}**  
                👉 **Invest `{invest_pct:.1f}%`** of your deployable cash.  
                🪙 Fear offers value buys, consider accumulating high-quality assets at lower valuations.
                """)
            elif self.current_mmi > 50:
                liquid_hold_pct = (self.current_mmi - 50) * 2
                st.info(f"""
                😬  **Greed in Market MMI = {self.current_mmi:.2f}**
                
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
            
                if self.current_streak < greed_ci_range[0]:
                    st.warning(f"""
            📉 **Market in Greed** – but still early in the cycle.  
            This phase is ideal for:
            
            - 🏦 **Booking profits** on outperformers  
            - 🔁 **Rotating into safer assets**  
            - 💵 **Holding cash** to prepare for possible pullbacks  
            
            📊 **Action Tip**  
            If your portfolio has gained over **{threshold:.0f}%**, it’s wise to secure some gains.  
            For more active strategies, start rotating once returns cross **{active_threshold:.0f}%** to stay agile and reduce downside risk.
                    """)
                else:
                    st.warning(f"""
            🛑 **Market in Greed** – Current streak: `{self.current_streak}` days  
            **Average**: `{greed_mean:.0f}` days
            
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
                if self.current_streak < fear_ci_range[0]:
                    st.success("""
            🟢 **Market in Fear – Early Stage**  
            Great time to **accumulate quality stocks** with fresh capital.  
            Sentiment is low — stay calm and think long term.
                    """)
                else:
                    st.success("""
            📘 **Market in Fear – Mature Phase**  
            Be cautious. Accumulate **selectively** and avoid panic selling.  
            
            ✅ If you find a better opportunity, consider switching  
            only if current holdings show **+7% or more profit**.  
            
            🧘‍♂️ Don’t sell just out of fear.  
            📉 Reassess fundamentals before averaging down.  
            ⏳ This phase may stretch — protect capital, stay alert.
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

# Google Sheet configuration
sheet_id = "1f1N_2T9xvifzf4BjeiwVgpAcak8_AVaEEbae_NXua8c"
sheet_name = "Sheet1"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing"

# Link to open the original Google Sheet
st.markdown(f"🔗 [Open Google Sheet]({sheet_url})")

try:
    # Load CSV without header (Google Sheets export format)
    df_raw = pd.read_csv(csv_url, header=None)

    # Slice rows 1 to 991 (i.e., A2 to A992), columns A to D (0:4)
    df_reco = df_raw.iloc[2:992, 0:4]
    df_reco.columns = ["Stock", "Buy", "Sell", "Recommendation"]

    # Clean and format
    df_reco.dropna(subset=["Stock"], inplace=True)
    df_reco["Stock"] = df_reco["Stock"].astype(str).str.strip()
    df_reco["Buy"] = df_reco["Buy"].astype(str).str.strip()
    df_reco["Sell"] = df_reco["Sell"].astype(str).str.strip()
    df_reco = df_reco[df_reco["Stock"] != ""]
    df_reco.reset_index(drop=True, inplace=True)

    if not df_reco.empty:
        st.markdown("These are **community-sourced stock ideas**. Use them as a starting point, not financial advice.")

        # Apply minimal styling
        styled_df = df_reco.style.set_table_styles([
            {"selector": "th", "props": [("padding", "4px"), ("font-size", "13px")]},
            {"selector": "td", "props": [("padding", "4px"), ("font-size", "13px")]}
        ])

        # Display the styled table
        st.dataframe(styled_df, use_container_width=True)

    else:
        st.warning("⚠️ No valid stock entries found between rows A2 to D992.")

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

import openpyxl

def extract_net_pl_and_charges(file) -> tuple:
    wb = openpyxl.load_workbook(file, data_only=True)
    ws = wb.active  # Or wb['Sheet1'] if known

    net_pl = ws['B9'].value or 0.0
    charges = ws['B26'].value or 0.0

    return float(net_pl), float(charges)

# ==== PROFIT BOOKING SECTION ====
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
        col3.metric("📊 Overall P&L", f"₹{total_pl:,.2f}", delta=f"{(total_pl / total_invested) * 100:.2f}%")

        USER_ID = "default_user"

        # 📂 Upload optional Net P&L Report
        uploaded_report = st.file_uploader("📄 Upload your P&L Report (B9 = Net P&L, B26 = Charges)", type=["xlsx"])

        # Load defaults from saved MongoDB values
        latest_params = get_latest_input_params(USER_ID)
        net_pl_default = float(latest_params.get('net_pl', 0.0))
        charges_default = float(latest_params.get('charges', 0.0))

        # Override if report uploaded
        if uploaded_report is not None:
            try:
                net_pl_from_file, charges_from_file = extract_net_pl_and_charges(uploaded_report)
                net_pl_default = net_pl_from_file
                charges_default = charges_from_file
                st.success("✅ Auto-filled Net P&L and Charges from uploaded report.")
            except Exception as e:
                st.error(f"⚠️ Failed to extract values from report: {e}")

        st.subheader("🎯 Profit Booking Strategy")
        st.subheader("🔧 Adjust Profit Booking Parameters")

        # User inputs
        net_pl = st.number_input("Enter net P&L (INR)",
                                 value=net_pl_default,
                                 min_value=0.0,
                                 step=1000.0)

        charges_input = st.number_input("Enter charges (INR)",
                                        value=charges_default,
                                        min_value=0.0,
                                        step=100.0)

        target_net_daily_pct = st.number_input("Target net daily return (%)",
                                               value=float(latest_params.get('target_pct', 0.28)),
                                               min_value=0.01,
                                               max_value=5.0,
                                               step=0.01)

        # Save input values
        save_input_params(USER_ID, net_pl, charges_input, target_net_daily_pct)

        if net_pl > 0:
            sell_limit_multiplier = calculate_dynamic_sell_limit(net_pl, charges_input, target_net_daily_pct)
            daily_return_pct = round((sell_limit_multiplier - 1) * 100, 4)
            st.markdown(f"💡 *Dynamic sell limit calculated at {daily_return_pct}% above buy price*")

            rotation_option = st.radio(
                "Select rotation strategy for calculating target profit:",
                ["Daily", "Weekly", "Monthly"],
                index=0,
                horizontal=True
            )

            # Determine profit target
            if rotation_option == "Daily":
                default_target = round(total_invested * (sell_limit_multiplier - 1), 2)
            elif rotation_option == "Weekly":
                default_target = round(total_invested * (sell_limit_multiplier - 1) * 4.84615385, 2)
            else:
                default_target = round(total_invested * (sell_limit_multiplier - 1) * 21, 2)

            target_rupees = st.number_input("Enter target profit (₹)",
                                            value=default_target,
                                            min_value=0.0,
                                            step=100.0)

            # Filter profitable holdings
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
                st.download_button("📥 Download Sell Plan", csv, "sell_plan.csv", "text/csv", key='download-sell-plan')
            else:
                st.warning("📉 Not enough profitable stocks to meet target")
                st.info("⏳ Check back tomorrow when market conditions may improve")
        else:
            st.error("❌ Cannot calculate sell limit with zero or negative P&L")

# --- UI Section ---
with st.expander("⚖️ Leverage Decision Based on NIFTY 200-Day MA", expanded=False):
    result = should_use_leverage()

    if result.get("error"):
        st.error(f"⚠️ Error fetching data: {result['error']}")
    else:
        st.metric("📈 NIFTY Close", f"{result['latest_close']}")
        st.metric("📊 200-Day MA", f"{result['ma_value']}")
        st.metric("🔺 Max % Above MA", f"{result['max_pct_above_ma'] * 100:.2f}%")

        if result["should_leverage"]:
            st.success("✅ NIFTY is above its 200-day MA → Leverage allowed")

            mmi = st.number_input("📊 Market Mood Index (MMI)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            mf_corpus = st.number_input("💼 Enter Mutual Fund Corpus (₹)", value=10_00_000.0, step=10_000.0)

            lamf_pct = compute_lamf_pct(
                result["pct_above_ma"],
                mmi,
                result["alpha"]
            )
            lamf_amt = mf_corpus * lamf_pct

            st.metric("📌 LAMF % Recommended", f"{lamf_pct * 100:.2f}%")
            st.metric("💸 Max LAMF Amount", f"₹{lamf_amt:,.0f}")

        else:
            st.warning("🛑 NIFTY is below 200-DMA → Avoid leverage")
            st.markdown("💼 Stay defensive: shift to cash, T-Bills, or liquid funds.")
            
import streamlit as st
from datetime import datetime, timedelta

# 🎯 Input your target profit %
target_profit_percent = 7.2

# 📆 Auto-calculate holding period
holding_days = round((target_profit_percent * 7) / 5 / 0.278)

# 📅 Calculate dates
today = datetime.today().date()
exit_date = today + timedelta(days=holding_days)

# 📌 Display holding rule
st.markdown(f"""
### 📌 Sell or Hold Strategy

- ✅ **Hold for {holding_days} days** from today (**{today.strftime('%b %d, %Y')}**)  
  ➤ Target exit on **{exit_date.strftime('%b %d, %Y')}**

- 📉 **If in loss after {holding_days} days**, continue to **hold until breakeven**

- 💰 **Sell anytime** after if **net profit ≥ {target_profit_percent}%** using brokerage-inclusive estimates:
  - [Groww Brokerage Calculator](https://groww.in/calculators/brokerage-calculator)
  - [Zerodha Brokerage Calculator](https://zerodha.com/brokerage-calculator/#tab-equities)
""")

