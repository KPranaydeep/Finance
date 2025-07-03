import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# --- Streamlit page setup ---
st.set_page_config(page_title="📈 Stock Performance Tracker", layout="centered")
st.title("📊 Stock Performance Tracker")

st.markdown("## 🎯 July 2025 Targets")

st.markdown("""
### 🎯 **Target Summary (By 31 July 2025)**

| Scenario       | **Target ROI (%)** | **Target Annualized Return (%)** |
|----------------|--------------------|----------------------------------|
| 🟢 Conservative | `6.8`              | `30.5 – 31.0`                     |
| 🟡 Realistic    | `7.2`              | `32.5 – 33.5`                     |
| 🔴 Stretch      | `7.6`              | `34.5 – 36.0`                     |
""")

st.markdown("""
### 📌 Notes:
- **ROI** = (Cumulative Net Profit / Buy Value) × 100
- **Annualized Return** accounts for time decay — achieving ROI earlier in the month improves the annualized figure.
- With ~29 days left in July, **ROI acceleration early in the month** is key to surpassing stretch targets.
""")

# --- MongoDB Setup ---
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    st.success("✅ Connected to MongoDB")
except Exception as e:
    st.error("❌ MongoDB Connection Failed")
    st.stop()

db = client['finance_db']
collection = db['stock_performance']

# --- Load Data from MongoDB ---
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

if df.empty:
    df = pd.DataFrame(columns=['Date', 'Buy', 'Sell', 'Charges'])

# --- Ensure Correct Data Types ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Buy'] = pd.to_numeric(df['Buy'], errors='coerce')
df['Sell'] = pd.to_numeric(df['Sell'], errors='coerce')
df['Charges'] = pd.to_numeric(df['Charges'], errors='coerce')
df = df.dropna(subset=['Date'])

# --- Collapsible Input Section ---
with st.expander("➕ Add New Entry"):
    default_date = datetime.date.today() - datetime.timedelta(days=1)
    input_date = st.date_input("📅 Date", value=default_date)
    buy_value = st.number_input("💰 Buy Value", min_value=0.0, format="%.2f")
    sell_value = st.number_input("💸 Sell Value", min_value=0.0, format="%.2f")
    charges = st.number_input("⚙️ Charges", min_value=0.0, format="%.2f")
    
    if st.button("✅ Add Entry"):
        new_row = pd.DataFrame([{
            'Date': input_date,
            'Buy': buy_value,
            'Sell': sell_value,
            'Charges': charges
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        collection.insert_one({
            'Date': input_date.strftime('%Y-%m-%d'),
            'Buy': buy_value,
            'Sell': sell_value,
            'Charges': charges
        })
        st.experimental_rerun()
        st.success("✅ Entry added!")

# --- Calculations ---
if not df.empty:
    df['Net Profit'] = df['Sell'] - df['Buy'] - df['Charges']
    df['ROI'] = df['Net Profit'] / df['Buy']
    if not df.empty:
        df['ROI'] = (df['Sell'] - df['Buy'] - df['Charges']) / df['Buy']
        max_roi = df['ROI'].max()
    else:
        max_roi = 0  # fallback
    def get_max_roi():
        data = list(collection.find({}, {"_id": 0}))
        df = pd.DataFrame(data)
    
        if df.empty:
            return 0
    
        df['Buy'] = pd.to_numeric(df['Buy'], errors='coerce')
        df['Sell'] = pd.to_numeric(df['Sell'], errors='coerce')
        df['Charges'] = pd.to_numeric(df['Charges'], errors='coerce')
    
        df = df.dropna(subset=['Buy', 'Sell', 'Charges'])
    
        df['ROI'] = (df['Sell'] - df['Buy'] - df['Charges']) / df['Buy']
        return df['ROI'].max()

    df['Charges %'] = df['Charges'] / df['Buy'] * 100
    df['Days Held'] = (df['Date'] - pd.to_datetime("2025-04-01")).dt.days + 1
    df['Annualized Return'] = ((1 + df['ROI']) ** (365 / df['Days Held'])) - 1

    # ✅ Sort by Date and remove duplicates (keep last entry if duplicate dates exist)
    df_plot = df.sort_values('Date').groupby('Date', as_index=False).last()

    import matplotlib.dates as mdates

    # --- Plot 1: Charges % over Time ---
    st.subheader("📉 Charges % Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=150)
    ax1.plot(df_plot['Date'].values, df_plot['Charges %'].values, marker='o', linestyle='-', color='crimson')
    ax1.set_ylabel("Charges (% of Buy)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title("Charges % Over Time", fontsize=14, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    # --- Plot 2: Annualized Return over Time ---
    st.subheader("📈 Annualized Return Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=150)
    ax2.plot(df_plot['Date'].values, df_plot['Annualized Return'].values * 100, marker='s', linestyle='-', color='darkgreen')
    ax2.set_ylabel("Annualized Return (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Annualized Return vs Time", fontsize=14, weight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    # --- Plot 3: ROI over Time ---
    st.subheader("📊 ROI Over Time")
    fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=150)
    ax3.plot(df_plot['Date'].values, df_plot['ROI'].values * 100, marker='^', linestyle='-', color='navy')
    ax3.set_ylabel("ROI (%)", fontsize=12)
    ax3.set_xlabel("Date", fontsize=12)
    ax3.set_title("ROI vs Time", fontsize=14, weight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig3.autofmt_xdate()
    
    # --- Annotate max ROI ---
    max_roi = (df_plot['ROI'].max()) * 100  # In percentage
    ax3.text(
        1.0, -0.15,  # X and Y relative coordinates (bottom right)
        f"Max ROI: {max_roi:.2f}%",
        ha='right', va='center',
        fontsize=10, color='green',
        transform=ax3.transAxes,
        bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3')
    )
    
    st.pyplot(fig3)

    # --- Display Table ---
    # Keep only the latest entry per date (in case of multiple entries per date)
    df_display = df.sort_values('Date').groupby('Date', as_index=False).last()
    df_display['Buy'] = df_display['Buy'].map('₹{:,.2f}'.format)
    df_display['Sell'] = df_display['Sell'].map('₹{:,.2f}'.format)
    df_display['Charges'] = df_display['Charges'].map('₹{:,.2f}'.format)
    df_display['Net Profit'] = df_display['Net Profit'].map('₹{:,.2f}'.format)
    df_display['ROI'] = (df['ROI'] * 100).round(2).astype(str) + '%'
    df_display['Charges %'] = df['Charges %'].round(2).astype(str) + '%'
    df_display['Annualized Return'] = (df['Annualized Return'] * 100).round(2).astype(str) + '%'

    # with st.expander("📋 Show Performance Table"):
    #     st.dataframe(df_display)


