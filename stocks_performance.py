import streamlit as st
import pandas as pd
import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).parent
DATA_FILE = APP_DIR / 'performance_data.csv'

# Load existing data or create new DataFrame
if DATA_FILE.exists():
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=['Date', 'Buy', 'Sell', 'Charges'])

# Ensure Date is datetime and drop invalid
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

st.write("Rows after removing invalid dates:", len(df))

st.success(f"Entry added! Saved to: {DATA_FILE}")

# --- Input Section ---
st.title("Stock Performance Tracker")

# Default to yesterday
default_date = datetime.date.today() - datetime.timedelta(days=1)
input_date = st.date_input("Date", value=default_date)

buy_value = st.number_input("Buy Value", min_value=0.0, format="%.2f")
sell_value = st.number_input("Sell Value", min_value=0.0, format="%.2f")
charges = st.number_input("Charges", min_value=0.0, format="%.2f")

if st.button("Add Entry"):
    new_row = pd.DataFrame([{
        'Date': input_date,
        'Buy': buy_value,
        'Sell': sell_value,
        'Charges': charges
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.success("Entry added!")

# --- Calculations ---
df['Net Profit'] = df['Sell'] - df['Buy'] - df['Charges']
df['ROI'] = df['Net Profit'] / df['Buy']
df['Charges %'] = df['Charges'] / df['Buy'] * 100
df['Days Held'] = (df['Date'] - pd.to_datetime("2025-04-01")).dt.days + 1
df['Annualized Return'] = ((1 + df['ROI']) ** (365 / df['Days Held'])) - 1

import matplotlib.dates as mdates

# --- Refined Plots ---
if not df.empty:
    st.subheader("Charges % of Buy Value Over Time")

    fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=150)
    ax1.plot(df['Date'], df['Charges %'], marker='o', linestyle='-', color='darkblue', linewidth=2, markersize=6)
    ax1.set_ylabel("Charges (% of Buy Value)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title("Charges % Over Time", fontsize=14, weight='bold')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig1.autofmt_xdate()
    st.pyplot(fig1)

    st.subheader("Annualized Return Over Time")

    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=150)
    ax2.plot(df['Date'], df['Annualized Return'] * 100, marker='s', linestyle='-', color='darkgreen', linewidth=2, markersize=6)
    ax2.set_ylabel("Annualized Return (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Annualized Return vs Time", fontsize=14, weight='bold')
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    fig2.autofmt_xdate()
    st.pyplot(fig2)

    # --- Display formatted Data Table ---
    df_display = df.copy()
    df_display['Buy'] = df_display['Buy'].map('₹{:,.2f}'.format)
    df_display['Sell'] = df_display['Sell'].map('₹{:,.2f}'.format)
    df_display['Charges'] = df_display['Charges'].map('₹{:,.2f}'.format)
    df_display['Net Profit'] = df_display['Net Profit'].map('₹{:,.2f}'.format)
    df_display['ROI'] = (df['ROI'] * 100).round(2).astype(str) + '%'
    df_display['Charges %'] = df['Charges %'].round(2).astype(str) + '%'
    df_display['Annualized Return'] = (df['Annualized Return'] * 100).round(2).astype(str) + '%'

    st.subheader("Formatted Performance Table")
    st.dataframe(df_display)
