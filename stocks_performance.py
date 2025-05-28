import streamlit as st
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

DATA_FILE = 'performance_data.csv'

# Load existing data or create new DataFrame
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
else:
    df = pd.DataFrame(columns=['Date', 'Buy', 'Sell', 'Charges'])

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

# --- Plots ---
if not df.empty:
    st.subheader("Charges % of Buy Value Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['Date'], df['Charges %'], marker='o')
    ax1.set_ylabel("Charges (%)")
    ax1.set_xlabel("Date")
    st.pyplot(fig1)

    st.subheader("Annualized Return Over Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(df['Date'], df['Annualized Return'] * 100, marker='o')
    ax2.set_ylabel("Annualized Return (%)")
    ax2.set_xlabel("Date")
    st.pyplot(fig2)

    st.dataframe(df)
