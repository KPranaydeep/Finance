from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

# Connect to MongoDB
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
client = MongoClient(uri, server_api=ServerApi('1'))

# Ping to confirm connection
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ MongoDB connection error:", e)

# Define database and collection
db = client['finance_db']
collection = db['stock_performance']

# Load data from MongoDB
data = list(collection.find({}, {"_id": 0}))
df = pd.DataFrame(data)

# Handle first time empty
if df.empty:
    df = pd.DataFrame(columns=['Date', 'Buy', 'Sell', 'Charges'])

# Parse date properly
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

if st.button("Add Entry"):
    new_entry = {
        'Date': input_date.strftime("%Y-%m-%d"),
        'Buy': float(buy_value),
        'Sell': float(sell_value),
        'Charges': float(charges)
    }
    collection.insert_one(new_entry)
    st.success("✅ Entry added to MongoDB!")
    st.experimental_rerun()  # Refresh to show new data
