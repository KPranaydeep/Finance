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
