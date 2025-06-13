import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# MongoDB connection
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
client = MongoClient(uri, server_api=ServerApi('1'))

# Connect to DB & collection
db = client['finance_db']
collection = db['stock_performance']

# Read your CSV data
csv_data = """
Date,Buy,Sell,Charges
2025-05-28,697382.44,739054.06,5594.4
2025-05-29,697382.44,739054.06,5611.04
2025-06-05,745414.69,787800.92,6042.56
2025-06-07,751492.93,794472.95,6135.66
2025-06-09,783039.63,828429.75,6394.62
2025-06-10,813541.67,863852.47,6629.08
2025-06-12,844662.99,899626.61,6857.65
"""

from io import StringIO
df = pd.read_csv(StringIO(csv_data))
df['Date'] = pd.to_datetime(df['Date'])

# Convert each row to a dict and insert into MongoDB
records = df.to_dict(orient='records')
# Optional: remove any existing records first (CAUTION!)
# collection.delete_many({})
collection.insert_many(records)

print(f"✅ Imported {len(records)} records to MongoDB."
