from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

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
