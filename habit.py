import streamlit as st
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import calendar
import pandas as pd

# MongoDB setup
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client.get_database("Finance")
habits_collection = db.get_collection("habit_tracker")

# Streamlit config
st.set_page_config(page_title="Habit Vote Tracker", layout="wide")
st.title("📈 Habit Builder: Vote for the Person You Want to Become")
st.markdown("Every action you take is a vote for the type of person you wish to become.")

# Habit input section
with st.sidebar:
    st.header("✨ Create or Track Habits")
    user = st.text_input("Your Name")
    habit = st.text_input("New Habit Name", placeholder="e.g., Morning Pages")
    if st.button("➕ Add New Habit"):
        if user and habit:
            existing = habits_collection.find_one({"user": user, "habit": habit})
            if not existing:
                habits_collection.insert_one({"user": user, "habit": habit, "date": None})
                st.success(f"New habit '{habit}' added!")
            else:
                st.warning("This habit already exists.")
        else:
            st.error("Please enter both name and habit")

    st.divider()
    habit_to_delete = st.text_input("Delete a Habit (Exact Name)")
    if st.button("🗑️ Delete Habit"):
        if user and habit_to_delete:
            result = habits_collection.delete_many({"user": user, "habit": habit_to_delete})
            if result.deleted_count > 0:
                st.success(f"Deleted habit '{habit_to_delete}' and all its records.")
            else:
                st.warning("No such habit found.")

# Habit vote submission
if user:
    habits = habits_collection.distinct("habit", {"user": user, "date": {"$ne": None}})
    if habits:
        selected_habit = st.selectbox("Select a Habit to Submit Today's Vote", habits)
        if st.button("✅ Record Today's Vote"):
            today = datetime.now().date().isoformat()
            if not habits_collection.find_one({"user": user, "habit": selected_habit, "date": today}):
                habits_collection.insert_one({"user": user, "habit": selected_habit, "date": today})
                st.success("Vote recorded for today!")
            else:
                st.info("Vote already recorded today for this habit.")

# Main display
st.subheader("📅 Your Habit Calendar")
selected_user = st.text_input("Enter your name to view calendar")
if selected_user:
    records = list(habits_collection.find({"user": selected_user, "date": {"$ne": None}}))
    df = pd.DataFrame(records)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df["day"] = df["date"].dt.date
        df = df.drop_duplicates(["habit", "day"])

        habit_counts = df.groupby("habit")["day"].count().reset_index().rename(columns={"day": "votes"})
        st.dataframe(habit_counts)

        for habit_name in habit_counts["habit"]:
            st.markdown(f"### 📌 Habit: `{habit_name}`")
            habit_days = df[df["habit"] == habit_name]["day"].tolist()

            vote_calendar = pd.DataFrame({"date": habit_days})
            vote_calendar["check"] = 1
            vote_calendar.set_index("date", inplace=True)

            start_date = df["day"].min()
            end_date = datetime.now().date()
            full_range = pd.date_range(start=start_date, end=end_date)
            full_df = pd.DataFrame(index=full_range)
            merged = full_df.join(vote_calendar, how="left").fillna(0)

            merged["Month"] = merged.index.to_series().apply(lambda x: x.strftime('%b %Y'))
            months = merged["Month"].unique()

            for month in months:
                month_df = merged[merged["Month"] == month]
                st.markdown(f"#### {month}")
                st.bar_chart(month_df["check"])

            if habit_counts.loc[habit_counts["habit"] == habit_name, "votes"].values[0] >= 254:
                st.success(f"🎉 Habit '{habit_name}' has been automated (254 votes)! Keep it up!")
    else:
        st.info("No records found for this user yet.")

# Explanation
st.markdown("---")
st.markdown("✅ **One repetition** means recording the habit being done **once per day**. After 254 repetitions (votes), the habit is considered **automated**.")
