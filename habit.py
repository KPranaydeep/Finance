import streamlit as st
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import calendar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# MongoDB setup
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client.get_database("Finance")
habits_collection = db.get_collection("habit_tracker")
votes_collection = db.get_collection("habit_votes")

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
                habits_collection.insert_one({"user": user, "habit": habit})
                st.success(f"New habit '{habit}' added!")
            else:
                st.warning("This habit already exists.")
        else:
            st.error("Please enter both name and habit")

    st.divider()
    habit_to_delete = st.text_input("Delete a Habit (Exact Name)")
    if st.button("🗑️ Delete Habit"):
        if user and habit_to_delete:
            habits_collection.delete_many({"user": user, "habit": habit_to_delete})
            votes_collection.delete_many({"user": user, "habit": habit_to_delete})
            st.success(f"Deleted habit '{habit_to_delete}' and all its records.")
        else:
            st.error("Please provide both user and habit name.")

# Habit vote submission
if user:
    habits = habits_collection.distinct("habit", {"user": user})
    if habits:
        selected_habit = st.selectbox("Select a Habit to Submit Vote", habits)
        repetitions = st.number_input("How many times did you repeat this habit today?", min_value=1, max_value=50, step=1, value=1)
        if st.button("✅ Record Today's Vote(s)"):
            today = datetime.now().date().isoformat()
            for _ in range(repetitions):
                votes_collection.insert_one({"user": user, "habit": selected_habit, "date": today, "timestamp": datetime.now()})
            st.success(f"{repetitions} vote(s) recorded for today!")

# Main display
st.subheader("📅 Your Habit Calendar")
selected_user = st.text_input("Enter your name to view calendar")
if selected_user:
    records = list(votes_collection.find({"user": selected_user}))
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
            habit_df = df[df["habit"] == habit_name]
            heatmap_df = habit_df.groupby("date").size().reset_index(name="count")
            heatmap_df.set_index("date", inplace=True)

            all_dates = pd.date_range(start=heatmap_df.index.min(), end=heatmap_df.index.max())
            heatmap_df = heatmap_df.reindex(all_dates, fill_value=0)
            heatmap_df.index.name = "date"

            calendar_df = heatmap_df.copy()
            calendar_df = calendar_df.reset_index()
            calendar_df["dow"] = calendar_df["date"].dt.weekday
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["year"] = calendar_df["date"].dt.isocalendar().year

            pivot = calendar_df.pivot_table(index="dow", columns=["year", "week"], values="count", fill_value=0)

            fig, ax = plt.subplots(figsize=(12, 2))
            sns.heatmap(pivot, cmap="Greens", cbar=False, linewidths=0.5, ax=ax)
            ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
            ax.set_title(f"Heatmap: {habit_name}", loc='left')
            st.pyplot(fig)

            if habit_counts.loc[habit_counts["habit"] == habit_name, "votes"].values[0] >= 254:
                st.success(f"🎉 Habit '{habit_name}' has been automated (254 votes)! Keep it up!")
    else:
        st.info("No records found for this user yet.")

# Explanation
st.markdown("---")
st.markdown("✅ **One repetition** means recording the habit being done **once per day**. After 254 repetitions (votes), the habit is considered **automated**.")
