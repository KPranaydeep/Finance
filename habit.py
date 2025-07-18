import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calplot
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# === MongoDB Setup ===
uri = "mongodb+srv://hwre2224:jXJxkTNTy4GYx164@finance.le7ka8a.mongodb.net/?retryWrites=true&w=majority&appName=Finance"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["Finance"]
habits_collection = db["habit_tracker"]
votes_collection = db["habit_votes"]

# === Streamlit Config ===
st.set_page_config(page_title="Habit Tracker", layout="wide")
st.title("📈 Habit Builder: Vote for the Person You Want to Become")
st.markdown("Every action you take is a vote for the type of person you wish to become.")

# === Sidebar: Habit Management ===
with st.sidebar:
    st.header("✨ Manage Habits")
    user = st.text_input("Your Name")

    habit_to_delete = st.text_input("Delete a Habit (Exact Name)")
    if st.button("🗑️ Delete Habit"):
        if user and habit_to_delete:
            habits_collection.delete_many({"user": user, "habit": habit_to_delete})
            votes_collection.delete_many({"user": user, "habit": habit_to_delete})
            st.success(f"Deleted habit '{habit_to_delete}' and all its records.")
        else:
            st.error("Please provide both user and habit name.")

    st.divider()

    habit_to_reset = st.text_input("Reset Votes for Habit (Exact Name)")
    if st.button("♻️ Reset Votes Only"):
        if user and habit_to_reset:
            result = votes_collection.delete_many({"user": user, "habit": habit_to_reset})
            st.success(f"Reset successful! {result.deleted_count} vote(s) deleted for '{habit_to_reset}'.")
        else:
            st.error("Please provide both user and habit name.")

# === Expanded Section: Create New Habit ===
with st.expander("➕ Create a New Habit", expanded=False):
    st.subheader("🆕 Add a New Habit")
    creator_user = st.text_input("👤 Your Name (to create habit)", key="create_user")
    new_habit = st.text_input("📝 Habit Name", placeholder="e.g., Morning Pages", key="habit_name")
    description = st.text_area("🗒️ Description or Intention for the Habit", placeholder="Why are you building this habit?", key="habit_desc")

    if st.button("✅ Save Habit"):
        if creator_user and new_habit:
            if not habits_collection.find_one({"user": creator_user, "habit": new_habit}):
                habits_collection.insert_one({
                    "user": creator_user,
                    "habit": new_habit,
                    "description": description
                })
                st.success(f"Habit '{new_habit}' created successfully!")
            else:
                st.warning("This habit already exists.")
        else:
            st.error("Please enter both user and habit name.")

# === Main: Habit Vote ===
if user:
    habits = habits_collection.distinct("habit", {"user": user})
    if habits:
        selected_habit = st.selectbox("Select a Habit to Submit Vote", habits)
        repetitions = st.number_input("Repetitions Today", min_value=1, max_value=50, step=1, value=1)
        if st.button("✅ Record Today's Vote(s)"):
            today = datetime.now().date().isoformat()
            for _ in range(repetitions):
                votes_collection.insert_one({
                    "user": user,
                    "habit": selected_habit,
                    "date": today,
                    "timestamp": datetime.now()
                })
            st.success(f"{repetitions} vote(s) recorded for today!")

# === Main: Calendar Display ===
st.subheader("📅 Your Habit Calendar")
selected_user = st.text_input("Enter your name to view calendar")

if selected_user:
    records = list(votes_collection.find({"user": selected_user}))
    df = pd.DataFrame(records)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.date

        habit_counts = df.groupby("habit")["day"].count().reset_index().rename(columns={"day": "votes"})
        st.dataframe(habit_counts)

        for habit_name in habit_counts["habit"]:
            habit_doc = habits_collection.find_one({"user": selected_user, "habit": habit_name})
            st.markdown(f"### 📌 Habit: {habit_name}")
            if habit_doc and "description" in habit_doc:
                st.caption(f"🗒️ {habit_doc['description']}")

            habit_df = df[df["habit"] == habit_name]
            daily_counts = habit_df.groupby("date").size()
            daily_counts = daily_counts[daily_counts > 0]  # filter 0s
            daily_counts.index.name = "date"

            try:
                fig, ax = calplot.calplot(
                    daily_counts,
                    cmap="YlGn",
                    suptitle=f"{habit_name}",
                    colorbar=True,
                    textformat="{:.0f}",
                    textcolor="black",
                    figsize=(10, 2),
                    linewidth=0.1,
                    yearlabel_kws={"color": "black", "fontsize": 9}
                )
            
                for a in ax.flat:
                    for txt in a.texts:
                        txt.set_alpha(0.0)        # Make labels semi-transparent
                        txt.set_fontsize(8)
            
                total_votes = habit_counts.loc[habit_counts["habit"] == habit_name, "votes"].values[0]
                votes_left = max(0, 254 - total_votes)
            
                plt.figtext(
                    0.5, -0.05,
                    f"{votes_left} repetition(s) left to automate '{habit_name}'",
                    ha="center", fontsize=10, color="darkgreen"
                )
            
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"⚠️ Calendar plot failed for '{habit_name}'. Check your data.")
                st.exception(e)


            if habit_counts.loc[habit_counts["habit"] == habit_name, "votes"].values[0] >= 254:
                st.success(f"🎉 Habit '{habit_name}' has been automated (254 votes)! Keep it up!")
    else:
        st.info("No records found for this user yet.")

# === Footer Info ===
st.markdown("---")
st.markdown("✅ **One repetition** = recording the habit once that day. After 254 votes, the habit is considered **automated**.")
