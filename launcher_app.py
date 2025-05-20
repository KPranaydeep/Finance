import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Finance Tools Hub", layout="wide")

st.title("📊 Personal Finance Toolkit")
st.markdown("Select a tool below to launch it in a separate Streamlit session.")

apps = {
    "💰 LAMF Simulator": "lamf_simulator.py",
    "🧓 Retire With This": "retire_with_this.py",
    "🎯 Target Corpus Planner": "target_corpus.py"
}

choice = st.selectbox("Choose an App", list(apps.keys()))

if st.button("🚀 Launch Selected App"):
    script = apps[choice]
    st.success(f"Launching {script}...")
    
    # Launch in a new Streamlit tab or process (best for dev/testing)
    cmd = f"streamlit run {script}"
    os.system(cmd)
