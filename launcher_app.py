import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="📊 Personal Finance Tools", layout="centered")

st.title("📊 Personal Finance Toolkit")
st.markdown("Quick access to all your personal finance tools. Click on a button below to launch an app in a new tab.")

# App list with names and URLs
apps = {
    "🎯 Target Corpus Planner": "https://finance-knqvcpyxszgty2sbj2gsqt.streamlit.app/",
    "📅 Stock Performance Tracker": "https://stocks-performance.streamlit.app/",
    "💰 Loan Against Mutual Funds Simulator": "https://lamfsimulatorpy-dupueqmb2r5eo52ad4tkvf.streamlit.app/",
    "🧓 Smart SWP Planner: Inflation-Proof": "https://finance-retire.streamlit.app/",
    "📈 Stock Holdings Analysis & Sell Plan": "https://sellplan.streamlit.app/"
    
}

# Button style
def render_button(name, url):
    components.html(
        f"""
        <a href="{url}" target="_blank" style="text-decoration: none;">
            <button style="
                background-color:#30475e;
                color:#f5f5f5;
                padding: 14px 28px;
                margin: 10px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                width: 100%;
                transition: background-color 0.3s;
                cursor: pointer;">
                {name}
            </button>
        </a>
        """,
        height=90,
    )

# Layout with 2 columns per row
cols = st.columns(2)
for idx, (name, url) in enumerate(apps.items()):
    with cols[idx % 2]:
        render_button(name, url)

st.markdown("---")
st.info("💡 Tip: All tools open in a **new tab** so you can run multiple tools in parallel.")
