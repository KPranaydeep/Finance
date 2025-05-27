import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="📊 Personal Finance Tools", layout="centered")

st.title("📊 Personal Finance Toolkit")
st.markdown("Use the buttons below to launch a tool in a new tab.")

# Define the local URLs assuming you're running each app on a different port
apps = {
    "💰 Loan Against Mutual Funds Simulator": "https://lamfsimulatorpy-dupueqmb2r5eo52ad4tkvf.streamlit.app/",
    "🧓 Smart SWP Planner: Inflation-Proof": "https://finance-retire.streamlit.app/",
    "🎯 Target Corpus Planner": "https://finance-knqvcpyxszgty2sbj2gsqt.streamlit.app/",
    "📈 Stock Holdings Analysis & Sell Plan": "https://sellplan.streamlit.app/"
}

# Button-styled links
for name, url in apps.items():
    components.html(
        f"""
        <a href="{url}" target="_blank">
            <button style="
                background-color:#222831;
                border:none;
                color:white;
                padding:10px 24px;
                text-align:center;
                text-decoration:none;
                display:inline-block;
                font-size:16px;
                margin:10px 2px;
                cursor:pointer;
                border-radius:8px;">
                {name}
            </button>
        </a>
        """,
        height=70,
    )
