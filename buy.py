# buy.py
import streamlit as st

def show_buy_plan(analyzer):
    st.subheader("💸 Suggested Buy Allocation Plan")

    investable_amount = st.number_input("Enter Investable Amount (₹)", min_value=1000, step=1000, value=50000)
    total_days = st.slider("Allocation Span (Market Days)", min_value=5, max_value=40, value=21)

    if st.button("Generate Buy Plan"):
        html_output = analyzer.generate_allocation_plan(investable_amount, total_days)
        st.components.v1.html(html_output, height=500, scrolling=True)

        # Optional download
        df = analyzer.generate_allocation_plan(investable_amount, total_days)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Allocation Plan (CSV)", data=csv, file_name="buy_plan.csv", mime="text/csv")
