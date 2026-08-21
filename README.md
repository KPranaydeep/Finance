# Personal Finance Toolkit 📊

A minimal Streamlit suite for smarter money planning:

- 💰 LAMF Simulator  
- 🎯 Target Corpus Planner
- 🧠 Smart SWP Planner (Inflation-Proof)
- 📈 Nifty 50 + alternative-assets rebalancer

🔗 [Launch App](https://finance-master.streamlit.app/)

## Nifty alternative-assets rebalancer

The root-level `nifty_alternative_rebalancer.py` Streamlit app compares Nifty 50
with selectable non-company assets such as precious metals, cryptocurrencies,
and ETFs. Selected alternatives are fetched together and presented as separate
Nifty–alternative dual portfolios in one result table and overlapping chart.

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run nifty_alternative_rebalancer.py
```

For Streamlit Community Cloud, use `nifty_alternative_rebalancer.py` as the main
file path.
