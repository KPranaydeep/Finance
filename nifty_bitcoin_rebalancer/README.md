# Nifty 50 + Alternative Asset Portfolio Rebalancer

A Streamlit version of the supplied notebook workflow.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app downloads Nifty 50 (`^NSEI`) and a selectable Yahoo Finance ticker. It
includes presets for Bitcoin, gold, and silver, plus a custom ticker input. The
monthly constrained minimum-variance workflow supports a rolling covariance
window, transaction costs, and out-of-sample selection of the rebalance limit.
