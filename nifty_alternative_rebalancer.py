import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import Week

BASE = "^NSEI"
PRESETS = {
    # Cryptocurrencies in the pasted top-100 list
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    # Precious metals in the list, represented by Yahoo futures tickers
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    # ETFs in the pasted top-100 list
    "Vanguard S&P 500 ETF": "VOO",
    "iShares Core S&P 500 ETF": "IVV",
    "SPDR S&P 500 ETF": "SPY",
    "Vanguard Total Stock Market ETF": "VTI",
    "Invesco QQQ Trust": "QQQ",
    "Vanguard FTSE Developed Markets ETF": "VEA",
    "Vanguard Growth ETF": "VUG",
}

st.set_page_config(page_title="Nifty + Alternatives", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_batch(
    tickers: tuple[str, ...], start: str, end: str
) -> pd.DataFrame:
    """Fetch Nifty and every selected alternative in one Yahoo request."""
    assets = [BASE, *tickers]
    raw = yf.download(assets, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError("Yahoo Finance returned no data")
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.reindex(columns=assets)


def pair_prices(batch: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create one clean two-asset view from the shared batch."""
    if BASE not in batch.columns or ticker not in batch.columns:
        raise ValueError("ticker was not returned by Yahoo Finance")
    prices = batch[[BASE, ticker]].dropna()
    if prices.empty:
        raise ValueError("no overlapping Nifty and alternative-asset prices")
    return prices


def min_variance(history: pd.DataFrame) -> np.ndarray:
    inverse = np.linalg.pinv(history.cov())
    ones = np.ones(2)
    weights = inverse @ ones
    weights /= ones.T @ inverse @ ones
    return np.clip(weights, 0, 1)


def simulate(
    returns: pd.DataFrame,
    month_ends: pd.DatetimeIndex,
    limit_pct: float,
    window_months: int,
    cost_pct: float,
) -> tuple[pd.DataFrame, pd.Series]:
    weights = pd.DataFrame(index=month_ends, columns=returns.columns, dtype=float)
    allocation = np.array([1.0, 0.0])
    weights.iloc[0] = allocation
    output, dates = [], []

    for index in range(1, len(month_ends)):
        current, previous = month_ends[index], month_ends[index - 1]
        history = returns.loc[:current]
        if window_months:
            history = history.loc[current - pd.DateOffset(months=window_months) :]
        optimal = min_variance(history)
        max_shift = limit_pct / 100 * allocation[0]
        alt_weight = min(allocation[1] + max_shift, optimal[1])
        new_allocation = np.array([1 - alt_weight, alt_weight])
        weights.iloc[index] = new_allocation

        # Strict lower boundary prevents duplicated month-end returns.
        period = returns.loc[(returns.index > previous) & (returns.index <= current)]
        period_output = period @ allocation
        turnover = np.abs(new_allocation - allocation).sum() / 2
        if len(period_output) and cost_pct:
            period_output.iloc[-1] -= turnover * cost_pct / 100
        output.extend(period_output.tolist())
        dates.extend(period_output.index.tolist())
        allocation = new_allocation
    return weights, pd.Series(output, index=dates, dtype=float)


def annual_growth(series: pd.Series) -> float:
    if len(series) < 2:
        return np.nan
    curve = (1 + series).cumprod()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    return curve.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan


def detail_chart(item: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    returns, ticker = item["returns"], item["ticker"]
    ax.plot((1 + returns[BASE]).cumprod(), label="Nifty 50 only")
    ax.plot((1 + returns[ticker]).cumprod(), label=f"{item['name']} only")
    ax.plot(item["curve"], label=f"Nifty + {item['name']} dual portfolio")
    ax.set_title(f"Nifty 50 vs {item['name']} vs dual portfolio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


st.title("Nifty 50 + Alternative Assets Rebalancer")
st.caption("Each checked asset creates a separate Nifty–alternative dual portfolio.")
today = pd.Timestamp.today()
start_date = f"{today.year - 11}-01-01"
end_date = (today - Week(weekday=6)).strftime("%Y-%m-%d")

with st.sidebar:
    st.header("Assets")
    selected = st.multiselect(
        "Run alternatives",
        list(PRESETS),
        default=list(PRESETS),
        help=(
            "All non-company assets from the pasted top-100 list are available "
            "and selected by default. Uncheck any you do not want to compare."
        ),
    )
    custom_text = st.text_input("Extra Yahoo tickers", placeholder="ETH-USD, QQQ")
    st.header("Model")
    limit_pct = st.number_input(
        "Maximum monthly Nifty-to-alternative shift (%)",
        min_value=0.0,
        value=0.05,
        step=0.05,
        format="%.2f",
    )
    window = st.number_input("Covariance window (months)", 0, value=36, step=1)
    cost = st.number_input("Transaction cost (%)", 0.0, value=0.10, step=0.01)
    train_pct = st.slider("Training history (%)", 50, 90, 70, 5)
    st.header("Current values (optional)")
    current_nifty = st.number_input("Nifty value (₹)", 0.0, value=0.0, step=1000.0)
    current_values = {
        name: st.number_input(
            f"{name} value (₹)", 0.0, value=0.0, step=1000.0, key=f"value_{name}"
        )
        for name in selected
    }
    st.caption("Results refresh automatically when a selection or setting changes.")

custom = [(ticker, ticker) for ticker in {x.strip().upper() for x in custom_text.split(",")} if ticker and ticker != BASE]
pairs = [(name, PRESETS[name]) for name in selected] + custom
if not pairs:
    st.info("Select at least one alternative asset to display results.")
    st.stop()

results, failures = [], []
bar = st.progress(0, text="Starting...")
unique_tickers = tuple(dict.fromkeys(ticker for _, ticker in pairs))
try:
    price_batch = download_price_batch(unique_tickers, start_date, end_date)
except Exception as exc:
    st.error(f"The shared Yahoo Finance download failed: {exc}")
    st.stop()

for number, (name, ticker) in enumerate(pairs, 1):
    try:
        bar.progress((number - 1) / len(pairs), text=f"Running {name}...")
        prices = pair_prices(price_batch, ticker)
        returns = prices.pct_change().dropna()
        month_ends = returns.resample("ME").last().index
        if len(month_ends) < 3:
            raise ValueError("insufficient overlapping monthly history")
        weights, portfolio = simulate(returns, month_ends, limit_pct, int(window), cost)
        split = month_ends[max(1, min(len(month_ends) - 2, int(len(month_ends) * train_pct / 100)))]

        searches, candidates = [], {}
        for candidate in np.arange(0.05, 5.1, 0.05):
            _, series = simulate(returns, month_ends, float(candidate), int(window), cost)
            candidates[float(candidate)] = series
            searches.append((float(candidate), annual_growth(series.loc[series.index <= split])))
        search = pd.DataFrame(searches, columns=["Limit (%)", "Training CAGR"])
        best = search.loc[search["Training CAGR"].idxmax()]
        chosen = float(best["Limit (%)"])
        test_cagr = annual_growth(candidates[chosen].loc[candidates[chosen].index > split])

        nifty_weight, alt_weight = map(float, weights.iloc[-1])
        ratio = alt_weight / nifty_weight if nifty_weight else np.nan
        current_alt = current_values.get(name, 0.0)
        total = current_nifty + current_alt
        target_alt = total * alt_weight
        results.append(
            dict(
                name=name, ticker=ticker, returns=returns, weights=weights,
                curve=(1 + portfolio).cumprod(), nifty=nifty_weight, alt=alt_weight,
                ratio=ratio, chosen=chosen, train=float(best["Training CAGR"]),
                test=test_cagr, split=split, search=search, total=total,
                target_nifty=total * nifty_weight, target_alt=target_alt,
                trade=target_alt - current_alt,
            )
        )
    except Exception as exc:
        failures.append(f"{name} ({ticker}): {exc}")
bar.empty()
for failure in failures:
    st.warning(failure)
if not results:
    st.error("No selected scenario could be calculated.")
    st.stop()

st.subheader("Output rows")
summary = pd.DataFrame([
    {
        "Alternative": x["name"], "Ticker": x["ticker"],
        "Nifty target": x["nifty"], "Alternative target": x["alt"],
        "Alternative per ₹1 Nifty": x["ratio"],
        "Selected limit": x["chosen"] / 100,
        "Training CAGR": x["train"], "Test CAGR": x["test"],
    } for x in results
])
percent_columns = ["Nifty target", "Alternative target", "Selected limit", "Training CAGR", "Test CAGR"]
formats = {column: "{:.2%}" for column in percent_columns}
formats["Alternative per ₹1 Nifty"] = "₹{:.3f}"
st.dataframe(summary.style.format(formats), use_container_width=True, hide_index=True)

st.subheader("Overlapped base vs dual portfolios")
fig, ax = plt.subplots(figsize=(12, 5))
base_returns = results[0]["returns"][BASE]
ax.plot((1 + base_returns).cumprod(), label="Nifty 50 only", color="black", linewidth=2.5)
for item in results:
    ax.plot(item["curve"], label=f"Nifty + {item['name']}")
ax.set_title("Nifty 50 base versus selected dual portfolios")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)

tabs = st.tabs([x["name"] for x in results])
for tab, item in zip(tabs, results):
    with tab:
        st.pyplot(detail_chart(item), use_container_width=True)
        st.info(
            f"For every ₹1 in Nifty, hold ₹{item['ratio']:.3f} in {item['name']}. "
            f"Targets: {item['nifty']:.2%} Nifty and {item['alt']:.2%} {item['name']}."
        )
        if item["total"] > 0:
            c1, c2 = st.columns(2)
            c1.metric("Target Nifty", f"₹{item['target_nifty']:,.2f}")
            c2.metric(f"Target {item['name']}", f"₹{item['target_alt']:,.2f}")
            direction = f"Nifty to {item['name']}" if item["trade"] >= 0 else f"{item['name']} to Nifty"
            st.success(f"Move ₹{abs(item['trade']):,.2f} from {direction}.")
        with st.expander("Weights and limit-search details"):
            st.dataframe(item["weights"].style.format("{:.2%}"), use_container_width=True)
            st.dataframe(item["search"].style.format({"Limit (%)": "{:.2f}", "Training CAGR": "{:.2%}"}), hide_index=True)
