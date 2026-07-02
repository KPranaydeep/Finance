from __future__ import annotations

from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    available_completed_quarters,
    drawdown_series,
    recommend_dry_powder,
    run_dry_powder_backtest,
    score_benchmarks,
    weighted_portfolio_returns,
)
from config import INDEX_CATALOG, POLICY_FLOORS
from market_data import (
    download_benchmark_close,
    download_many,
    parse_benchmark_csv,
    parse_holdings_csv,
)

st.set_page_config(page_title="Dry Powder Planner — India", page_icon="🛡️", layout="wide")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_benchmark(nse_name: str, symbol: str, start: date, end: date, provider: str):
    return download_benchmark_close(nse_name=nse_name, yahoo_symbol=symbol, start=start, end=end, provider=provider)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_many(symbols: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return download_many(symbols=symbols, start=start, end=end)


def money(value: float) -> str:
    return f"₹{value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_threshold_text(text: str) -> list[float]:
    values = []
    for token in text.split(","):
        token = token.strip().replace("%", "")
        if not token:
            continue
        value = float(token) / 100.0
        if not 0 < value < 1:
            raise ValueError("Each deployment trigger must be between 0% and 100%.")
        values.append(value)
    if not values:
        raise ValueError("Enter at least one deployment trigger.")
    return sorted(set(values))


def quarter_label(item: tuple[pd.Timestamp, pd.Timestamp]) -> str:
    start, end = item
    return f"{start.year} Q{start.quarter} — {start:%d %b} to {end:%d %b}"


def render_matcher(start: date, end: date) -> str | None:
    st.subheader("Find the benchmark that actually resembles your holdings")
    uploaded = st.file_uploader(
        "Upload holdings CSV",
        type=["csv"],
        help="Required: ticker/symbol plus value/amount or weight/allocation. NSE tickers may be entered without .NS.",
    )
    if uploaded is None:
        st.caption("Use the included sample_holdings.csv as a template. Until a file is uploaded, select the index manually below.")
        return None

    try:
        holdings = parse_holdings_csv(uploaded)
        st.dataframe(
            holdings.assign(weight=lambda x: (100 * x["weight"]).round(2)).rename(columns={"weight": "Weight %"}),
            use_container_width=True,
            hide_index=True,
        )
        candidate_names = st.multiselect(
            "Candidate benchmarks to test",
            options=list(INDEX_CATALOG),
            default=["NIFTY 50", "NIFTY 500", "NIFTY Midcap 100", "NIFTY Bank", "NIFTY IT"],
        )
        if not candidate_names:
            st.warning("Choose at least one candidate benchmark.")
            return None

        symbols = tuple(holdings["ticker"].tolist() + [INDEX_CATALOG[n]["symbol"] for n in candidate_names])
        with st.spinner("Matching portfolio behaviour against candidate benchmarks…"):
            prices = cached_many(symbols, start, end)
            portfolio_returns, missing = weighted_portfolio_returns(prices, holdings)
            candidate_symbols = [INDEX_CATALOG[n]["symbol"] for n in candidate_names]
            scores = score_benchmarks(portfolio_returns, prices[[s for s in candidate_symbols if s in prices.columns]])

        if missing:
            st.warning(f"Skipped {len(missing)} ticker(s) with insufficient data: {', '.join(missing[:8])}")
        if scores.empty:
            st.error("No candidate had enough overlapping data. Select different candidates or use manual selection.")
            return None

        reverse_names = {v["symbol"]: k for k, v in INDEX_CATALOG.items()}
        scores.insert(0, "Benchmark", scores["symbol"].map(reverse_names).fillna(scores["symbol"]))
        display = scores[["Benchmark", "match_score", "correlation", "beta", "tracking_error", "overlap_days"]].copy()
        display.columns = ["Benchmark", "Match score", "Correlation", "Beta", "Tracking error", "Days"]
        display["Match score"] = display["Match score"].round(1)
        display["Correlation"] = display["Correlation"].round(3)
        display["Beta"] = display["Beta"].round(2)
        display["Tracking error"] = (100 * display["Tracking error"]).round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True, hide_index=True)
        best_symbol = str(scores.iloc[0]["symbol"])
        best_name = reverse_names.get(best_symbol, best_symbol)
        st.success(f"Best statistical match in the tested set: **{best_name}**. You can still override it.")
        return best_name if best_name in INDEX_CATALOG else None
    except Exception as exc:
        st.error(f"Could not analyse holdings: {exc}")
        return None


st.title("🛡️ Dry Powder Planner for Indian Equity Portfolios")
st.write(
    "Choose or statistically match an NSE benchmark, estimate a transparent minimum cash reserve, "
    "and test how that reserve would have behaved during the last completed quarter."
)
st.info(
    "Dry powder is investable liquidity—not your emergency fund. This app does not model taxes, brokerage, "
    "slippage, debt-fund taxation, or option hedges. Market data may be delayed or revised."
)

with st.sidebar:
    st.header("Portfolio assumptions")
    portfolio_value = st.number_input("Current portfolio value (₹)", 100_000, 1_000_000_000, 2_000_000, 50_000)
    lookback_years = st.slider("Risk lookback", 3, 15, 5)
    tolerance = st.slider("Maximum drawdown you can tolerate", 5, 50, 25, 1) / 100
    policy_name = st.selectbox("Reserve policy", list(POLICY_FLOORS), index=1)
    stress_method = st.radio("Stress basis", ["Historical maximum drawdown", "Custom crash assumption"])
    custom_stress = None
    if stress_method == "Custom crash assumption":
        custom_stress = st.slider("Assumed benchmark crash", 10, 70, 35, 1) / 100
    cash_yield = st.number_input("Annual yield on dry powder (%)", 0.0, 15.0, 6.5, 0.25) / 100
    trigger_text = st.text_input("Deploy at drawdowns (%)", "5, 10, 15, 20")
    custom_symbol = st.text_input("Optional custom Yahoo Finance symbol", placeholder="Example: ^NSEI or an ETF ticker")

end_download = date.today() + timedelta(days=1)
start_download = date.today() - timedelta(days=365 * lookback_years + 120)

mode = st.radio(
    "Benchmark selection mode",
    ["Choose manually", "Match using holdings CSV"],
    horizontal=True,
)
matched_name = None
if mode == "Match using holdings CSV":
    matched_name = render_matcher(start_download, end_download)

index_names = list(INDEX_CATALOG)
default_index = matched_name or "NIFTY 50"
selected_name = st.selectbox(
    "Benchmark for risk and backtest",
    index_names,
    index=index_names.index(default_index),
)
selected = INDEX_CATALOG[selected_name]
selected_symbol = custom_symbol.strip() or selected["symbol"]
st.caption(f"**{selected_name}** · {selected['kind']} · {selected['description']} · Data symbol: `{selected_symbol}`")

source_mode = st.radio(
    "Price-data source",
    ["Automatic — NSE official, then Yahoo fallback", "NSE official", "Yahoo/yfinance", "Upload benchmark CSV"],
    horizontal=True,
)
benchmark_csv = None
if source_mode == "Upload benchmark CSV":
    benchmark_csv = st.file_uploader("Upload Date/Close benchmark CSV", type=["csv"], key="benchmark_csv")

try:
    thresholds = parse_threshold_text(trigger_text)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    with st.spinner("Loading benchmark history…"):
        if source_mode == "Upload benchmark CSV":
            if benchmark_csv is None:
                st.warning("Upload a benchmark CSV to continue.")
                st.stop()
            benchmark = parse_benchmark_csv(benchmark_csv)
            data_source_label = "Uploaded benchmark CSV"
        else:
            provider = {
                "Automatic — NSE official, then Yahoo fallback": "Automatic",
                "NSE official": "NSE official",
                "Yahoo/yfinance": "Yahoo/yfinance",
            }[source_mode]
            if custom_symbol.strip() and provider == "Automatic":
                provider = "Yahoo/yfinance"
            benchmark, data_source_label = cached_benchmark(
                selected["nse_name"], selected_symbol, start_download, end_download, provider
            )
except Exception as exc:
    st.error(f"Benchmark data could not be loaded: {exc}")
    st.info("Use the CSV option with Date and Close columns when a cloud host blocks NSE or Yahoo requests.")
    st.stop()

st.caption(f"Loaded {len(benchmark):,} observations from **{data_source_label}**; latest date: **{benchmark.index.max():%d %b %Y}**.")

risk_window_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)
risk_prices = benchmark.loc[benchmark.index >= risk_window_start]
if len(risk_prices) < 100:
    st.error("The selected benchmark has too little history for the chosen lookback. Reduce the lookback or choose another benchmark.")
    st.stop()

recommendation = recommend_dry_powder(
    prices=risk_prices,
    tolerance_drawdown=tolerance,
    policy_floor=POLICY_FLOORS[policy_name],
    custom_stress_drawdown=custom_stress,
)
recommended_amount = portfolio_value * recommendation.recommended_weight

st.header("1. Dry-powder requirement")
cols = st.columns(5)
cols[0].metric("Recommended reserve", pct(recommendation.recommended_weight), money(recommended_amount))
cols[1].metric("Historical max drawdown", pct(recommendation.historical_max_drawdown))
cols[2].metric("Stress drawdown used", pct(recommendation.stress_drawdown))
cols[3].metric("Your tolerance", pct(recommendation.tolerance_drawdown))
cols[4].metric("Estimated stressed portfolio DD", pct(recommendation.estimated_stress_drawdown))

st.write(
    f"The formula-only minimum is **{pct(recommendation.formula_minimum_weight)}**. "
    f"After applying your **{policy_name.split('—')[0].strip().lower()}** reserve floor and rounding upward, "
    f"the app suggests maintaining **{money(recommended_amount)}** as investable dry powder."
)
with st.expander("How the recommendation is calculated"):
    st.latex(r"\text{Minimum cash weight} = \max\left(0, 1 - \frac{\text{tolerable drawdown}}{\text{stress drawdown}}\right)")
    st.write(
        "This assumes the selected benchmark falls by the stress amount while cash remains broadly stable. "
        "It is an allocation estimate, not a guarantee. A concentrated portfolio may fall more than its benchmark."
    )

chart_df = pd.DataFrame({"Benchmark": risk_prices / risk_prices.iloc[0] * 100, "Drawdown": drawdown_series(risk_prices) * 100})
fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Benchmark"], name="Growth of 100"))
fig_risk.update_layout(title=f"{selected_name}: normalized history used for risk estimate", yaxis_title="Index level (start = 100)")
st.plotly_chart(fig_risk, use_container_width=True)

st.header("2. Previous-quarter dry-powder test")
quarters = available_completed_quarters(benchmark, count=12)
if not quarters:
    st.error("No completed quarter is available in the loaded data.")
    st.stop()
selected_quarter = st.selectbox("Quarter", quarters, format_func=quarter_label)

result, summary, events = run_dry_powder_backtest(
    prices=benchmark,
    start_date=selected_quarter[0],
    end_date=selected_quarter[1],
    initial_capital=portfolio_value,
    dry_powder_weight=recommendation.recommended_weight,
    thresholds=thresholds,
    annual_cash_yield=cash_yield,
)

m = st.columns(5)
m[0].metric("Fully invested end value", money(summary.fully_invested_end), pct(summary.fully_invested_return))
m[1].metric("Dry-powder strategy end", money(summary.strategy_end), pct(summary.strategy_return))
m[2].metric("Strategy vs fully invested", money(summary.opportunity_cost_or_gain))
m[3].metric("Max-DD reduction", f"{(summary.fully_invested_max_drawdown - summary.strategy_max_drawdown) * 100:.1f} pp")
m[4].metric("Cash deployed", money(summary.cash_deployed), f"{summary.triggers_hit} trigger(s)")

normalized = result[["Fully invested", "Dry powder deployed", "Dry powder kept idle"]] / portfolio_value * 100
fig = go.Figure()
for column in normalized.columns:
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized[column], name=column))
fig.update_layout(title=f"Portfolio paths during {quarter_label(selected_quarter)}", yaxis_title="Portfolio value (start = 100)")
st.plotly_chart(fig, use_container_width=True)

if summary.triggers_hit == 0:
    st.warning(
        f"No deployment trigger was reached. The reserve reduced the strategy's maximum drawdown from "
        f"{pct(summary.fully_invested_max_drawdown)} to {pct(summary.strategy_max_drawdown)}, but keeping capital out "
        f"of equities changed the ending value by {money(summary.opportunity_cost_or_gain)} versus being fully invested."
    )
else:
    direction = "more" if summary.opportunity_cost_or_gain >= 0 else "less"
    st.success(
        f"The strategy deployed {money(summary.cash_deployed)} across {summary.triggers_hit} trigger(s). "
        f"Deployment added {money(summary.deployment_value_add)} versus leaving the reserve idle. "
        f"It ended with {money(abs(summary.opportunity_cost_or_gain))} {direction} than the fully invested portfolio, "
        f"while maximum drawdown changed from {pct(summary.fully_invested_max_drawdown)} to {pct(summary.strategy_max_drawdown)}."
    )

st.subheader("Deployment events")
if events.empty:
    st.caption("No event occurred in the selected quarter.")
else:
    event_display = events.copy()
    event_display["Trigger"] = (event_display["Trigger"] * 100).round(1).astype(str) + "%"
    event_display["Invested"] = event_display["Invested"].map(money)
    event_display["Index price"] = event_display["Index price"].round(2)
    event_display["Units bought"] = event_display["Units bought"].round(4)
    st.dataframe(event_display, use_container_width=True, hide_index=True)

export = result.reset_index().copy()
export.insert(1, "Benchmark", selected_name)
export.insert(2, "Symbol", selected_symbol)
st.download_button(
    "Download quarter simulation CSV",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=f"dry_powder_{selected_quarter[0].year}_Q{selected_quarter[0].quarter}.csv",
    mime="text/csv",
)

st.header("3. Interpretation")
st.write(
    "A reserve is useful when it either prevents forced selling, reduces a drawdown to a level you can actually tolerate, "
    "or gives you capital to buy after preset declines. It is not automatically return-enhancing: in steadily rising quarters, "
    "cash usually creates an opportunity cost. Judge it across both difficult and strong quarters—not from one quarter alone."
)
st.caption(
    "Educational tool only. Before acting, verify the selected benchmark, your portfolio's actual beta and concentration, "
    "the safety/liquidity of the reserve instrument, taxes, exit loads and transaction costs."
)
