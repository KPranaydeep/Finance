"""Production public portfolio: observed evidence and accountable outlook."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from public_basket_postgres import DEFAULT_BASKET_ID, connect_public_basket_db, get_public_basket_database_url
from public_portfolio_publications import load_trust_records, verify_trust_audit
from public_portfolio_trust import CALCULATION_VERSION, forecast_calibration, performance_metrics, select_horizon, xirr
from public_release_checks import inspect_public_data

IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger(__name__)
HORIZONS = {"14D":14,"30D":30,"3M":91,"6M":183,"1Y":365,"3Y":1096,"5Y":1826,"MAX":None}
REBALANCE_MIN_EXPECTED_IMPROVEMENT = 0.06

st.set_page_config(page_title="Public Portfolio", page_icon="📈", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_public_record(basket_id: str) -> dict[str, Any]:
    url=get_public_basket_database_url()
    if not url: raise RuntimeError("Public record is not configured")
    with connect_public_basket_db(url) as conn:
        basket=conn.execute("SELECT * FROM public_baskets WHERE basket_id=%s",(basket_id,)).fetchone()
        if not basket: return {"basket":None}
        nav=conn.execute("""SELECT DISTINCT ON (nav_date) * FROM daily_nav WHERE basket_id=%s
                            ORDER BY nav_date,calculation_version DESC""",(basket_id,)).fetchall()
        rebalances=conn.execute("""SELECT rebalance_id,effective_at,status,rationale,payload_sha256
            FROM rebalance_events WHERE basket_id=%s ORDER BY created_at DESC""",(basket_id,)).fetchall()
        executions=conn.execute("""SELECT x.executed_at,o.symbol,o.side,x.quantity,x.execution_price,x.fees_inr,x.taxes_inr
            FROM trade_executions x JOIN trade_orders o ON o.order_id=x.order_id
            JOIN rebalance_events r ON r.rebalance_id=o.rebalance_id WHERE r.basket_id=%s ORDER BY x.executed_at DESC""",(basket_id,)).fetchall()
        trust=load_trust_records(conn,basket_id)
    return {"basket":dict(basket),"nav":[dict(r) for r in nav],"rebalances":[dict(r) for r in rebalances],
            "executions":[dict(r) for r in executions],**trust}


def pct(value: float | None) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.2%}"


def investor_cash_flows(rows: list[dict], terminal_date, terminal_value: float) -> list[tuple]:
    flows=[]
    for row in rows:
        kind=str(row["event_type"]).upper()
        amount=abs(float(row["amount_inr"]))
        if kind in {"CONTRIBUTION","DEPOSIT","CAPITAL_CONTRIBUTION"}: flows.append((row["event_at"],-amount))
        elif kind in {"WITHDRAWAL","CAPITAL_WITHDRAWAL"}: flows.append((row["event_at"],amount))
    if flows and terminal_value > 0: flows.append((terminal_date,terminal_value))
    return flows


@st.cache_data(ttl=900, show_spinner=False)
def load_latest_prices(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Return the latest available unadjusted closes without blocking the page on failures."""
    if not tickers:
        return {}
    try:
        data = yf.download(
            list(tickers), period="7d", interval="1d", auto_adjust=False,
            progress=False, threads=True, group_by="column",
        )
        if data.empty:
            return {}
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data.get("Close")
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        result = {}
        for ticker in tickers:
            if close is None or ticker not in close.columns:
                continue
            series = pd.to_numeric(close[ticker], errors="coerce").dropna()
            if not series.empty:
                result[ticker] = {
                    "price": float(series.iloc[-1]),
                    "price_as_of": pd.Timestamp(series.index[-1]).date().isoformat(),
                }
        return result
    except Exception:
        LOGGER.exception("Latest public allocation prices could not be loaded")
        return {}


def build_execution_plan_prompt(current: dict, constituents: list[dict], prices: dict[str, dict]) -> tuple[str, dict]:
    """Build a version-bound prompt containing public data only."""
    public_target = {
        "basket_id": current["basket_id"],
        "publication_id": current["publication_id"],
        "portfolio_version": f"P{int(current['portfolio_version']):03d}",
        "portfolio_fingerprint": current["portfolio_fingerprint"],
        "strategy_version": current["strategy_version"],
        "calculation_version": current["calculation_version"],
        "as_of": str(current["as_of"]),
        "published_at": str(current["published_at"]),
        "target_positions": [
            {
                "ticker": row["ticker"],
                "target_weight_pct": round(float(row["target_weight"]) * 100),
                "planning_price": prices.get(row["ticker"], {}).get("price"),
                "price_as_of": prices.get(row["ticker"], {}).get("price_as_of"),
            }
            for row in constituents
        ],
        "cash_weight_pct": round(float(current["cash_weight"]) * 100),
        "price_source": "Yahoo Finance latest available unadjusted close; reference data, not part of the immutable portfolio fingerprint",
    }
    target_json = json.dumps(public_target, sort_keys=True, indent=2, default=str)
    threshold_pct = REBALANCE_MIN_EXPECTED_IMPROVEMENT * 100
    prompt = f"""Create a short, actionable, user-reviewed portfolio execution plan from my attached broker holdings report and the immutable public target below. Do not rerun or modify the public optimizer.

PUBLIC TARGET SNAPSHOT
{target_json}

WORKING RULES
1. Parse the report directly. Use security name and ISIN to resolve exchange tickers from reliable public sources. A holding is not "unresolved" merely because it is absent from the target; a resolved non-target holding has target weight 0%.
2. Do not repeat personal identifiers. Give only one short redaction warning if the report contains them.
3. Treat the broker report as the complete stock portfolio unless it explicitly says otherwise. If cash is absent, assume opening cash is zero and fund buys from sale proceeds. State this assumption once; do not stop.
4. Use the embedded target planning prices when dated within five calendar days. Broker closing prices within the same limit are acceptable for current holdings. Prefer newer reliable prices when tools permit. State the price dates once. Do not block the plan merely because prices were not independently verified.
5. Estimate both portfolios consistently. Use adjusted price history over the longest common period up to three years, requiring at least one year. Calculate each portfolio's annualized geometric return using its weights. Deduct estimated one-time taxes, brokerage, spread, and slippage from the proposed portfolio benefit. Label this a historical return-based estimate, not a guarantee.
6. DECISION = REBALANCE only when proposed net annualized return minus current annualized return is at least {threshold_pct:.0f} percentage points. Otherwise DECISION = HOLD. If market-history tools are unavailable, ask only for permission to fetch prices/history or for a price-history file; do not produce a long refusal table.
7. When REBALANCE applies, calculate practical whole-share trades. Sell non-target holdings and overweight holdings first; use those proceeds for buys. Never require additional cash unless the user explicitly requests investment of new money.
8. Reduce churn: ignore a position within 1 percentage point of target; suppress a trade below the greater of INR 100 or 0.5% of portfolio value; do not sell and rebuy equivalent exposure; do not exceed available proceeds; flag illiquid securities and large tax impact.
9. Allocate rounding residue to the largest underweight affordable target. Show residual cash. Never place orders automatically.

OUTPUT — KEEP IT SHORT
Start with exactly these five lines:
DECISION: REBALANCE, HOLD, or NEEDS DATA
CURRENT ESTIMATED ANNUAL RETURN: x%
TARGET ESTIMATED ANNUAL RETURN: x%
NET ESTIMATED IMPROVEMENT: x percentage points
WHY: one sentence

If DECISION is REBALANCE, show one execution table containing only actual trades:
Sequence | Ticker | BUY/SELL | Whole shares | Planning price | Approx. value | Reason

Then show only:
- Total sales, total purchases, estimated costs/slippage, turnover, and residual cash.
- "Execute sells first, wait for proceeds, then execute buys in sequence. Recheck live prices before every order."
- At most three warnings that could materially change execution.

If DECISION is HOLD, do not print every target row. Show a maximum of five largest allocation differences and the next review trigger.

If any security truly cannot be resolved after using its ISIN and company name, mark only that row REVIEW and continue calculating the resolvable portfolio when reasonable. Never fabricate a ticker, holding, price, return, or quantity.
"""
    return prompt, public_target


st.title("PUBLIC PORTFOLIO")
st.caption("An immutable public record. Historical results are observed; outlooks are statistical scenarios.")
try:
    record=load_public_record(DEFAULT_BASKET_ID)
except Exception:
    LOGGER.exception("Public portfolio load failed")
    st.error("The verified public record is temporarily unavailable.")
    st.stop()
if not record.get("basket"):
    st.info("No public portfolio has been published yet.")
    st.stop()

basket,current=record["basket"],record.get("current")
if not current:
    st.info("The basket exists, but no approved portfolio version has been published.")
    st.stop()

st.subheader("Portfolio")
c1,c2,c3=st.columns(3)
c1.metric("Portfolio version",f"P{int(current['portfolio_version']):03d}")
c2.metric("Constituents",len(record["constituents"]))
c3.metric("As of",current["as_of"].astimezone(IST).strftime("%d %b %Y"))
allocation=pd.DataFrame(record["constituents"])
price_snapshot=load_latest_prices(tuple(allocation["ticker"].astype(str)))
if float(current["cash_weight"])>0:
    allocation=pd.concat([allocation,pd.DataFrame([{"ticker":"CASH","target_weight":current["cash_weight"]}])],ignore_index=True)
allocation["Allocation"]=allocation["target_weight"].map(lambda x:f"{float(x):.2%}")
allocation["Price"]=allocation["ticker"].map(
    lambda ticker: f"{price_snapshot[ticker]['price']:,.2f}" if ticker in price_snapshot else "N/A"
)
st.dataframe(allocation[["ticker","Allocation","Price"]],use_container_width=True,hide_index=True)
price_dates=sorted({item["price_as_of"] for item in price_snapshot.values()})
if price_dates:
    st.caption(f"Prices: latest available unadjusted close from Yahoo Finance · through {price_dates[-1]}")
st.caption(f"Strategy {current['strategy_version']} · Published {current['published_at'].astimezone(IST):%d %b %Y %H:%M IST}")

st.subheader("Build your private execution plan")
st.write(
    "Download your latest holdings report from your broker, then give it and the prompt below "
    "to the AI assistant of your choice. Your holdings are never uploaded to this website."
)
st.warning(
    "Before sharing a broker report, remove your name, PAN, demat/account number, email, phone, "
    "address, and any credentials. Review the AI provider's privacy policy."
)
execution_prompt, public_target = build_execution_plan_prompt(current, record["constituents"], price_snapshot)
version_label = f"p{int(current['portfolio_version']):03d}"
download_1, download_2 = st.columns(2)
download_1.download_button(
    "Download execution-plan prompt",
    data=execution_prompt.encode("utf-8"),
    file_name=f"{DEFAULT_BASKET_ID.lower()}-{version_label}-execution-prompt.txt",
    mime="text/plain",
    use_container_width=True,
)
download_2.download_button(
    "Download public target JSON",
    data=json.dumps(public_target, sort_keys=True, indent=2, default=str).encode("utf-8"),
    file_name=f"{DEFAULT_BASKET_ID.lower()}-{version_label}-target.json",
    mime="application/json",
    use_container_width=True,
)
with st.expander("Copy execution-plan prompt"):
    st.caption("Use the copy icon in the top-right of the prompt, then paste it beside your broker report.")
    st.code(execution_prompt, language=None)
st.caption(
    "The prompt gives a decision first and authorizes trades only when a consistent three-year historical "
    "return estimate shows at least 6 percentage points of net annualized improvement. It uses sale proceeds, "
    "whole shares, tolerance bands, minimum trade values, costs and slippage to reduce churn."
)

st.subheader("Performance — historical, observed")
nav=record["nav"]
all_metrics=performance_metrics(nav)
terminal=float(nav[-1]["total_value"]) if nav else 0.0
flows=investor_cash_flows(record["cash_flows"],nav[-1]["nav_date"] if nav else datetime.now(IST),terminal)
xirr_value=xirr(flows) if flows else None
m1,m2,m3,m4=st.columns(4)
m1.metric("Since inception return",pct(all_metrics.get("total_return")))
m2.metric("Historical XIRR",pct(xirr_value))
m3.metric("Maximum drawdown",pct(all_metrics.get("maximum_drawdown")))
m4.metric("Annualized volatility",pct(all_metrics.get("annualized_volatility")))
if xirr_value is None:
    st.caption("Historical XIRR: N/A — insufficient/invalid external cash-flow history. Portfolio index return is shown separately.")
else:
    st.caption(f"Historical XIRR uses actual investor-perspective cash flows from {flows[0][0]:%d %b %Y} to {flows[-1][0]:%d %b %Y}.")

available={label:days for label,days in HORIZONS.items() if select_horizon(nav,days)}
if available:
    selected=st.segmented_control("Period",list(available),default=list(available)[-1])
    metrics=performance_metrics(select_horizon(nav,available[selected]))
    a,b,c,d=st.columns(4)
    a.metric("Total return",pct(metrics.get("total_return")))
    b.metric("Annualized return",pct(metrics.get("annualized_return")))
    c.metric("Current drawdown",pct(metrics.get("current_drawdown")))
    d.metric("Positive days",pct(metrics.get("positive_day_percentage")))
    detail=pd.DataFrame([{k:v for k,v in metrics.items() if k not in {"start_date","end_date"}}])
    with st.expander("Detailed period statistics"): st.json(metrics)
if nav:
    chart=pd.DataFrame(nav); chart["nav_date"]=pd.to_datetime(chart["nav_date"])
    st.line_chart(chart.set_index("nav_date")[["nav"]],y_label="Portfolio index")

st.subheader("14-Day Outlook — statistical estimate")
forecasts=record.get("active_forecasts",record["forecasts"])
if not forecasts:
    st.info("No accountable 14-day forecast has been recorded yet.")
else:
    forecast=forecasts[0]; values=forecast["forecast_json"]
    o1,o2,o3,o4=st.columns(4)
    o1.metric("Median outcome",pct(values.get("median_return")))
    o2.metric("50% range",f"{pct(values.get('lower_50'))} to {pct(values.get('upper_50'))}")
    o3.metric("90% range",f"{pct(values.get('lower_90'))} to {pct(values.get('upper_90'))}")
    o4.metric("Probability of gain",pct(values.get("probability_positive")))
    st.write(f"Probability of loss: **{pct(values.get('probability_negative'))}** · Probability of loss greater than 5%: **{pct(values.get('probability_loss_gt_threshold'))}**")
    st.warning("Statistical scenario — not a guaranteed prediction.")
    st.caption(f"14-day {values.get('method')} of observed daily returns · sample {values.get('sample_start')} to {values.get('sample_end')} · {values.get('observation_count')} observations · {forecast['calculation_version']}")

calibration=forecast_calibration(forecasts,record.get("active_forecast_realizations",record["forecast_realizations"]))
if calibration["sufficient"]:
    st.caption(
        f"Forecast validation ({calibration['sample_size']} completed): 50% coverage {calibration['coverage_50']:.1%}; "
        f"90% coverage {calibration['coverage_90']:.1%}; directional accuracy {calibration['directional_accuracy']:.1%}; "
        f"mean error {calibration['mean_forecast_error']:.2%}; median error {calibration['median_forecast_error']:.2%}."
    )

st.subheader("History")
tabs=st.tabs(["Portfolio versions","Rebalances","Actual executions","NAV history"])
with tabs[0]: st.dataframe(pd.DataFrame(record["publications"]),use_container_width=True,hide_index=True)
with tabs[1]: st.dataframe(pd.DataFrame(record["rebalances"]),use_container_width=True,hide_index=True)
with tabs[2]: st.dataframe(pd.DataFrame(record["executions"]),use_container_width=True,hide_index=True)
with tabs[3]: st.dataframe(pd.DataFrame(nav),use_container_width=True,hide_index=True)

st.subheader("Verification")
audit_ok,audit_message=verify_trust_audit(record["audit"],DEFAULT_BASKET_ID)
(st.success if audit_ok else st.warning)(audit_message)
st.write("Portfolio versions and forecasts are append-only, fingerprinted, and verified within this basket.")
evidence_state={**record,"performance_metrics":all_metrics,"historical_xirr":xirr_value,
                "xirr_cash_flows":[(str(day),amount) for day,amount in flows],"forecast_calibration":calibration,
                "methodology":{"performance":CALCULATION_VERSION,"forecast":"bootstrap historical daily portfolio returns"}}
security_findings=inspect_public_data(evidence_state,production=True)
if security_findings:
    st.error("Evidence export is unavailable because the public-data inspection did not pass.")
    st.stop()
evidence=json.dumps(evidence_state,sort_keys=True,indent=2,default=str).encode()
st.download_button("Download evidence bundle",evidence,f"{DEFAULT_BASKET_ID.lower()}-evidence.json","application/json",use_container_width=True)
st.caption(f"Calculation version {CALCULATION_VERSION} · Data refreshed every five minutes")
st.info("Historical performance and statistical scenarios are not investment advice and do not guarantee future results.")
