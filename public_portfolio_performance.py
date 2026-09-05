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


def build_execution_plan_prompt(current: dict, constituents: list[dict]) -> tuple[str, dict]:
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
            {"ticker": row["ticker"], "target_weight_pct": round(float(row["target_weight"]) * 100)}
            for row in constituents
        ],
        "cash_weight_pct": round(float(current["cash_weight"]) * 100),
    }
    target_json = json.dumps(public_target, sort_keys=True, indent=2, default=str)
    threshold_pct = REBALANCE_MIN_EXPECTED_IMPROVEMENT * 100
    prompt = f"""You are preparing a private, user-reviewed portfolio execution worksheet.

I have attached or pasted a holdings report downloaded from my broker. Read that report and compare it with the immutable public target snapshot below. Do not rerun, modify, or improve the public optimizer.

PUBLIC TARGET SNAPSHOT
{target_json}

MANDATORY PROCESS
1. Read CSV, XLSX, PDF, or pasted holdings. Identify the original security name, ticker/ISIN, exchange, quantity, average cost, and available cash when present.
2. Never expose, repeat, or retain my name, PAN, demat/account number, email, phone, address, or broker credentials. Tell me to redact them if detected.
3. Normalize securities to the target tickers. Before calculating trades, show every ambiguous or unresolved mapping. Never guess a mapping, price, quantity, currency, or holding.
4. Use current, timestamped market prices only when you can access a reliable source. Otherwise ask me for prices and stop before giving quantities.
5. Calculate current value and weight for every holding, including positions outside the public target. Confirm that parsed quantities and total value look plausible.
6. Produce two clearly separated results:
   A. Drift-only comparison, which never implies that a trade should be executed.
   B. Conditional execution worksheet using the gate below.
7. The rebalance gate is a net expected annualized-return improvement of at least {threshold_pct:.0f} percentage points. Net improvement means proposed expected annual return minus current expected annual return minus annualized brokerage, taxes, spreads, slippage, and other estimated implementation costs.
8. Do not invent expected returns. Use the gate only when both current and proposed expected returns come from an explicit, comparable, documented methodology with an uncertainty range. If this cannot be established, set every actionable call to HOLD and state: "6 percentage-point improvement not established."
9. Even when the gate passes, reduce churn: use a 1 percentage-point target-weight tolerance band, ignore trades below the greater of INR 1,000 or 0.5% of portfolio value, prefer fewer netting trades, use whole shares, do not exceed available cash, and flag tax or illiquidity concerns. Do not sell and rebuy an economically equivalent position.
10. Never place orders or claim guaranteed returns. This is a planning worksheet that I must verify with my broker and a qualified adviser.

REQUIRED OUTPUT
- First: parsing summary, detected broker/format, data date, total portfolio value, available cash, unresolved rows, stale/missing prices, and assumptions.
- Second: a mapping table with Broker Security, Resolved Ticker, ISIN, Exchange, Quantity, Price, Mapping Confidence, and Mapping Reason.
- Third: an execution table with Ticker, Current Quantity, Current Weight %, Target Weight %, Difference %, Call (BUY/SELL/HOLD), Whole-share Quantity, Estimated Trade Value, Estimated Costs and Slippage, Net Expected Annualized Improvement, Confidence, and Reason.
- Fourth: totals for buys, sells, costs, turnover, residual cash, and post-trade weight sum.
- Finish with a concise manual execution sequence and a verification checklist. Mark unresolved rows as REVIEW, never as BUY or SELL.
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
if float(current["cash_weight"])>0:
    allocation=pd.concat([allocation,pd.DataFrame([{"ticker":"CASH","target_weight":current["cash_weight"]}])],ignore_index=True)
allocation["Allocation"]=allocation["target_weight"].map(lambda x:f"{float(x):.2%}")
st.dataframe(allocation[["ticker","Allocation"]],use_container_width=True,hide_index=True)
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
execution_prompt, public_target = build_execution_plan_prompt(current, record["constituents"])
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
    "The prompt defaults to HOLD unless a comparable, documented net expected annualized-return "
    "improvement of at least 6 percentage points is established. It also applies tolerance, minimum-trade, "
    "whole-share, cost, slippage, and cash checks to reduce churn."
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
