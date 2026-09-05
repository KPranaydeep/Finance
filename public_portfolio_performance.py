"""Production public portfolio: observed evidence and accountable outlook."""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from public_basket_postgres import DEFAULT_BASKET_ID, connect_public_basket_db, get_public_basket_database_url
from public_portfolio_publications import load_trust_records, verify_trust_audit
from public_portfolio_trust import CALCULATION_VERSION, forecast_calibration, performance_metrics, select_horizon, xirr
from public_release_checks import inspect_public_data

IST = ZoneInfo("Asia/Kolkata")
LOGGER = logging.getLogger(__name__)
HORIZONS = {"14D":14,"30D":30,"3M":91,"6M":183,"1Y":365,"3Y":1096,"5Y":1826,"MAX":None}
REBALANCE_MIN_EXPECTED_IMPROVEMENT = 0.06
EXECUTION_SCENARIOS = {
    "Start fresh with cash": (
        "I am starting without existing holdings. Ask me only for the total amount available to invest. "
        "Do not ask about a cash reserve and do not request a broker report. Allocate as much as practical "
        "to the public target using whole shares, leaving only unavoidable rounding residue. For a small amount, "
        "build the closest feasible starter allocation from affordable target securities; partial target coverage "
        "is valid. Do not apply the 6 percentage-point rebalance gate or a minimum-trade-value rule."
    ),
    "Rebalance existing holdings": (
        "I have an existing portfolio. Read my attached broker report, compare it with the public target, "
        "and apply the 6 percentage-point net annualized-improvement gate before suggesting any trades."
    ),
    "Add fresh cash to existing holdings": (
        "I will attach my existing holdings report and provide new cash. Ask only for the new cash amount; "
        "do not ask about a cash reserve. Prefer BUY-only trades that reduce underweights; do not sell existing holdings and "
        "do not apply the 6 percentage-point gate to deployment of new cash."
    ),
    "Raise cash from existing holdings": (
        "I will attach my existing holdings report. Ask only for the cash amount I need. Produce the smallest "
        "practical SELL list, prioritizing non-target and overweight holdings while minimizing churn, taxes and slippage."
    ),
}

st.set_page_config(page_title="Public Portfolio", page_icon="📈", layout="wide")

st.markdown(
    """<style>
    .block-container {max-width: 1220px; padding-top: 2rem; padding-bottom: 5rem;}
    .trust-hero {
        padding: 2.2rem 2.4rem; margin-bottom: 1.5rem; border-radius: 24px;
        background: radial-gradient(circle at 85% 15%, rgba(45,212,191,.22), transparent 32%),
                    linear-gradient(135deg, #12213c 0%, #172554 48%, #0f3b46 100%);
        border: 1px solid rgba(148,163,184,.2); box-shadow: 0 18px 55px rgba(2,6,23,.28);
    }
    .trust-kicker {color:#5eead4; font-size:.75rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase;}
    .trust-title {color:#f8fafc; font-size:clamp(2.1rem,5vw,4rem); line-height:1.02; font-weight:800; margin:.55rem 0 .8rem;}
    .trust-subtitle {color:#cbd5e1; max-width:760px; font-size:1.05rem; line-height:1.65; margin:0;}
    .trust-badge {display:inline-block; margin-top:1.25rem; padding:.42rem .78rem; border-radius:999px;
        color:#ccfbf1; background:rgba(20,184,166,.14); border:1px solid rgba(94,234,212,.32); font-size:.82rem;}
    .metric-card {padding:1.15rem 1.25rem; border-radius:17px; min-height:112px;
        background:linear-gradient(145deg,rgba(30,41,59,.72),rgba(15,23,42,.42));
        border:1px solid rgba(148,163,184,.18); box-shadow:0 8px 24px rgba(2,6,23,.12);}
    .metric-label {color:#94a3b8; font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;}
    .metric-value {color:#f8fafc; font-size:1.75rem; font-weight:750; margin-top:.4rem;}
    .metric-note {color:#94a3b8; font-size:.75rem; margin-top:.28rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
    .metric-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin:.5rem 0 1.35rem;}
    div[data-testid="stDownloadButton"] button {border-radius:12px; min-height:3rem; font-weight:700;}
    div[data-testid="stSelectbox"] > div {border-radius:12px;}
    h2 {padding-top:1.25rem !important; letter-spacing:-.025em;}
    [data-testid="stDataFrame"] {border:1px solid rgba(148,163,184,.16); border-radius:16px; overflow:hidden;}
    .allocation-wrap {overflow-x:auto; border:1px solid rgba(148,163,184,.18); border-radius:16px;
        background:rgba(15,23,42,.22); -webkit-overflow-scrolling:touch;}
    .allocation-table {width:100%; border-collapse:collapse; min-width:540px;}
    .allocation-table th {padding:.8rem 1rem; text-align:left; color:#94a3b8; font-size:.75rem;
        letter-spacing:.06em; text-transform:uppercase; background:rgba(30,41,59,.68);}
    .allocation-table td {padding:.72rem 1rem; border-top:1px solid rgba(148,163,184,.13); color:#e2e8f0;}
    .allocation-table tr:hover td {background:rgba(45,212,191,.045);}
    .ticker-cell {font-weight:700; color:#f8fafc !important; white-space:nowrap;}
    .weight-line {display:flex; align-items:center; gap:.7rem; min-width:180px;}
    .weight-track {width:110px; height:7px; overflow:hidden; border-radius:99px; background:rgba(148,163,184,.18);}
    .weight-fill {height:100%; border-radius:99px; background:linear-gradient(90deg,#14b8a6,#5eead4);}
    .price-cell {font-variant-numeric:tabular-nums; white-space:nowrap;}
    @media (max-width:640px) {
      .block-container {padding-left:.85rem; padding-right:.85rem; padding-top:.75rem;}
      .trust-hero {padding:1.15rem 1rem; margin-bottom:1rem; border-radius:16px;}
      .trust-kicker {font-size:.62rem; letter-spacing:.12em;}
      .trust-title {font-size:1.72rem; line-height:1.08; margin:.4rem 0 .55rem;}
      .trust-subtitle {font-size:.88rem; line-height:1.45;}
      .trust-badge {font-size:.68rem; margin-top:.8rem; padding:.3rem .55rem;}
      .metric-grid {grid-template-columns:repeat(2,minmax(0,1fr)); gap:.55rem; margin:.35rem 0 1rem;}
      .metric-card {padding:.75rem .8rem; border-radius:13px; min-height:86px; box-shadow:none;}
      .metric-label {font-size:.62rem; letter-spacing:.045em;}
      .metric-value {font-size:1.18rem; margin-top:.22rem;}
      .metric-note {font-size:.64rem; margin-top:.18rem;}
      h2 {font-size:1.32rem !important; padding-top:.7rem !important;}
      .allocation-table {min-width:0; table-layout:fixed;}
      .allocation-table th,.allocation-table td {padding:.65rem .62rem; font-size:.82rem;}
      .allocation-table th:nth-child(1),.allocation-table td:nth-child(1) {width:38%;}
      .allocation-table th:nth-child(2),.allocation-table td:nth-child(2) {width:38%;}
      .allocation-table th:nth-child(3),.allocation-table td:nth-child(3) {width:24%; text-align:right;}
      .weight-track {width:58px;}
      .weight-line {min-width:0; gap:.4rem;}
      .ticker-cell {overflow:hidden; text-overflow:ellipsis;}
    }
    </style>""",
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def share_prompt_button(prompt: str, version: str) -> None:
    """Render a native Web Share button with a copy fallback."""
    prompt_json = json.dumps(prompt)
    title_json = json.dumps(f"PUBLIC-01 {version} private execution-plan prompt")
    components.html(
        f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
        <style>
        * {{box-sizing:border-box}} body {{margin:0;background:transparent;font-family:system-ui,-apple-system,sans-serif}}
        button {{width:100%;height:48px;border:1px solid rgba(45,212,191,.55);border-radius:12px;
          background:linear-gradient(135deg,#0f766e,#0d9488);color:white;font-size:15px;font-weight:750;
          cursor:pointer;box-shadow:0 7px 18px rgba(13,148,136,.22)}}
        button:hover {{filter:brightness(1.08);transform:translateY(-1px)}}
        #status {{height:18px;margin-top:5px;color:#94a3b8;text-align:center;font-size:12px}}
        </style></head><body>
        <button id="share" type="button">↗&nbsp;&nbsp;Share execution prompt</button><div id="status"></div>
        <script>
        const promptText={prompt_json}; const shareTitle={title_json};
        const status=document.getElementById('status');
        document.getElementById('share').addEventListener('click', async () => {{
          try {{
            if (navigator.share) {{
              await navigator.share({{title:shareTitle,text:promptText}});
              status.textContent='Shared';
            }} else {{
              await copyFallback(promptText); status.textContent='Share menu unavailable — prompt copied';
            }}
          }} catch (error) {{
            if (error && error.name === 'AbortError') {{status.textContent='Share cancelled'; return;}}
            try {{await copyFallback(promptText); status.textContent='Prompt copied — paste it into your app';}}
            catch (_) {{status.textContent='Use “Copy execution-plan prompt” below';}}
          }}
        }});
        async function copyFallback(text) {{
          if (navigator.clipboard && window.isSecureContext) {{await navigator.clipboard.writeText(text); return;}}
          const box=document.createElement('textarea'); box.value=text; box.style.position='fixed'; box.style.opacity='0';
          document.body.appendChild(box); box.focus(); box.select();
          if (!document.execCommand('copy')) throw new Error('copy failed'); box.remove();
        }}
        </script></body></html>""",
        height=72,
    )


def copy_prompt_button(prompt: str) -> None:
    """Render an explicit clipboard control independent of Streamlit's code toolbar."""
    prompt_json = json.dumps(prompt)
    components.html(
        f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
        <style>
        * {{box-sizing:border-box}} body {{margin:0;background:transparent;font-family:system-ui,-apple-system,sans-serif}}
        button {{width:100%;height:48px;border:1px solid rgba(148,163,184,.35);border-radius:12px;
          background:rgba(30,41,59,.82);color:#f8fafc;font-size:15px;font-weight:750;cursor:pointer}}
        button:hover {{border-color:#5eead4;background:rgba(30,41,59,.98);transform:translateY(-1px)}}
        #status {{height:18px;margin-top:5px;color:#94a3b8;text-align:center;font-size:12px}}
        </style></head><body>
        <button id="copy" type="button">⧉&nbsp;&nbsp;Copy prompt</button><div id="status"></div>
        <script>
        const promptText={prompt_json}; const status=document.getElementById('status');
        document.getElementById('copy').addEventListener('click', async () => {{
          try {{
            if (navigator.clipboard && window.isSecureContext) {{await navigator.clipboard.writeText(promptText);}}
            else {{
              const box=document.createElement('textarea'); box.value=promptText; box.style.position='fixed'; box.style.opacity='0';
              document.body.appendChild(box); box.focus(); box.select();
              if (!document.execCommand('copy')) throw new Error('copy failed'); box.remove();
            }}
            status.textContent='Copied — paste into your AI app';
          }} catch (_) {{status.textContent='Open the prompt below and use its copy icon';}}
        }});
        </script></body></html>""",
        height=72,
    )


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


def build_execution_plan_prompt(current: dict, constituents: list[dict], prices: dict[str, dict], scenario: str) -> tuple[str, dict]:
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
    scenario_instruction = EXECUTION_SCENARIOS[scenario]
    prompt = f"""Create a short, actionable, user-reviewed portfolio execution plan using the immutable public target below. Do not rerun or modify the public optimizer.

SELECTED SCENARIO: {scenario}
{scenario_instruction}

PUBLIC TARGET SNAPSHOT
{target_json}

WORKING RULES
1. Follow only the selected scenario. Ask only for the missing investment or withdrawal amount described there, then proceed. Never ask the user to choose a cash reserve. Parse a broker report only when that scenario requires one. Use security name and ISIN to resolve exchange tickers from reliable public sources. A holding is not "unresolved" merely because it is absent from the target; a resolved non-target holding has target weight 0%.
2. Do not repeat personal identifiers. Give only one short redaction warning if the report contains them.
3. Treat the broker report as the complete stock portfolio unless it explicitly says otherwise. If cash is absent, assume opening cash is zero and fund buys from sale proceeds. State this assumption once; do not stop.
4. Use the embedded target planning prices when dated within five calendar days. Broker closing prices within the same limit are acceptable for current holdings. Prefer newer reliable prices when tools permit. State the price dates once. Do not block the plan merely because prices were not independently verified.
5. Estimate both portfolios consistently. Use adjusted price history over the longest common period up to three years, requiring at least one year. Calculate each portfolio's annualized geometric return using its weights. Deduct estimated one-time taxes, brokerage, spread, and slippage from the proposed portfolio benefit. Label this a historical return-based estimate, not a guarantee.
6. Only for "Rebalance existing holdings": DECISION = REBALANCE when proposed net annualized return minus current annualized return is at least {threshold_pct:.0f} percentage points; otherwise DECISION = HOLD. Do not use this gate for fresh deployment, adding fresh cash, or raising cash. If required market-history tools are unavailable, ask only for permission to fetch prices/history or for a price-history file; do not produce a long refusal table.
7. When REBALANCE applies, calculate practical whole-share trades. Sell non-target holdings and overweight holdings first; use those proceeds for buys. Never require additional cash unless the user explicitly requests investment of new money.
8. For an existing-portfolio rebalance or cash withdrawal, reduce churn: ignore a position within 1 percentage point of target and suppress a trade below the greater of INR 100 or 0.5% of portfolio value. Do not apply that minimum to fresh deployment or BUY-only deployment of new cash.
9. For fresh or added cash, solve a whole-share integer allocation under the available budget. Repeatedly choose affordable target shares that most reduce total target-weight error, recalculate weights after each share, include estimated charges, and stop only when no additional target share fits. Do not calculate each target independently as amount × target weight and round all of them to zero.
10. A small amount may hold only a subset of the target. Prefer useful diversification and closeness to target over forcing all 21 securities. Allocate rounding residue to the most underweight affordable target. Never recommend increasing the investment amount merely because every target cannot be purchased.
11. Never include a BUY or SELL row with zero shares. Omit unavailable trades entirely. Show residual cash and never place orders automatically.

OUTPUT — KEEP IT SHORT
For "Start fresh with cash", begin with exactly:
DECISION: DEPLOY
AMOUNT: ₹x
PLANNED INVESTMENT: ₹x
RESIDUAL CASH: ₹x
STARTER COVERAGE: x target securities

For "Add fresh cash to existing holdings", begin with DECISION: ADD CASH and the same amount, planned-investment, residual-cash, and coverage lines.

Only for "Rebalance existing holdings", begin with:
DECISION: REBALANCE, HOLD, or NEEDS DATA
CURRENT ESTIMATED ANNUAL RETURN: x%
TARGET ESTIMATED ANNUAL RETURN: x%
NET ESTIMATED IMPROVEMENT: x percentage points
WHY: one sentence

For "Raise cash from existing holdings", begin with DECISION: RAISE CASH, requested amount, planned proceeds, estimated costs, and expected net proceeds.

If DECISION is DEPLOY, ADD CASH, RAISE CASH, or REBALANCE, show one execution table containing only actual trades:
Sequence | Ticker | BUY/SELL | Whole shares | Planning price | Approx. value | Reason

Then show only:
- Total sales, total purchases, estimated costs/slippage, turnover, and residual cash.
- Give an execution sentence matching the scenario. Mention sells first only when the plan actually contains sells; for fresh deployment say to execute the listed buys in sequence and recheck live prices.
- At most three warnings that could materially change execution.

If DECISION is HOLD, do not print every target row. Show a maximum of five largest allocation differences and the next review trigger.

If any security truly cannot be resolved after using its ISIN and company name, mark only that row REVIEW and continue calculating the resolvable portfolio when reasonable. Never fabricate a ticker, holding, price, return, or quantity.
"""
    return prompt, public_target


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

st.markdown(
    f"""<section class="trust-hero">
      <div class="trust-kicker">PUBLIC-01 · VERIFIED MODEL PORTFOLIO</div>
      <div class="trust-title">Invest with a clear target.</div>
      <p class="trust-subtitle">Transparent allocation, observed performance, and private execution planning.</p>
      <span class="trust-badge">✓ Immutable publication · P{int(current['portfolio_version']):03d}</span>
    </section>""",
    unsafe_allow_html=True,
)

st.subheader("Portfolio at a glance")
st.markdown(
    '<div class="metric-grid">'
    f'<div class="metric-card"><div class="metric-label">Version</div><div class="metric-value">P{int(current["portfolio_version"]):03d}</div><div class="metric-note">Immutable snapshot</div></div>'
    f'<div class="metric-card"><div class="metric-label">Constituents</div><div class="metric-value">{len(record["constituents"])}</div><div class="metric-note">Target holdings</div></div>'
    f'<div class="metric-card"><div class="metric-label">Invested target</div><div class="metric-value">{(1-float(current["cash_weight"])):.0%}</div><div class="metric-note">Across securities</div></div>'
    f'<div class="metric-card"><div class="metric-label">Data as of</div><div class="metric-value">{current["as_of"].astimezone(IST):%d %b %Y}</div><div class="metric-note">Asia/Kolkata</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.subheader("Target allocation")
allocation=pd.DataFrame(record["constituents"])
price_snapshot=load_latest_prices(tuple(allocation["ticker"].astype(str)))
if float(current["cash_weight"])>0:
    allocation=pd.concat([allocation,pd.DataFrame([{"ticker":"CASH","target_weight":current["cash_weight"]}])],ignore_index=True)
allocation["Allocation"]=allocation["target_weight"].astype(float)*100
allocation["Price"]=allocation["ticker"].map(lambda ticker: price_snapshot.get(ticker,{}).get("price"))
allocation=allocation.rename(columns={"ticker":"Security"})
allocation_rows=[]
for item in allocation[["Security","Allocation","Price"]].to_dict("records"):
    security=html.escape(str(item["Security"]))
    weight=float(item["Allocation"])
    price="N/A" if pd.isna(item["Price"]) else f"₹{float(item['Price']):,.2f}"
    allocation_rows.append(
        f'<tr><td class="ticker-cell" title="{security}">{security}</td>'
        f'<td><div class="weight-line"><span>{weight:.0f}%</span><span class="weight-track">'
        f'<span class="weight-fill" style="display:block;width:{min(max(weight,0),100):.2f}%"></span>'
        f'</span></div></td><td class="price-cell">{price}</td></tr>'
    )
st.markdown(
    '<div class="allocation-wrap"><table class="allocation-table"><thead><tr>'
    '<th>Security</th><th>Target weight</th><th>Latest close</th></tr></thead><tbody>'
    + ''.join(allocation_rows) + '</tbody></table></div>',
    unsafe_allow_html=True,
)
price_dates=sorted({item["price_as_of"] for item in price_snapshot.values()})
if price_dates:
    st.caption(f"Prices: latest available unadjusted close from Yahoo Finance · through {price_dates[-1]}")
st.caption(f"Strategy {current['strategy_version']} · Published {current['published_at'].astimezone(IST):%d %b %Y %H:%M IST}")

st.subheader("Build your private execution plan")
st.write(
    "Choose what you want to do, then give the generated prompt to the AI assistant of your choice. "
    "For an existing portfolio, attach your broker report there—not on this website."
)
execution_scenario = st.selectbox("What do you want to do?", list(EXECUTION_SCENARIOS))
st.warning(
    "If you share a broker report, first remove your name, PAN, demat/account number, email, phone, "
    "address, and any credentials. Review the AI provider's privacy policy."
)
execution_prompt, public_target = build_execution_plan_prompt(
    current, record["constituents"], price_snapshot, execution_scenario
)
version_label = f"p{int(current['portfolio_version']):03d}"
action_1, action_2, action_3, action_4 = st.columns(4)
with action_1:
    share_prompt_button(execution_prompt,version_label.upper())
with action_2:
    copy_prompt_button(execution_prompt)
action_3.download_button(
    "Download prompt.txt",
    data=execution_prompt.encode("utf-8"),
    file_name=f"{DEFAULT_BASKET_ID.lower()}-{version_label}-execution-prompt.txt",
    mime="text/plain",
    use_container_width=True,
)
action_4.download_button(
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
    "Fresh deployment and new-cash scenarios follow the public target directly. The 6 percentage-point gate "
    "applies only when replacing an existing allocation. Fresh cash uses whole-share integer allocation even "
    "for small amounts; churn thresholds apply only when selling or replacing existing holdings."
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
