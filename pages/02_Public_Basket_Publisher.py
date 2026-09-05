"""Operator-only publication of an already-approved private optimizer result."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from public_basket_postgres import DEFAULT_BASKET_ID, connect_public_basket_db, get_public_basket_database_url
from public_portfolio_publications import publish_approved_portfolio, validate_constituents
from public_portfolio_trust import canonical_json, round_weights_to_whole_percent

IST=ZoneInfo("Asia/Kolkata")
PAGE_VERSION="approved-snapshot-publisher-v1"
st.set_page_config(page_title="Public Portfolio Publisher",page_icon="🧾",layout="wide")
st.title("🧾 Public Portfolio Publisher")
st.caption(f"{PAGE_VERSION} · Publishes approved private results · Never runs the optimizer")
st.warning("Operator-only. Each publication is immutable and creates a new portfolio version.")

try: token=str(st.secrets["public_basket"].get("publisher_token","")).strip()
except Exception: token=""
if len(token)<32:
    st.error("Configure [public_basket].publisher_token with at least 32 characters."); st.stop()
entered=st.text_input("Publisher token",type="password")
if not entered or not hmac.compare_digest(entered,token):
    st.info("Enter the private publisher token to continue."); st.stop()

uploaded=st.file_uploader("Approved private optimizer result",type=["json"])
if uploaded is None:
    st.info("Upload the approved JSON exported by the private rebalancer."); st.stop()
try:
    source=json.loads(uploaded.getvalue().decode("utf-8"))
    approved=source.get("publication_candidate") or source.get("approved_publication") or source
    allocation=approved.get("constituents") or approved.get("optimal_allocation")
    if not isinstance(allocation,list): raise ValueError("Missing approved constituents")
    constituents=[]
    for row in allocation:
        ticker=row.get("ticker") or row.get("Yahoo Ticker") or row.get("yahoo_ticker")
        weight=row.get("target_weight",row.get("Optimal Weight"))
        constituents.append({"ticker":ticker,"target_weight":weight})
    cash_weight=float(approved.get("cash_weight",0.0))
    constituents=round_weights_to_whole_percent(constituents,cash_weight=cash_weight)
    constituents=validate_constituents(constituents,cash_weight)
    as_of=datetime.fromisoformat(str(approved.get("as_of") or source.get("saved_at")))
    if as_of.tzinfo is None: as_of=as_of.replace(tzinfo=IST)
    run_id=str(approved.get("run_id") or source.get("run_id") or hashlib.sha256(uploaded.getvalue()).hexdigest()[:24])
    calculation_version=str(approved.get("calculation_version") or "private-rebalancer-v1")
    strategy_version=str(approved.get("strategy_version") or "portfolio-rebalancer-v1")
except Exception as exc:
    st.error(f"This is not a valid approved publication: {exc}"); st.stop()

material={"basket_id":DEFAULT_BASKET_ID,"run_id":run_id,"as_of":as_of.isoformat(),
          "calculation_version":calculation_version,"strategy_version":strategy_version,
          "cash_weight":cash_weight,"constituents":constituents}
digest=hashlib.sha256(canonical_json(material).encode()).hexdigest()
st.subheader("Read-only publication preview")
c1,c2=st.columns(2); c1.metric("Constituents",len(constituents)); c2.metric("Fingerprint",digest[:12])
display_rows=[{"Ticker":row["ticker"],"Weight":f"{row['target_weight']:.0%}"} for row in constituents]
display_rows.append({"Ticker":"TOTAL","Weight":f"{sum(row['target_weight'] for row in constituents)+cash_weight:.0%}"})
st.dataframe(display_rows,use_container_width=True,hide_index=True)
st.caption(f"As of {as_of.astimezone(IST):%d %b %Y %H:%M IST} · Strategy {strategy_version} · Calculation {calculation_version}")

phrase=f"PUBLISH {digest[:12].upper()}"
confirmed=st.checkbox("I approved this private optimizer result and reviewed every constituent weight.")
typed=st.text_input(f"Type exactly: {phrase}")
if st.button("Publish immutable portfolio version",type="primary",disabled=not(confirmed and typed==phrase)):
    try:
        with connect_public_basket_db(get_public_basket_database_url()) as conn:
            result=publish_approved_portfolio(conn,basket_id=DEFAULT_BASKET_ID,run_id=run_id,as_of=as_of,
                calculation_version=calculation_version,strategy_version=strategy_version,
                constituents=constituents,cash_weight=cash_weight)
        st.success(f"Published portfolio version P{int(result['portfolio_version']):03d}.")
        st.json({"publication_id":result["publication_id"],"portfolio_version":result["portfolio_version"],"fingerprint":result["portfolio_fingerprint"]})
        st.cache_data.clear()
    except Exception as exc:
        st.error(f"Publication failed; no partial version was created: {exc}")
