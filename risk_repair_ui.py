"""Independent, dated risk-reduction draft; never an order submission surface."""
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from risk_repair import load_repair


def render_risk_repair(owner, connect, current_context, prepare, *, key):
    with st.expander("Sell-only risk reduction", expanded=False):
        st.caption("Uses actual current holdings; keeps sale proceeds as cash. Independent of saved optimal weights and X/Y search.")
        if st.button("Prepare risk-reduction draft", key=f"risk_repair_{key}", icon=":material/health_and_safety:"):
            try:
                with st.spinner("Checking owned histories and calculating minimum-turnover reductions…"):
                    prepare()
            except Exception as exc:
                st.warning(f"Could not refresh the draft; previous snapshot retained: {exc}")
        with connect() as conn:
            report = load_repair(conn, owner)
        if not report:
            st.info("No risk-reduction draft prepared yet.")
            return
        try:
            stale = report.get("context") != current_context()
        except (ValueError, TypeError, KeyError):
            stale = True
        status = report.get("status")
        if stale:
            st.warning("Snapshot needs refresh: holdings, restrictions, policy or date changed. This older draft is not current.")
        elif status == "checked_draft":
            st.success("Sell-only draft passed its recorded checks. Review the dated evidence before placing any order.")
        elif status == "no_repair_needed":
            st.info("Current holdings passed this policy snapshot; no sell-only repair was needed.")
        else:
            st.warning("No complete checked repair established. No trades released.")
        try:
            prepared = datetime.fromisoformat(report["computed_at"]).astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %H:%M IST")
        except (ValueError, TypeError, KeyError):
            prepared = "unavailable"
        st.caption(f"Prepared {prepared} · history through {report.get('history_through','unavailable')} · {report.get('elapsed_seconds',0):.2f}s calculation · X/Y search not run.")
        if report.get("reason"):
            st.write(report["reason"])
        if report.get("blocked_tickers"):
            st.caption("Preserved paused/frozen holdings: " + ", ".join(report["blocked_tickers"]))
        if report.get("rows"):
            st.dataframe(pd.DataFrame(report["rows"]), hide_index=True, width="stretch", column_config={
                **{n:st.column_config.NumberColumn(format="%.2f%%") for n in ("Current allocation (%)","After repair (%)","Cap (%)")},
                "Sell quantity":st.column_config.NumberColumn(format="%.8f"),
                "Sale value (INR)":st.column_config.NumberColumn(format="%.2f"),
            })
        before, after = report.get("before") or {}, report.get("validation") or {}
        if after:
            st.write(f"Proposed cash after modeled fees: ₹{report.get('cash_after_costs_inr',after.get('cash_after_costs_inr',0)):,.2f} · estimated fees ₹{report.get('estimated_cost_inr',0):,.2f}.")
            b, a = before.get("allocation_policy") or {}, after.get("allocation_policy") or {}
            if b.get("cvar") is not None and a.get("cvar") is not None:
                st.caption(f"Same-scenario tail loss: {b['cvar']:.2%} → {a['cvar']:.2%}. Historical drawdown: {(before.get('risk') or {}).get('historical_drawdown',0):.2%} → {(after.get('risk') or {}).get('historical_drawdown',0):.2%}.")
            for violation in after.get("violations") or []:
                st.write("• " + violation)
        elif report.get("before"):
            st.caption("Current unresolved checks; no after-repair allocation was established:")
            for violation in before.get("violations") or []:
                st.write("• " + violation)
        st.caption("Minimizes gross sales in recorded tradable lots under concentration and joint-tail budgets; historical drawdown/volatility are checked afterwards. Tax lots/taxes and reserve yield are not modeled. A sale does not itself improve whole-account XIRR. No broker orders are submitted.")
        if status == "checked_draft" and not stale and report.get("orders"):
            st.download_button("Download risk-reduction draft CSV", pd.DataFrame(report["orders"]).to_csv(index=False).encode(),
                file_name="risk_reduction_draft.csv", mime="text/csv", key=f"repair_csv_{key}")
