"""Explicit private-profile weight-bound overrides, never verified price evidence."""
from datetime import datetime, timezone
import hashlib
import math
import re


def _recorded_weight_freeze_overrides(conn, owner):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='weight_freeze_override_events'").fetchone():
        return {}
    result={}
    for ticker,enabled,reason,recorded in conn.execute(
        "SELECT ticker,enabled,reason,recorded_at FROM weight_freeze_override_events WHERE owner=? ORDER BY id",(str(owner),)):
        if enabled:
            result[ticker]={"reason":reason,"recorded_at":recorded}
        else:
            result.pop(ticker,None)
    return result


def _owned_tickers(conn, owner):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='master_holdings'").fetchone():
        return None
    owned=set()
    for ticker,quantity in conn.execute("SELECT yahoo_ticker,quantity FROM master_holdings WHERE owner=?",(str(owner),)):
        try:
            quantity=float(quantity)
        except (TypeError,ValueError,OverflowError):
            continue
        if math.isfinite(quantity) and quantity>0:
            owned.add(str(ticker or "").strip().upper())
    return owned


def load_weight_freeze_overrides(conn, owner):
    """Read effective overrides; an exited listing has no owned weight to release."""
    active=_recorded_weight_freeze_overrides(conn,owner)
    if not active:
        return active
    owned=_owned_tickers(conn,owner)
    return active if owned is None else {t:event for t,event in active.items() if t in owned}


def retire_exited_weight_freeze_overrides(conn, owner):
    """Append restorations in the caller's holdings transaction, without committing."""
    active=_recorded_weight_freeze_overrides(conn,owner)
    if not active:
        return ()
    owned=_owned_tickers(conn,owner)
    if owned is None:
        return ()
    exited=tuple(sorted(set(active)-owned))
    conn.executemany("INSERT INTO weight_freeze_override_events(owner,ticker,enabled,reason,recorded_at) VALUES(?,?,?,?,?)",
        [(str(owner),t,0,"Holding fully exited; weight-freeze override retired",datetime.now(timezone.utc).isoformat()) for t in exited])
    return exited


def set_weight_freeze_override(conn, owner, ticker, *, enabled, reason):
    ticker=str(ticker).strip().upper();reason=str(reason).strip()
    if not str(owner).strip() or not re.fullmatch(r"[A-Z0-9^][A-Z0-9.&^=\-]{0,39}",ticker) or not reason or len(reason)>400:
        raise ValueError("An exact listing ticker, profile and short reason are required")
    if not isinstance(enabled,bool):
        raise ValueError("Override must be explicitly enabled or restored")
    owned=_owned_tickers(conn,owner) if enabled else None
    if owned is not None and ticker not in owned:
        raise ValueError("Weight changes can be allowed only for a currently owned exact listing")
    conn.execute("""CREATE TABLE IF NOT EXISTS weight_freeze_override_events(
        id INTEGER PRIMARY KEY,owner TEXT NOT NULL,ticker TEXT NOT NULL,
        enabled INTEGER NOT NULL CHECK(enabled IN(0,1)),reason TEXT NOT NULL,recorded_at TEXT NOT NULL)""")
    conn.execute("INSERT INTO weight_freeze_override_events(owner,ticker,enabled,reason,recorded_at) VALUES(?,?,?,?,?)",
                 (str(owner),ticker,int(enabled),reason,datetime.now(timezone.utc).isoformat()))
    conn.commit()
    return ticker


def released_price_freezes(quarantined, allocation):
    requested=set(allocation.attrs.get("weight_freeze_overrides") or [])
    return set(quarantined)&requested


def mark_unverified_weight_research(report, validation, tickers):
    report=dict(report);validation=dict(validation)
    reason=("Weight freeze released for "+", ".join(sorted(tickers))+
            "; historical price jumps remain unverified. Fresh weights are review-only; risk limits are unchanged.")
    report.update(status="conditional_research",research_kind="price_freeze_override",
        selected="unverified-price-research",research_only=True,constraints_applied=False,
        weight_freeze_overrides=sorted(tickers),strict_allocation_validation=validation,reason=reason)
    validation=dict(validation,passed=False,violations=list(validation.get("violations") or [])+[reason])
    return report,validation


def render_weight_freeze_controls(owner, connect):
    import streamlit as st
    key=hashlib.sha256(str(owner).encode()).hexdigest()[:12]
    with connect() as conn:
        active=load_weight_freeze_overrides(conn,owner)
    with st.expander("Weight freeze overrides",expanded=False):
        st.caption("Allow an owned listing's weight to change despite quarantined historical prices. Overrides are cleared on full exit. Price warnings, pauses and risk limits remain; resulting allocations are review-only.")
        status=st.empty()
        status.write("Weight changes allowed: "+", ".join(sorted(active)) if active else "No weight-freeze overrides.")
        with st.form(f"weight_freeze_override_{key}"):
            ticker=st.text_input("Exact listing ticker",placeholder="GVT&D.NS",key=f"freeze_ticker_{key}")
            allow=st.form_submit_button("Allow weight changes")
            restore=st.form_submit_button("Restore automatic freeze")
        if allow or restore:
            try:
                with connect() as conn:
                    changed=set_weight_freeze_override(conn,owner,ticker,enabled=bool(allow),
                        reason="Explicit profile override from weight freeze controls" if allow else "Automatic price-integrity freeze restored by profile")
                st.success(f"{changed}: "+("weight changes allowed on the next optimization; history remains unverified." if allow else "automatic freeze restored on the next optimization."))
                with connect() as conn:
                    active=load_weight_freeze_overrides(conn,owner)
                status.write("Weight changes allowed: "+", ".join(sorted(active)) if active else "No weight-freeze overrides.")
            except ValueError as exc:
                st.error(str(exc))
    return active
