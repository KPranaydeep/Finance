"""Scheduled model-only operations. Does not create optimizer/trade events."""
from datetime import datetime, timezone
from copy import deepcopy
import math
import os
import pandas as pd
from . import VERSION
from . import store, market
from .core import freeze, evaluate, compare_exits, decision, digest
from .forecast import estimate, validate, METHOD
from .notifications import send

SAFE_ERRORS = {
    "INSTRUMENT_CLASSIFICATION_REQUIRED", "FROZEN_CLASSIFICATION_REVIEW_REQUIRED",
    "AWAITING_MARKET_ENTRY", "CALENDAR_REVIEW_REQUIRED", "ONLY_NSE_DELIVERY_SUPPORTED",
    "MISSING_MARKET_HISTORY", "STALE_OR_INCOMPLETE_MARKET_HISTORY",
    "INVALID_OR_NONTRADING_PRICE", "INSUFFICIENT_COMMON_HISTORY",
    "COMMON_HISTORY_HAS_MISSING_SESSIONS", "CORPORATE_ACTION_REVIEW_REQUIRED",
    "ENTRY_PRICE_REVISION_REVIEW_REQUIRED", "INCOMPLETE_SESSION_CALENDAR",
    "AWARE_PUBLICATION_TIME_REQUIRED", "AWARE_TIME_REQUIRED",
}


def publications(conn, basket):
    # Read existing ledger only, without calling schema-mutating trust loaders.
    rows = conn.execute("""SELECT v.* FROM public_portfolio_versions v
        WHERE v.basket_id=%s AND v.publication_status='PUBLISHED' AND NOT EXISTS (SELECT 1 FROM public_portfolio_corrections c
          WHERE c.publication_id=v.publication_id) ORDER BY v.portfolio_version DESC""", (basket,)).fetchall()
    result = []
    for row in rows:
        row = dict(row)
        row["weights"] = {r["ticker"]: float(r["target_weight"]) for r in conn.execute(
            "SELECT ticker,target_weight FROM public_portfolio_positions WHERE publication_id=%s", (row["publication_id"],)).fetchall()}
        result.append(row)
    return result


def build_assessment(baseline, histories, as_of, future, policy, prior, latest_weights, now, comparisons=True):
    prices = {t: float(h.loc[as_of, "Close"]) for t, h in histories.items()}
    dividends = []
    for lot in baseline["lots"]:
        h = histories[lot["ticker"]]
        # Yahoo historical closes are split-normalized. A subsequent split makes
        # frozen units incomparable; do not silently use the wrong quantities.
        after_capture = h.loc[h.index > baseline["captured_at"][:10]]
        if "Stock Splits" in after_capture and (after_capture["Stock Splits"] != 0).any():
            raise ValueError("CORPORATE_ACTION_REVIEW_REQUIRED")
        if not math.isclose(float(h.loc[baseline["entry_date"], "Open"]), lot["price"], rel_tol=.005):
            raise ValueError("ENTRY_PRICE_REVISION_REVIEW_REQUIRED")
        for day, row in h.loc[h.index > baseline["entry_date"]].iterrows():
            dividend = float(row.get("Dividends", 0.))
            if dividend:
                dividends.append({"ticker": lot["ticker"], "date": day,
                                  "net": round(dividend * lot["quantity"] * (1 - policy["slab_rate"] * (1 + policy["surcharge_rate"]) * 1.04), 2)})
    metrics = evaluate(baseline, prices, as_of, policy, dividends)
    closes = pd.DataFrame({t: h.Close for t, h in histories.items()}).sort_index().dropna()
    if len(closes) < 2 or not (closes > 0).all().all():
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    expected = market.calendar(closes.index[0], as_of, policy)
    expected_days = {str(d.date()) for d in expected.index}
    if not expected_days.issubset(set(closes.index)):
        raise ValueError("COMMON_HISTORY_HAS_MISSING_SESSIONS")
    returns = closes.pct_change(fill_method=None).dropna()
    # Reconstruct peak from the same frozen model, never from unrelated NAV/backfill.
    peak = baseline["capital"]
    for d, row in closes.loc[closes.index >= baseline["entry_date"]].iterrows():
        m = evaluate(baseline, row.to_dict(), d, policy, [x for x in dividends if x["date"] <= d])
        peak = max(peak, m["net_proceeds"])
    validation = {"passed": False, "reason": "INSUFFICIENT_COMMON_HISTORY"}
    forecast = {"status": "INSUFFICIENT_COMMON_HISTORY", "next_review": None}
    if len(returns) >= 126:
        validation = validate(returns, baseline, prices, list(returns.index), policy)
        future_baseline = deepcopy(baseline)
        future_baseline["cash"] += sum(x["net"] for x in dividends)
        forecast = estimate(future_baseline, prices, returns, future, policy, peak, validation, dividends=dividends)
    ack = store.latest(prior, "ACKNOWLEDGED", baseline["baseline_id"])
    last = store.latest(prior, "ASSESSMENT", baseline["baseline_id"])
    promised = (last["payload"].get("decision", {}).get("next_review") if last and
                (not ack or ack["seq"] < last["seq"]) else None)
    # Keep an operational risk-check date even when statistical timing fails
    # validation; do not label that fallback as a target-crossing forecast.
    timing = {"next_review": forecast.get("next_review") or (future[0] if future else None)}
    assessed = decision(metrics, baseline, latest_weights, policy, peak, timing, promised)
    assessed["date_basis"] = "VALIDATED_FORECAST" if forecast.get("next_review") else "NEXT_SESSION_RISK_CHECK"
    try:
        from public_market_mood import fetch_mmi
        mood = fetch_mmi()
    except Exception:
        mood = {"status": "UNAVAILABLE", "role": "context_only"}
    return {"version": VERSION, "baseline_id": baseline["baseline_id"], "as_of": as_of,
            "checked_at": now.isoformat(), "policy": policy, "metrics": metrics,
            "decision": assessed, "forecast": forecast, "validation": validation,
            "comparisons": compare_exits(baseline, prices, as_of, policy, metrics) if comparisons else [],
            "mmi": mood, "price_hash": digest({t: {d: float(v) for d, v in h.Close.items()} for t, h in histories.items()}),
            "dividend_assumption": "Net distributions credited as model cash on ex-date, not verified broker payment dates",
            "model_only": True}


def run(conn, basket, policy, *, acknowledge=None, now=None):
    now = now or datetime.now(timezone.utc)
    store.init(conn)
    conn.commit()
    pubs = publications(conn, basket)
    if not pubs:
        raise ValueError("NO_ACTIVE_PUBLICATION")
    history = store.read(conn, basket)
    if acknowledge:
        if not any(r["kind"] == "BASELINE" and r["baseline_id"] == acknowledge for r in history):
            raise ValueError("UNKNOWN_BASELINE")
        store.append(conn, basket, "ack:" + acknowledge + ":" + now.date().isoformat(), "ACKNOWLEDGED", acknowledge,
                     {"at": now.isoformat(), "note": "Review acknowledged; no trade recorded"})
        conn.commit()
        history = store.read(conn, basket)
    # Continue monitoring already-created investments. Latest is added, never
    # replaces previous baselines or changes their entry clock.
    existing = {r["payload"]["publication_id"]: r["payload"] for r in history if r["kind"] == "BASELINE"}
    selected = [p for p in pubs if p["publication_id"] in existing or p == pubs[0]]
    failures, waiting = 0, 0
    results = []
    for publication in selected:
        baseline = existing.get(publication["publication_id"])
        baseline_id = baseline["baseline_id"] if baseline else publication["publication_id"]
        stage = "session_calendar"
        try:
            entry, as_of, future = market.sessions(now, publication["published_at"], policy)
            stage = "instrument_classification"
            if baseline:
                entry = baseline["entry_date"]
                if any(policy["instrument_kinds"].get(l["ticker"]) != l["kind"] for l in baseline["lots"]):
                    raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
            weights = publication["weights"]
            missing = set(weights) - set(policy["instrument_kinds"])
            if missing:
                raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
            tickers = [r["ticker"] for r in baseline["lots"]] if baseline else list(weights)
            stage = "market_history"
            data = market.fetch(tickers, entry, as_of, policy)
            stage = "freeze_baseline"
            if baseline is None:
                entry_prices = {t: float(h.loc[entry, "Open"]) for t, h in data.items()}
                # Same economic concept as practical full-target entry, measured
                # using ENTRY prices, not today's prices; reserve fees explicitly.
                capital = policy["capital_inr"] or math.ceil(max((p + 60) / weights[t] for t, p in entry_prices.items()) / 100) * 100
                baseline = freeze(publication, weights, entry_prices, entry, capital,
                                  policy["instrument_kinds"], policy, now.isoformat())
                baseline_id = baseline["baseline_id"]
                store.append(conn, basket, "baseline:" + publication["publication_id"], "BASELINE", baseline_id, baseline)
                conn.commit()
                history = store.read(conn, basket)
                # If another worker created this publication concurrently, use
                # the committed baseline, not the losing allocation.
                baseline = next(r["payload"] for r in history if r["event_key"] == "baseline:" + publication["publication_id"])
                baseline_id = baseline["baseline_id"]
                data = {t: h for t, h in data.items() if t in {r["ticker"] for r in baseline["lots"]}}
            stage = "assessment"
            payload = build_assessment(baseline, data, as_of, future, policy, history, pubs[0]["weights"], now)
            key = "assessment:" + baseline_id + ":" + digest({"method": METHOD, "as_of": as_of, "policy": policy,
                  "prices": payload["price_hash"], "ack": (store.latest(history, "ACKNOWLEDGED", baseline_id) or {}).get("seq")})
            store.append(conn, basket, key, "ASSESSMENT", baseline_id, payload)
            store.append(conn, basket, "heartbeat:" + baseline_id + ":" + now.isoformat(),
                         "HEARTBEAT", baseline_id, {"at": now.isoformat(), "as_of": as_of,
                         "alerts_enabled": os.getenv("PUBLIC_REVIEW_ALERT_CHANNEL", "none") in {"email", "telegram"}})
            conn.commit()
            stage = "notification"
            alert_ok = notify_safely(conn, basket, baseline_id, payload, now)
            if not alert_ok:
                failures += 1
            results.append({"publication_id": publication["publication_id"],
                            "status": "ASSESSED" if alert_ok else "ALERT_FAILED",
                            "reason": payload["decision"]["status"] if alert_ok else "ALERT_DELIVERY_FAILED"})
        except Exception as exc:
            conn.rollback()
            # Persist only allowlisted safe codes; never DB URLs, provider bodies
            # or exception traces (may contain credentials).
            code = str(exc) if isinstance(exc, ValueError) and str(exc) in SAFE_ERRORS else "MONITOR_CHECK_FAILED"
            if code == "AWAITING_MARKET_ENTRY" and baseline is None:
                waiting += 1
                payload = {"status": "AWAITING_MARKET_ENTRY", "reason": code,
                           "publication_id": publication["publication_id"],
                           "checked_at": now.isoformat(), "model_only": True,
                           "entry_date": getattr(exc, "entry_date", None),
                           "ready_at": getattr(exc, "ready_at", None)}
                store.append(conn, basket, "waiting:" + baseline_id + ":" + now.isoformat(),
                             "WAITING", baseline_id, payload)
                conn.commit()
                results.append({"publication_id": publication["publication_id"],
                                "status": "WAITING", "reason": code,
                                "entry_date": payload["entry_date"], "ready_at": payload["ready_at"]})
                continue
            failures += 1
            payload = {"status": "CANNOT_ASSESS", "reason": code, "checked_at": now.isoformat(),
                       "model_only": True, "publication_id": publication["publication_id"], "stage": stage}
            store.append(conn, basket, "failure:" + baseline_id + ":" + now.isoformat() + ":" + code,
                         "FAILURE", baseline_id, payload)
            conn.commit()
            notify_safely(conn, basket, baseline_id, payload, now)
            results.append({"publication_id": publication["publication_id"],
                            "status": "CANNOT_ASSESS", "reason": code, "stage": stage})
    return {"checked": len(selected), "failed": failures, "waiting": waiting, "results": results}


def maybe_notify(conn, basket, baseline_id, payload, now):
    decision_ = payload.get("decision", {})
    status = decision_.get("status", payload.get("status", "CANNOT_ASSESS"))
    review = decision_.get("next_review")
    fingerprint = digest({"baseline": baseline_id, "status": status, "date": review,
                          "reason": payload.get("reason"), "reasons": decision_.get("reasons", [])})
    history = store.read(conn, basket)
    last = store.latest(history, "ALERT_SENT", baseline_id)
    if last and last["payload"].get("fingerprint") == fingerprint:
        return
    review_label = "now" if decision_.get("reasons") else (review or "unavailable")
    message = (f"{basket} model review: {status.replace('_', ' ')}. "
               f"Next suggested review: {review_label}. "
               f"Checked: {now.isoformat()}. Model estimates only, not a trade instruction. "
               "Open https://theportfolio.streamlit.app/ for details.")
    if send(message):
        store.append(conn, basket, "alert:" + baseline_id + ":" + fingerprint + ":" + now.isoformat(),
                     "ALERT_SENT", baseline_id, {"fingerprint": fingerprint, "at": now.isoformat()})
        conn.commit()


def notify_safely(conn, basket, baseline_id, payload, now):
    try:
        maybe_notify(conn, basket, baseline_id, payload, now)
        return True
    except Exception:
        conn.rollback()
        store.append(conn, basket, "alert-failed:" + baseline_id + ":" + now.isoformat(),
                     "ALERT_FAILED", baseline_id, {"at": now.isoformat(), "reason": "ALERT_DELIVERY_FAILED"})
        conn.commit()
        return False
