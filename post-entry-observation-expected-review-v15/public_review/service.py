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
    "INSTRUMENT_METADATA_UNAVAILABLE", "INSTRUMENT_METADATA_INVALID",
    "FOREIGN_REVIEW_COST_MODEL_REQUIRED",
    "ENTRY_INTRADAY_HISTORY_UNAVAILABLE", "INVALID_POLICY_ENTRY_QUOTE_INTERVAL",
    "INVALID_POLICY_PROFIT_REVIEW_RULE",
    "INVALID_POLICY_MINIMUM_FORECAST_REVIEW_SESSIONS",
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
        after_capture = h.loc[h.index > lot["entry_date"]]
        if "Stock Splits" in after_capture and (after_capture["Stock Splits"] != 0).any():
            raise ValueError("CORPORATE_ACTION_REVIEW_REQUIRED")
        for day, row in h.loc[h.index > lot["entry_date"]].iterrows():
            dividend = float(row.get("Dividends", 0.))
            if dividend:
                dividends.append({"ticker": lot["ticker"], "date": day,
                                  "net": round(dividend * lot["quantity"] * (1 - policy["slab_rate"] * (1 + policy["surcharge_rate"]) * 1.04), 2)})
    metrics = evaluate(baseline, prices, as_of, policy, dividends)
    from .history import common_history
    closes, returns, coverage = common_history(histories, as_of, policy)
    # Reconstruct peak from the same frozen model, never from unrelated NAV/backfill.
    peak = baseline["capital"]
    for d, row in closes.loc[closes.index >= baseline.get("fully_invested_date", baseline["entry_date"])].iterrows():
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
            "decision": assessed, "forecast": forecast, "validation": validation, "history_coverage": coverage,
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
    latest_publication_id = pubs[0]["publication_id"]
    if (latest_publication_id in existing and
            existing[latest_publication_id].get("entry_model_version") != market.ENTRY_MODEL_VERSION):
        # Preserve the immutable legacy baseline, but replace it operationally
        # for the latest publication under the explicitly versioned entry rule.
        existing.pop(latest_publication_id)
    selected = [p for p in pubs if p["publication_id"] in existing or p == pubs[0]]
    failures, waiting = 0, 0
    results = []
    for publication in selected:
        baseline = existing.get(publication["publication_id"])
        baseline_id = baseline["baseline_id"] if baseline else publication["publication_id"]
        stage = "instrument_cost_model"
        try:
            from .instruments import require_supported_review
            require_supported_review(publication["weights"])
            stage = "instrument_classification"
            weights = publication["weights"]
            missing = set(weights) - set(policy["instrument_kinds"])
            if missing:
                raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
            kinds = {ticker: policy["instrument_kinds"][ticker] for ticker in weights}
            if baseline:
                if any(policy["instrument_kinds"].get(l["ticker"]) != l["kind"] for l in baseline["lots"]):
                    raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
            tickers = [r["ticker"] for r in baseline["lots"]] if baseline else list(weights)
            if baseline is None:
                stage = "entry_calendar"
                schedule = market.security_entry_schedule(
                    now, publication["published_at"], policy, kinds)
                entry_waits = {}
                captured = {row["payload"]["ticker"]: row["payload"] for row in history
                            if row["kind"] == "SECURITY_ENTRY" and
                            row["payload"].get("publication_id") == publication["publication_id"]}
                for ticker, planned in schedule.items():
                    if ticker in captured or not planned["ready"]:
                        continue
                    stage = "entry_market_data"
                    try:
                        quote = market.fetch_entry_quote(ticker, planned, policy, now=now)
                    except market.AwaitingMarketEntry as exc:
                        schedule[ticker] = exc.planned_entry
                        entry_waits[ticker] = exc
                        continue
                    except ValueError as exc:
                        raise
                    quote["publication_id"] = publication["publication_id"]
                    store.append(conn, basket,
                                 "security-entry:" + market.ENTRY_MODEL_VERSION + ":" +
                                 publication["publication_id"] + ":" + ticker,
                                 "SECURITY_ENTRY", publication["publication_id"], quote)
                    conn.commit()
                    captured[ticker] = quote
                missing_entries = sorted(set(tickers) - set(captured))
                if missing_entries:
                    from .partial import estimate_captured_security, METHOD as PARTIAL_METHOD
                    for captured_ticker, captured_entry in sorted(captured.items()):
                        try:
                            partial = estimate_captured_security(publication, captured_entry,
                                                                 policy, now)
                            partial_key = ("partial-security-review:" + PARTIAL_METHOD + ":" +
                                           publication["publication_id"] + ":" + captured_ticker +
                                           ":" + partial["as_of"] + ":" + digest(policy))
                            store.append(conn, basket, partial_key, "SECURITY_REVIEW_PREVIEW",
                                         publication["publication_id"], partial)
                            conn.commit()
                        except Exception:
                            # Entry capture must never be blocked by an optional
                            # provisional forecast. The final baseline remains authoritative.
                            conn.rollback()
                    pending = [schedule[ticker] for ticker in missing_entries]
                    next_entry = min(pending, key=lambda row: row["requested_entry_at"])
                    exc = entry_waits.get(next_entry["ticker"]) or market.AwaitingMarketEntry(
                        next_entry["entry_date"], next_entry["requested_entry_at"])
                    exc.captured_entries = len(captured)
                    exc.total_entries = len(tickers)
                    exc.pending_tickers = missing_entries
                    raise exc
                entry_prices = {ticker: float(captured[ticker]["price_inr"])
                                for ticker in tickers}
                entry = min(row["entry_date"] for row in captured.values())
                # Same economic concept as practical full-target entry, measured
                # using immutable per-security entry prices; reserve fees explicitly.
                capital = policy["capital_inr"] or math.ceil(max((p + 60) / weights[t] for t, p in entry_prices.items()) / 100) * 100
                baseline = freeze(publication, weights, entry_prices, entry, capital,
                                  policy["instrument_kinds"], policy, now.isoformat(), captured)
                baseline_id = baseline["baseline_id"]
                baseline_key = ("baseline:" + market.ENTRY_MODEL_VERSION + ":" +
                                publication["publication_id"])
                store.append(conn, basket, baseline_key, "BASELINE", baseline_id, baseline)
                conn.commit()
                history = store.read(conn, basket)
                # If another worker created this publication concurrently, use
                # the committed baseline, not the losing allocation.
                baseline = next(r["payload"] for r in history if r["event_key"] == baseline_key)
                baseline_id = baseline["baseline_id"]
                tickers = [row["ticker"] for row in baseline["lots"]]
            stage = "forecast_observation_window"
            market.require_forecast_observation_sessions(baseline, policy, now)
            stage = "session_calendar"
            entry, as_of, future = market.sessions(
                now, publication["published_at"], policy, kinds)
            if baseline:
                entry = baseline.get("fully_invested_date", baseline["entry_date"])
            stage = "market_history"
            mixed_markets = any(not ticker.endswith(".NS") for ticker in tickers)
            data = market.fetch(tickers, entry, as_of, policy,
                                allow_incomplete_end=mixed_markets)
            entry, as_of = market.synchronized_dates(
                data, entry, as_of, new_baseline=False)
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
            if (code == "STALE_OR_INCOMPLETE_MARKET_HISTORY" and
                    baseline is not None and stage in {"market_history", "assessment"}):
                try:
                    from .preview import _immediate_baseline_preview
                    ack_epoch = max((row.get("seq", 0) for row in history
                                     if row["kind"] == "ACKNOWLEDGED"), default=0)
                    provisional = _immediate_baseline_preview(
                        publication, baseline, policy, history, now, ack_epoch)
                    provisional_key = "preview:" + baseline_id + ":" + digest({
                        "policy": policy,
                        "valuation_timing": provisional.get("valuation_timing"),
                        "forecast": provisional.get("forecast"),
                    })
                    store.append(conn, basket, provisional_key, "PREVIEW",
                                 baseline_id, provisional)
                    conn.commit()
                    results.append({"publication_id": publication["publication_id"],
                                    "status": "PROVISIONAL_ASSESSED",
                                    "reason": "MIXED_TIME_CHRONOLOGY_SAFE_MARKS"})
                    continue
                except Exception:
                    conn.rollback()
            if code == "AWAITING_MARKET_ENTRY":
                waiting += 1
                reason = "AWAITING_FIRST_ASSESSMENT" if baseline is not None else code
                payload = {"status": "AWAITING_MARKET_ENTRY", "reason": code,
                           "publication_id": publication["publication_id"],
                           "entry_model_version": market.ENTRY_MODEL_VERSION,
                           "policy_version": policy["policy_version"],
                           "checked_at": now.isoformat(), "model_only": True,
                           "entry_frozen": baseline is not None,
                           "waiting_for": reason,
                           "captured_entries": getattr(exc, "captured_entries", None),
                           "total_entries": getattr(exc, "total_entries", None),
                           "pending_tickers": getattr(exc, "pending_tickers", None),
                           "pending_ticker": getattr(exc, "pending_ticker", None),
                           "wait_reason": getattr(exc, "wait_reason", None),
                           "observation_sessions": getattr(exc, "observation_sessions", None),
                           "observation_rows": getattr(exc, "observation_rows", None),
                           "entry_date": getattr(exc, "entry_date", None),
                           "ready_at": getattr(exc, "ready_at", None)}
                store.append(conn, basket, "waiting:" + baseline_id + ":" + now.isoformat(),
                             "WAITING", baseline_id, payload)
                conn.commit()
                results.append({"publication_id": publication["publication_id"],
                                "status": "WAITING", "reason": reason,
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
