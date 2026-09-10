"""Read-only, cached-by-caller historical review; never freezes ledger entries."""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import math
import pandas as pd
from . import market
from .core import freeze
from .forecast import estimate, validate


def historical_preview(publication, policy, events, now=None):
    from .instruments import complete_policy
    policy = complete_policy(policy, publication["weights"])
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo("Asia/Kolkata")).date()
    ack_epoch = max((r.get("seq", 0) for r in events if r["kind"] == "ACKNOWLEDGED"), default=0)
    schedule = market.calendar(local - timedelta(days=10),
                               min(str(local + timedelta(days=100)), policy["calendar_verified_through"]), policy)
    completed = schedule[schedule.market_close + pd.Timedelta(minutes=30) <= pd.Timestamp(now)]
    future = schedule[schedule.market_close + pd.Timedelta(minutes=30) > pd.Timestamp(now)]
    if completed.empty or len(future) < policy["max_review_sessions"]:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    as_of = str(completed.index[-1].date())
    days = [str(d.date()) for d in future.index[:policy["max_review_sessions"]]]
    row = next((r for r in reversed(events) if r["kind"] == "BASELINE" and
                r["payload"]["publication_id"] == publication["publication_id"]), None)
    if row:
        # Actual frozen model, but read-only: no orders, acknowledgment or DB writes.
        from .service import build_assessment
        b = row["payload"]
        if any(policy["instrument_kinds"].get(l["ticker"]) != l["kind"] for l in b["lots"]):
            raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
        histories = market.fetch([l["ticker"] for l in b["lots"]], b["entry_date"], as_of, policy)
        p = build_assessment(b, histories, as_of, days, policy, events, publication["weights"],
                             now, comparisons=False)
        return {"provisional": False, "as_of": as_of, "checked_at": now.isoformat(),
                "ack_epoch": ack_epoch,
                "forecast": p["forecast"], "decision": p["decision"], "publication_id": publication["publication_id"]}
    weights = publication["weights"]
    if set(weights) - set(policy["instrument_kinds"]):
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    histories = market.fetch(list(weights), as_of, as_of, policy, allow_incomplete_end=True)
    # Provisional estimates may use the previous completed session, explicitly
    # dated. Never substitute old prices into an actual frozen model valuation.
    import numpy as np
    candidates = pd.DataFrame({t: h.Close for t, h in histories.items()}).sort_index()
    candidates = candidates.where(np.isfinite(candidates) & (candidates > 0)).dropna()
    eligible_days = [str(d.date()) for d in completed.index[-2:]]
    available = [d for d in eligible_days if d in candidates.index and all(
        "Volume" not in h or (np.isfinite(h.loc[d, "Volume"]) and h.loc[d, "Volume"] > 0)
        for h in histories.values())]
    if not available:
        raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
    as_of = available[-1]
    from .history import common_history
    closes, all_returns, coverage = common_history(histories, as_of, policy)
    if len(all_returns) < 126:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    prices = closes.loc[as_of].to_dict()
    # Explicit assumed entry today at the last completed close. Not an assertion
    # that these prices were executable at publication, and never persisted as BASELINE.
    entry = str(local)
    days = [d for d in days if d > entry]
    if not days:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    capital = policy["capital_inr"] or math.ceil(max((p + 60) / weights[t] for t, p in prices.items()) / 100) * 100
    b = freeze(publication, weights, prices, entry, capital, policy["instrument_kinds"], policy, now.isoformat())
    held = [l["ticker"] for l in b["lots"]]
    returns = all_returns[held]
    v = validate(returns, b, {t: prices[t] for t in held}, list(returns.index), policy)
    f = estimate(b, prices, returns, days, policy, capital, v)
    # A research forecast must not become a validated trading recommendation.
    # Until validation passes, review next session rather than invent a crossing.
    candidate = f["next_review"] or days[0]
    prior = [r["payload"].get("decision", {}).get("next_review") for r in events
             if r["kind"] == "PREVIEW" and r["baseline_id"] == publication["publication_id"]]
    candidate = min([candidate] + [d for d in prior if d])
    return {"provisional": True, "publication_id": publication["publication_id"],
            "history_coverage": coverage,
            "ack_epoch": ack_epoch,
            "as_of": as_of, "checked_at": now.isoformat(), "assumed_entry_date": entry,
            "assumption": "Hypothetical entry today at latest completed close, including modeled entry charges; not earned returns.",
            "forecast": f, "decision": {"next_review": candidate, "reasons": [],
            "target_crossed_securities": [], "basis": "VALIDATED_FORECAST" if f["next_review"] else "NEXT_SESSION_RISK_CHECK"}}
