"""Read-only, cached-by-caller historical review; never freezes ledger entries."""
from datetime import datetime, timezone, timedelta
import math
import pandas as pd
from . import market
from .core import freeze
from .forecast import estimate, validate


def _immediate_baseline_preview(publication, baseline, policy, events, now, ack_epoch):
    """Forecast immediately from chronology-safe marks before global close.

    Entry evidence remains immutable. Each holding uses its latest completed
    post-entry close when available, otherwise its frozen entry price. A later
    synchronized assessment supersedes this read-only preview.
    """
    from .history import common_history
    tickers = [lot["ticker"] for lot in baseline["lots"]]
    local_day = pd.Timestamp(now).tz_convert("Asia/Kolkata").date()
    histories = market.fetch(tickers, baseline["entry_date"], str(local_day),
                             policy, allow_incomplete_end=True)
    buffer = pd.Timedelta(minutes=policy["assessment_wait_after_close_minutes"])
    marks, timing, completed_cutoffs = {}, [], []
    for lot in baseline["lots"]:
        ticker = lot["ticker"]
        market_name = market.KIND_CALENDARS[lot["kind"]]
        schedule = market.calendar(
            pd.Timestamp(lot["entry_date"]) - pd.Timedelta(days=10),
            local_day + timedelta(days=1), policy, market_name)
        completed = schedule[schedule.market_close + buffer <= pd.Timestamp(now)]
        if not completed.empty:
            completed_cutoffs.append(str(completed.index[-1].date()))
        post_entry = []
        for day in completed.index:
            price_date = str(day.date())
            if price_date < lot["entry_date"] or price_date not in histories[ticker].index:
                continue
            value = pd.to_numeric(
                pd.Series([histories[ticker].loc[price_date, "Close"]]),
                errors="coerce").iloc[0]
            if math.isfinite(float(value)) and float(value) > 0:
                post_entry.append(price_date)
        if post_entry:
            price_date = post_entry[-1]
            price = float(histories[ticker].loc[price_date, "Close"])
            mode = "LATEST_COMPLETED_POST_ENTRY_CLOSE"
            observed_at = price_date
        else:
            price = float(lot["price"])
            mode = "FROZEN_ENTRY_PRICE_PENDING_FIRST_CLOSE"
            observed_at = lot.get("entry_quote_at") or lot.get("entry_at") or lot["entry_date"]
        if not math.isfinite(price) or price <= 0:
            raise ValueError("INVALID_OR_NONTRADING_PRICE")
        marks[ticker] = price
        timing.append({"ticker": ticker, "price_source": mode,
                       "price_observed_at": str(observed_at),
                       "chronology_valid": True})
    if not completed_cutoffs:
        raise market.AwaitingMarketEntry(baseline["entry_date"], None)
    completed_cutoff = min(completed_cutoffs)
    common_valid = None
    for history in histories.values():
        close = pd.to_numeric(history["Close"], errors="coerce")
        valid = {str(day) for day, value in close.items()
                 if str(day) <= completed_cutoff and
                 math.isfinite(float(value)) and float(value) > 0}
        common_valid = valid if common_valid is None else common_valid & valid
    if not common_valid:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    history_as_of = max(common_valid)
    _, returns, coverage = common_history(histories, history_as_of, policy)
    returns = returns[tickers]
    if len(returns) < 126:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    kinds = {lot["ticker"]: lot["kind"] for lot in baseline["lots"]}
    schedule = market.joint_calendar(local_day, policy["calendar_verified_through"],
                                     policy, kinds)
    future = [str(day.date()) for day, session in schedule.iterrows()
              if session.market_open > pd.Timestamp(now)][:policy["max_review_sessions"]]
    if len(future) < policy["max_review_sessions"]:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    validation = validate(returns, baseline, marks, list(returns.index), policy)
    forecast = estimate(baseline, marks, returns, future, policy,
                        baseline["capital"], validation)
    candidate = forecast.get("next_review") or forecast.get("research_candidate") or future[0]
    prior = [r["payload"].get("decision", {}).get("next_review") for r in events
             if r["kind"] == "PREVIEW" and r["baseline_id"] == baseline["baseline_id"]]
    candidate = min([candidate] + [day for day in prior if day])
    price_dates = {row["price_observed_at"][:10] for row in timing}
    return {"provisional": True, "publication_id": publication["publication_id"],
            "history_coverage": coverage, "ack_epoch": ack_epoch,
            "as_of": "mixed chronology-safe marks", "checked_at": pd.Timestamp(now).isoformat(),
            "assumption": "Immediate entry-time forecast. Frozen entry prices are used until each market supplies a completed post-entry close; entry evidence is never revised.",
            "valuation_timing": {"mode": "ASYNCHRONOUS_CHRONOLOGY_SAFE",
                                  "all_prices_synchronized": len(price_dates) == 1,
                                  "rows": timing},
            "forecast": forecast,
            "decision": {"next_review": candidate, "reasons": [],
                         "target_crossed_securities": [],
                         "basis": "PROVISIONAL_ENTRY_TIME_FORECAST"}}


def historical_preview(publication, policy, events, now=None):
    from .instruments import complete_policy, require_supported_review
    policy = complete_policy(policy, publication["weights"])
    require_supported_review(publication["weights"])
    now = now or datetime.now(timezone.utc)
    ack_epoch = max((r.get("seq", 0) for r in events if r["kind"] == "ACKNOWLEDGED"), default=0)
    kinds = {ticker: policy["instrument_kinds"][ticker] for ticker in publication["weights"]}
    row = next((r for r in reversed(events) if r["kind"] == "BASELINE" and
                r["payload"]["publication_id"] == publication["publication_id"]), None)
    if row:
        # Actual frozen model, but read-only: no orders, acknowledgment or DB writes.
        from .service import build_assessment
        b = row["payload"]
        if any(policy["instrument_kinds"].get(l["ticker"]) != l["kind"] for l in b["lots"]):
            raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
        try:
            _, as_of, days = market.sessions(
                now, publication["published_at"], policy, kinds)
        except market.AwaitingMarketEntry:
            return _immediate_baseline_preview(
                publication, b, policy, events, now, ack_epoch)
        tickers = [l["ticker"] for l in b["lots"]]
        mixed_markets = any(not ticker.endswith(".NS") for ticker in tickers)
        histories = market.fetch(tickers, b["entry_date"], as_of, policy,
                                 allow_incomplete_end=mixed_markets)
        _, as_of = market.synchronized_dates(
            histories, b["entry_date"], as_of, new_baseline=False)
        days = [day for day in days if day > as_of]
        if not days:
            raise ValueError("INCOMPLETE_SESSION_CALENDAR")
        p = build_assessment(b, histories, as_of, days, policy, events, publication["weights"],
                             now, comparisons=False)
        return {"provisional": False, "as_of": as_of, "checked_at": now.isoformat(),
                "ack_epoch": ack_epoch,
                "forecast": p["forecast"], "decision": p["decision"], "publication_id": publication["publication_id"]}
    weights = publication["weights"]
    if set(weights) - set(policy["instrument_kinds"]):
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    entry_records = market.security_entry_schedule(
        now, publication["published_at"], policy, kinds)
    pending = [row for row in entry_records.values() if not row["ready"]]
    if pending:
        next_entry = min(pending, key=lambda item: item["requested_entry_at"])
        raise market.AwaitingMarketEntry(next_entry["entry_date"],
                                         next_entry["requested_entry_at"])
    entry_records = {ticker: market.fetch_entry_quote(ticker, entry, policy, now=now)
                     for ticker, entry in entry_records.items()}
    planned_entry = min(row["entry_date"] for row in entry_records.values())
    fully_invested = max(row["entry_date"] for row in entry_records.values())
    _, as_of, days = market.sessions(now, publication["published_at"], policy, kinds)
    mixed_markets = any(not ticker.endswith(".NS") for ticker in weights)
    histories = market.fetch(list(weights), fully_invested, as_of, policy,
                             allow_incomplete_end=mixed_markets)
    entry, as_of = market.synchronized_dates(
        histories, fully_invested, as_of, new_baseline=False)
    days = [day for day in days if day > as_of]
    if not days:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    from .history import common_history
    closes, all_returns, coverage = common_history(histories, as_of, policy)
    if len(all_returns) < 126:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    prices = closes.loc[as_of].to_dict()
    entry_prices = {ticker: float(record["price_inr"])
                    for ticker, record in entry_records.items()}
    # Same hypothetical entry rule as the durable workflow: first shared
    # session after publication, at that session's verified opening prices.
    capital = policy["capital_inr"] or math.ceil(max(
        (price + 60) / weights[ticker] for ticker, price in entry_prices.items()) / 100) * 100
    b = freeze(publication, weights, entry_prices, planned_entry, capital,
               policy["instrument_kinds"], policy, now.isoformat(), entry_records)
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
            "assumption": "Hypothetical entry at the first eligible shared market session after publication, using verified opening prices and modeled entry charges; not an actual trade.",
            "forecast": f, "decision": {"next_review": candidate, "reasons": [],
            "target_crossed_securities": [], "basis": "VALIDATED_FORECAST" if f["next_review"] else "NEXT_SESSION_RISK_CHECK"}}
