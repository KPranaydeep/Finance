"""Read-only, cached-by-caller historical review; never freezes ledger entries."""
from datetime import datetime, timezone
import math
from . import market
from .core import freeze
from .forecast import estimate, validate


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
        _, as_of, days = market.sessions(
            now, publication["published_at"], policy, kinds)
        # Actual frozen model, but read-only: no orders, acknowledgment or DB writes.
        from .service import build_assessment
        b = row["payload"]
        if any(policy["instrument_kinds"].get(l["ticker"]) != l["kind"] for l in b["lots"]):
            raise ValueError("FROZEN_CLASSIFICATION_REVIEW_REQUIRED")
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
    entry_records = {ticker: market.fetch_entry_quote(ticker, entry, policy)
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
