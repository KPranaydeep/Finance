"""Read-only, cached-by-caller historical review; never freezes ledger entries."""
from datetime import datetime, timezone, timedelta
import math
import pandas as pd
from . import market
from .core import digest, evaluate, freeze
from .forecast import estimate, validate


def _planning_outlook(returns, baseline, forecast_date):
    """Return a clearly provisional 28-calendar-day constant-weight outlook."""
    from public_outlook import calendar_outlook

    tickers = [lot["ticker"] for lot in baseline["lots"]]
    weights = pd.Series({
        ticker: float(baseline["weights"].get(ticker, 0.0))
        for ticker in tickers
    })
    if weights.sum() <= 0:
        return None
    weights = weights / weights.sum()
    portfolio_returns = returns[tickers].mul(weights, axis=1).sum(axis=1)
    nav = (1.0 + portfolio_returns).cumprod() * 100.0
    nav_dates = pd.to_datetime(nav.index, errors="coerce").normalize()
    if nav_dates.isna().any() or nav_dates.duplicated().any():
        return None
    rows = [
        {"nav_date": day.date(), "nav": float(value)}
        for day, value in zip(nav_dates, nav.to_numpy())
    ]
    return calendar_outlook(rows, pd.Timestamp(forecast_date).date())


def _immediate_baseline_preview(
    publication, baseline, policy, events, now, ack_epoch, *,
    history_cutoff=None, histories=None,
    price_source="FROZEN_ENTRY_PRICE", assumption=None,
):
    """Create a publication-time planning forecast without post-entry returns.

    Historical returns end at the latest date that was observable for every
    represented market when the portfolio was published. Frozen entry prices
    are used only to measure the future net-XIRR hurdle. A later observed
    assessment supersedes this read-only planning estimate.
    """
    from .history import common_history
    tickers = [lot["ticker"] for lot in baseline["lots"]]
    published = pd.Timestamp(publication["published_at"])
    if published.tzinfo is None:
        raise ValueError("AWARE_PUBLICATION_TIME_REQUIRED")
    if history_cutoff is None:
        completed_at_publication = []
        for lot in baseline["lots"]:
            schedule = market.calendar(
                published.date() - timedelta(days=14), published.date(), policy,
                market.KIND_CALENDARS[lot["kind"]])
            completed = schedule[schedule.market_close <= published]
            if completed.empty:
                raise ValueError("INSUFFICIENT_COMMON_HISTORY")
            completed_at_publication.append(str(completed.index[-1].date()))
        publication_cutoff = min(completed_at_publication)
    else:
        publication_cutoff = str(history_cutoff)[:10]
    # This is a publication-time *historical* forecast.  Its sample must end
    # before publication, but it must not start at the newly frozen entry date:
    # for a fresh publication that would create an empty or inverted range and
    # hide the planning review date until live observations arrive.
    history_start = (
        pd.Timestamp(publication_cutoff)
        - pd.DateOffset(years=int(policy["history_years"]))
    ).date().isoformat()
    if histories is None:
        histories = market.fetch(tickers, history_start, publication_cutoff,
                                 policy, allow_incomplete_end=True)
    marks, timing = {}, []
    for lot in baseline["lots"]:
        ticker = lot["ticker"]
        price = float(lot["price"])
        observed_at = lot.get("entry_quote_at") or lot.get("entry_at") or lot["entry_date"]
        if not math.isfinite(price) or price <= 0:
            raise ValueError("INVALID_OR_NONTRADING_PRICE")
        marks[ticker] = price
        timing.append({"ticker": ticker, "price_source": price_source,
                       "price_observed_at": str(observed_at),
                       "chronology_valid": True})
    common_valid = None
    for history in histories.values():
        close = pd.to_numeric(history["Close"], errors="coerce")
        valid = {str(day) for day, value in close.items()
                 if str(day) <= publication_cutoff and
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
    observation_ready_at, observation_rows = market.forecast_observation_ready_at(
        baseline, policy)
    schedule = market.joint_calendar(observation_ready_at.date(),
                                     policy["calendar_verified_through"],
                                     policy, kinds)
    required_future = policy["max_review_sessions"] + 5
    future_after = max(observation_ready_at, pd.Timestamp(now))
    future = [str(day.date()) for day, session in schedule.iterrows()
              if session.market_open > future_after][:required_future]
    if len(future) < required_future:
        raise ValueError("INCOMPLETE_SESSION_CALENDAR")
    validation = validate(returns, baseline, marks, list(returns.index), policy)
    forecast = estimate(baseline, marks, returns, future, policy,
                        baseline["capital"], validation)
    candidate = forecast.get("next_review") or forecast.get("research_candidate") or future[0]
    # Current baselines produced by ``freeze`` contain the exact quantities,
    # cash and entry charges required for a fully costed liquidation estimate.
    # Legacy/read-only fixtures may not; do not manufacture those fields.
    metrics = None
    if ("cash" in baseline and all(
            "quantity" in lot and "entry_charges" in lot
            for lot in baseline["lots"])):
        metrics = evaluate(
            baseline, marks,
            baseline.get("fully_invested_date", baseline["entry_date"]),
            policy)
    # The sample ends at the chronology-safe close, but the 28-day horizon
    # begins when this planning preview is produced, not in the past.
    outlook = _planning_outlook(returns, baseline, now)
    return {"provisional": True, "publication_id": publication["publication_id"],
            "planning_estimate": True,
            "provisional_net_return": True,
            "policy_version": policy.get("policy_version"),
            "policy_digest": digest(policy),
            "history_coverage": coverage, "ack_epoch": ack_epoch,
            "as_of": history_as_of, "checked_at": pd.Timestamp(now).isoformat(),
            "observation_ready_at": observation_ready_at.isoformat(),
            "observation_rows": observation_rows,
            "assumption": assumption or "Historical planning estimate using only returns observable before publication. Frozen entry prices model the fully loaded target hurdle; no post-entry return is used.",
            "valuation_timing": {"mode": "PUBLICATION_TIME_PLANNING",
                                  "all_prices_synchronized": False,
                                  "rows": timing},
            "forecast": forecast,
            "provisional_outlook": outlook,
            "metrics": metrics,
            "decision": {"next_review": candidate, "reasons": [],
                         "target_crossed_securities": [],
                         "basis": "PROVISIONAL_ENTRY_TIME_FORECAST"}}


def _last_close_publication_preview(publication, policy, events, now, ack_epoch):
    """Build an immediate, read-only planning model from observable closes.

    This never creates or replaces a durable baseline. It exists only to avoid
    an empty public page while per-security opening entries are being captured.
    """
    kinds = {ticker: policy["instrument_kinds"][ticker]
             for ticker in publication["weights"]}
    now_stamp = pd.Timestamp(now)
    buffer = pd.Timedelta(minutes=policy["assessment_wait_after_close_minutes"])
    completed_dates = []
    for market_name in sorted({market.KIND_CALENDARS[kind]
                               for kind in kinds.values()}):
        schedule = market.calendar(
            now_stamp.date() - timedelta(days=14), now_stamp.date(),
            policy, market_name)
        completed = schedule[schedule.market_close + buffer <= now_stamp]
        if completed.empty:
            raise ValueError("INSUFFICIENT_COMMON_HISTORY")
        completed_dates.append(str(completed.index[-1].date()))
    cutoff = min(completed_dates)
    history_start = (
        pd.Timestamp(cutoff) - pd.DateOffset(years=int(policy["history_years"]))
    ).date().isoformat()
    tickers = list(publication["weights"])
    histories = market.fetch(
        tickers, history_start, cutoff, policy, allow_incomplete_end=True)
    common_valid = None
    for history in histories.values():
        close = pd.to_numeric(history["Close"], errors="coerce")
        valid = {str(day) for day, value in close.items()
                 if str(day) <= cutoff and math.isfinite(float(value))
                 and float(value) > 0}
        common_valid = valid if common_valid is None else common_valid & valid
    if not common_valid:
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    history_as_of = max(common_valid)
    prices = {
        ticker: float(pd.to_numeric(history["Close"], errors="coerce").loc[history_as_of])
        for ticker, history in histories.items()
    }
    weights = publication["weights"]
    capital = policy["capital_inr"] or math.ceil(max(
        (price + 60) / weights[ticker] for ticker, price in prices.items()
    ) / 100) * 100
    baseline = freeze(
        publication, weights, prices, history_as_of, capital, kinds, policy,
        now_stamp.isoformat())
    preview = _immediate_baseline_preview(
        publication, baseline, policy, events, now, ack_epoch,
        history_cutoff=history_as_of, histories=histories,
        price_source="LATEST_COMPLETED_CLOSE_PLANNING_BASELINE",
        assumption=(
            "Immediate planning model using the latest completed common close "
            "observable across represented markets. It is not an executed entry; "
            "verified opening-price evidence will supersede it."
        ),
    )
    preview["last_close_planning_baseline"] = True
    preview["assumed_entry_date"] = history_as_of
    # Read-only input for the page's indicative 15-minute valuation. This is
    # never persisted and is superseded by a durable opening-price baseline.
    preview["planning_baseline"] = baseline
    return preview


def historical_preview(publication, policy, events, now=None):
    from .instruments import (complete_policy, frozen_instrument_kinds,
                              require_supported_review)
    policy = complete_policy(
        policy, publication["weights"],
        frozen_kinds=frozen_instrument_kinds(events))
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
        forecast_ready = True
        observation_ready_at = None
        observation_rows = []
        try:
            observation_rows = market.require_forecast_observation_sessions(
                b, policy, now)
        except market.AwaitingMarketEntry as observation_wait:
            # Do not let the forecast-confidence window suppress performance
            # that can already be valued from a completed post-entry session.
            forecast_ready = False
            observation_ready_at = observation_wait.ready_at
            observation_rows = getattr(
                observation_wait, "observation_rows", None) or []
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
        p = build_assessment(
            b, histories, as_of, days, policy, events,
            publication["weights"], now, comparisons=False,
            forecast_ready=forecast_ready,
            observation_ready_at=observation_ready_at,
            observation_rows=observation_rows,
        )
        return {**p, "provisional": False, "as_of": as_of,
                "checked_at": now.isoformat(), "ack_epoch": ack_epoch,
                "policy_version": policy.get("policy_version"),
                "policy_digest": digest(policy),
                "publication_id": publication["publication_id"]}
    weights = publication["weights"]
    if set(weights) - set(policy["instrument_kinds"]):
        raise ValueError("INSTRUMENT_CLASSIFICATION_REQUIRED")
    return _last_close_publication_preview(
        publication, policy, events, now, ack_epoch)
