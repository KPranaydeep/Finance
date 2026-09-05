"""Pure, deterministic analytics for the public portfolio trust layer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

CALCULATION_VERSION = "public-trust-v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _dated_flows(flows: Iterable[tuple[date | datetime, float]]) -> list[tuple[date, float]]:
    grouped: dict[date, float] = {}
    for when, amount in flows:
        day = when.date() if isinstance(when, datetime) else when
        number = float(amount)
        if not isinstance(day, date) or not math.isfinite(number):
            raise ValueError("Cash flows require valid dates and finite amounts")
        grouped[day] = grouped.get(day, 0.0) + number
    return sorted((day, amount) for day, amount in grouped.items() if abs(amount) > 1e-12)


def xnpv(rate: float, flows: Sequence[tuple[date, float]]) -> float:
    if rate <= -1:
        return math.inf
    origin = flows[0][0]
    return sum(amount / ((1.0 + rate) ** ((day - origin).days / 365.0)) for day, amount in flows)


def xirr(flows: Iterable[tuple[date | datetime, float]]) -> float | None:
    """Return the unique economically meaningful XIRR, or None when not established."""
    dated = _dated_flows(flows)
    if len(dated) < 2 or dated[0][0] == dated[-1][0]:
        return None
    amounts = [amount for _, amount in dated]
    if not any(x < 0 for x in amounts) or not any(x > 0 for x in amounts):
        return None
    grid = np.concatenate((np.linspace(-0.999999, -0.5, 100), np.linspace(-0.5, 10, 400), np.logspace(1, 6, 150)))
    roots: list[float] = []
    previous_rate, previous_value = float(grid[0]), xnpv(float(grid[0]), dated)
    if abs(previous_value) < 1e-10:
        roots.append(previous_rate)
    for candidate in grid[1:]:
        rate, value = float(candidate), xnpv(float(candidate), dated)
        if abs(value) < 1e-10 and not any(abs(rate - existing) < 1e-8 for existing in roots):
            roots.append(rate)
        if math.isfinite(value) and math.isfinite(previous_value) and value * previous_value < 0:
            root = float(brentq(lambda r: xnpv(r, dated), previous_rate, rate, maxiter=250))
            if not any(abs(root - existing) < 1e-8 for existing in roots):
                roots.append(root)
        previous_rate, previous_value = rate, value
    return roots[0] if len(roots) == 1 else None


def nav_frame(rows: Sequence[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    frame["nav_date"] = pd.to_datetime(frame["nav_date"], errors="coerce")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    return frame.dropna(subset=["nav_date", "nav"]).query("nav > 0").sort_values("nav_date").drop_duplicates("nav_date", keep="last")


def performance_metrics(rows: Sequence[dict]) -> dict:
    frame = nav_frame(rows)
    if frame.empty:
        return {}
    returns = frame["nav"].pct_change().dropna()
    elapsed = (frame.iloc[-1]["nav_date"] - frame.iloc[0]["nav_date"]).days
    total = float(frame.iloc[-1]["nav"] / frame.iloc[0]["nav"] - 1)
    drawdowns = frame["nav"] / frame["nav"].cummax() - 1
    downside = returns[returns < 0]
    return {
        "start_date": frame.iloc[0]["nav_date"].date(), "end_date": frame.iloc[-1]["nav_date"].date(),
        "observations": len(frame), "total_return": total,
        "annualized_return": float((1 + total) ** (365.25 / elapsed) - 1) if elapsed > 0 and total > -1 else None,
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) >= 2 else None,
        "maximum_drawdown": float(drawdowns.min()), "current_drawdown": float(drawdowns.iloc[-1]),
        "downside_deviation": float(downside.std(ddof=1) * np.sqrt(252)) if len(downside) >= 2 else None,
        "positive_day_percentage": float((returns > 0).mean()) if len(returns) else None,
        "best_day": float(returns.max()) if len(returns) else None,
        "worst_day": float(returns.min()) if len(returns) else None,
    }


def select_horizon(rows: Sequence[dict], days: int | None) -> list[dict]:
    frame = nav_frame(rows)
    if frame.empty or days is None:
        return frame.to_dict("records")
    cutoff = frame.iloc[-1]["nav_date"] - pd.Timedelta(days=days)
    selected = frame[frame["nav_date"] >= cutoff]
    return selected.to_dict("records") if len(selected) >= 2 else []


def versioned_model_nav(prices: pd.DataFrame, publications: Sequence[dict], *, initial_nav: float = 100.0) -> pd.DataFrame:
    """Chain observed asset returns using the portfolio version effective each day."""
    if prices.empty or not publications:
        return pd.DataFrame(columns=["nav_date", "nav", "daily_return", "drawdown", "publication_id"])
    frame=prices.copy().sort_index().apply(pd.to_numeric,errors="coerce").ffill()
    returns=frame.pct_change()
    versions=sorted(publications,key=lambda item:pd.Timestamp(item["as_of"]))
    nav=float(initial_nav); peak=nav; rows=[]
    for timestamp,row in returns.iterrows():
        eligible=[item for item in versions if pd.Timestamp(item["as_of"]).tz_localize(None) <= pd.Timestamp(timestamp).tz_localize(None)]
        if not eligible: continue
        current=eligible[-1]; weights=current["weights"]
        values=[]
        for ticker,weight in weights.items():
            value=row.get(ticker)
            if pd.notna(value): values.append((float(weight),float(value)))
        invested=sum(weight for weight,_ in values)
        if invested <= 0: continue
        daily=sum(weight*value for weight,value in values)/invested
        if not math.isfinite(daily): continue
        nav*=1+daily; peak=max(peak,nav)
        rows.append({"nav_date":pd.Timestamp(timestamp).date(),"nav":nav,"daily_return":daily,
                     "drawdown":nav/peak-1,"publication_id":current["publication_id"]})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ForecastResult:
    horizon_days: int
    method: str
    sample_start: str
    sample_end: str
    observation_count: int
    median_return: float
    lower_50: float
    upper_50: float
    lower_90: float
    upper_90: float
    probability_positive: float
    probability_negative: float
    probability_loss_gt_threshold: float
    loss_threshold: float


def bootstrap_outlook(
    daily_returns: Sequence[float], sample_dates: Sequence[date | str], *, horizon_days: int = 14,
    simulations: int = 10_000, seed: int = 0, loss_threshold: float = -0.05,
) -> ForecastResult | None:
    values = np.asarray(daily_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 60 or len(sample_dates) != len(values) or horizon_days <= 0:
        return None
    lag1 = float(pd.Series(values).autocorr(lag=1)) if len(values) > 2 else 0.0
    block = min(5, horizon_days) if math.isfinite(lag1) and abs(lag1) >= 0.10 else 1
    rng = np.random.default_rng(seed)
    paths = np.empty((simulations, horizon_days))
    for row in range(simulations):
        output: list[float] = []
        while len(output) < horizon_days:
            start = int(rng.integers(0, len(values) - block + 1))
            output.extend(values[start : start + block])
        paths[row] = output[:horizon_days]
    outcomes = np.prod(1 + paths, axis=1) - 1
    q05, q25, q50, q75, q95 = np.quantile(outcomes, [0.05, 0.25, 0.5, 0.75, 0.95])
    dates = pd.to_datetime(list(sample_dates))
    return ForecastResult(
        horizon_days, "moving_block_bootstrap" if block > 1 else "ordinary_bootstrap",
        dates.min().date().isoformat(), dates.max().date().isoformat(), len(values),
        float(q50), float(q25), float(q75), float(q05), float(q95),
        float(np.mean(outcomes > 0)), float(np.mean(outcomes < 0)),
        float(np.mean(outcomes < loss_threshold)), loss_threshold,
    )


def forecast_payload(result: ForecastResult) -> dict:
    return {"calculation_version": CALCULATION_VERSION, **asdict(result)}
