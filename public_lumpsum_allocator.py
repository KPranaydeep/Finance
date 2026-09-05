"""Deterministic whole-share allocation for the published public target."""

from __future__ import annotations

from typing import Iterable
import math

import numpy as np


def estimate_minimum_entry_capital(
    constituents: Iterable[dict],
    prices: dict[str, dict | float],
    *,
    fixed_cost_per_order: float = 20.0,
    statutory_cost_rate: float = 0.0012,
    slippage_rate: float = 0.0010,
    maximum_execution_drag: float = 0.0050,
    maximum_tracking_error_pp: float = 2.0,
    maximum_residual_ratio: float = 0.01,
) -> dict:
    """Estimate capital where the complete target is practical as whole shares."""
    target = [{"ticker":str(row["ticker"]).strip().upper(),"target_weight":float(row["target_weight"])} for row in constituents]
    usable=[]
    for row in target:
        raw=prices.get(row["ticker"])
        price=float(raw.get("price")) if isinstance(raw,dict) and raw.get("price") else (float(raw) if raw is not None else np.nan)
        if np.isfinite(price) and price>0 and row["target_weight"]>0:
            usable.append({**row,"price":price})
    if len(usable)!=len(target) or not usable:
        raise ValueError("A current price is required for every target security")
    weight_total=sum(row["target_weight"] for row in usable)
    for row in usable: row["target_weight"]/=weight_total
    price_values=np.array([row["price"] for row in usable],dtype=float)
    count=len(usable); mean_price=float(price_values.mean()); std_price=float(price_values.std(ddof=0))
    one_share_floor=float(price_values.sum())
    weighted_affordability_floor=max(row["price"]/row["target_weight"] for row in usable)
    dispersion_floor=count*(mean_price+std_price)
    variable_drag=statutory_cost_rate+slippage_rate
    if maximum_execution_drag<=variable_drag:
        raise ValueError("Maximum execution drag must exceed variable cost and slippage")
    cost_floor=count*fixed_cost_per_order/(maximum_execution_drag-variable_drag)
    candidate=max(one_share_floor,weighted_affordability_floor,dispersion_floor,cost_floor)
    candidate=math.ceil(candidate/1000.0)*1000.0

    price_map={row["ticker"]:row["price"] for row in usable}
    plan=None; tracking_error=float("inf")
    for _ in range(240):
        plan=allocate_public_lumpsum(usable,price_map,candidate)
        actual={row["ticker"]:row["estimated_value"]/candidate for row in plan["orders"]}
        tracking_error=100*math.sqrt(sum((actual.get(row["ticker"],0)-row["target_weight"])**2 for row in usable)/count)
        estimated_drag=(plan["invested_inr"]*variable_drag+count*fixed_cost_per_order)/candidate
        if (plan["coverage"]==count and plan["residual_cash_inr"]/candidate<=maximum_residual_ratio
                and tracking_error<=maximum_tracking_error_pp and estimated_drag<=maximum_execution_drag):
            break
        candidate=math.ceil(candidate*1.025/1000.0)*1000.0
    assert plan is not None
    estimated_cost=plan["invested_inr"]*statutory_cost_rate+count*fixed_cost_per_order
    estimated_slippage=plan["invested_inr"]*slippage_rate
    return {
        "minimum_capital_inr":candidate,"constituent_count":count,"mean_price":mean_price,
        "price_standard_deviation":std_price,"minimum_price":float(price_values.min()),
        "maximum_price":float(price_values.max()),"one_share_floor":one_share_floor,
        "weighted_affordability_floor":weighted_affordability_floor,"dispersion_floor":dispersion_floor,
        "cost_floor":cost_floor,"estimated_cost_inr":estimated_cost,"estimated_slippage_inr":estimated_slippage,
        "estimated_execution_drag":(estimated_cost+estimated_slippage)/candidate,
        "tracking_error_pp":tracking_error,"residual_ratio":plan["residual_cash_inr"]/candidate,
        "coverage":plan["coverage"],"assumptions":{
            "fixed_cost_per_order":fixed_cost_per_order,"statutory_cost_rate":statutory_cost_rate,
            "slippage_rate":slippage_rate,"maximum_execution_drag":maximum_execution_drag,
            "maximum_tracking_error_pp":maximum_tracking_error_pp,"maximum_residual_ratio":maximum_residual_ratio,
        },
    }


def allocate_public_lumpsum(
    constituents: Iterable[dict],
    prices: dict[str, dict | float],
    amount_inr: float,
    *,
    max_weight_per_asset: float = 0.10,
) -> dict:
    """Allocate cash with the production cap, falling back to a small-capital starter plan."""
    amount = float(amount_inr)
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("Investment amount must be positive")

    rows = sorted(
        ({"ticker": str(row["ticker"]).strip().upper(), "weight": float(row["target_weight"])}
         for row in constituents),
        key=lambda row: row["ticker"],
    )
    if not rows or any(not row["ticker"] or row["weight"] < 0 for row in rows):
        raise ValueError("Published target is invalid")
    weight_total = sum(row["weight"] for row in rows)
    if weight_total <= 0:
        raise ValueError("Published target weights must be positive")
    for row in rows:
        row["weight"] /= weight_total
        raw_price = prices.get(row["ticker"])
        row["price"] = float(raw_price.get("price")) if isinstance(raw_price, dict) and raw_price.get("price") else (
            float(raw_price) if raw_price is not None else np.nan
        )
    available = [row for row in rows if np.isfinite(row["price"]) and row["price"] > 0]
    if not available:
        raise ValueError("No target security has an available price")

    weights = np.array([row["weight"] for row in available], dtype=float)
    weights = weights / weights.sum()
    price_values = np.array([row["price"] for row in available], dtype=float)
    target_values = amount * weights
    quantities = np.floor(target_values / price_values).astype(int)
    invested = quantities * price_values
    cap_values = np.maximum(target_values, amount * float(max_weight_per_asset))

    def fill_with_cap() -> None:
        nonlocal quantities, invested
        while True:
            cash = amount - float(invested.sum())
            candidates = np.flatnonzero((price_values <= cash + 1e-9) & (invested + price_values <= cap_values + 1e-9))
            if not len(candidates):
                return
            improvement = np.abs(invested[candidates] - target_values[candidates]) - np.abs(
                invested[candidates] + price_values[candidates] - target_values[candidates]
            )
            best = candidates[int(np.argmax(improvement))]
            quantities[best] += 1
            invested[best] += price_values[best]

    fill_with_cap()
    mode = "TARGET_WEIGHT"

    # Whole-share caps can strand most of a small investment. In that case,
    # construct a diversified starter allocation by minimizing target error plus
    # a modest residual-cash penalty. This remains deterministic and bounded.
    if float(invested.sum()) < amount * 0.70:
        mode = "STARTER"
        quantities = np.zeros(len(available), dtype=int)
        invested = np.zeros(len(available), dtype=float)

        def objective(values: np.ndarray) -> float:
            actual = values / amount
            residual = max(0.0, amount - float(values.sum())) / amount
            return float(np.square(actual - weights).sum() + 0.15 * residual * residual)

        while True:
            cash = amount - float(invested.sum())
            candidates = np.flatnonzero(price_values <= cash + 1e-9)
            if not len(candidates):
                break
            current_score = objective(invested)
            scored = []
            for index in candidates:
                trial = invested.copy(); trial[index] += price_values[index]
                scored.append((objective(trial), available[index]["ticker"], int(index)))
            best_score, _, best = min(scored)
            if best_score >= current_score - 1e-12:
                break
            quantities[best] += 1
            invested[best] += price_values[best]

    orders = []
    for index, row in enumerate(available):
        if quantities[index] <= 0:
            continue
        orders.append({
            "ticker": row["ticker"],
            "target_weight": float(weights[index]),
            "quantity": int(quantities[index]),
            "planning_price": float(price_values[index]),
            "estimated_value": float(invested[index]),
            "post_trade_weight": float(invested[index] / amount),
        })
    orders.sort(key=lambda row: (-row["estimated_value"], row["ticker"]))
    invested_total = float(sum(row["estimated_value"] for row in orders))
    return {
        "mode": mode,
        "amount_inr": amount,
        "invested_inr": invested_total,
        "residual_cash_inr": max(0.0, amount - invested_total),
        "coverage": len(orders),
        "orders": orders,
        "missing_prices": [row["ticker"] for row in rows if not np.isfinite(row["price"]) or row["price"] <= 0],
    }
