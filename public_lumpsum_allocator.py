"""Deterministic whole-share allocation for the published public target."""

from __future__ import annotations

from typing import Iterable

import numpy as np


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
