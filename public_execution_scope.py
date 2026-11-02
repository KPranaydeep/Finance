"""Private execution scopes derived from an unchanged public allocation."""

import math
import re


EXECUTION_MODES = ("INR only", "Overseas only", "Global")


def execution_currency(ticker, prices):
    symbol = str(ticker).strip().upper()
    currency = str(prices.get(symbol, {}).get("source_currency") or "").upper()
    if currency:
        return currency
    if symbol.endswith((".NS", ".BO")):
        return "INR"
    # Match the public valuation engine's supported direct-US symbol contract.
    if re.fullmatch(r"[A-Z][A-Z0-9-]{0,14}", symbol):
        return "USD"
    return None


def select_execution_target(constituents, prices, mode):
    if mode not in EXECUTION_MODES:
        raise ValueError("Unknown execution mode")
    selected = []
    for row in constituents:
        weight = float(row["target_weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid target weight")
        currency = execution_currency(row["ticker"], prices)
        keep = (mode == "Global" or (currency == "INR" if mode == "INR only"
                                    else currency is not None and currency != "INR"))
        if keep and weight > 0:
            selected.append(dict(row))
    if mode != "Global" and selected:
        total = sum(float(row["target_weight"]) for row in selected)
        for row in selected:
            row["target_weight"] = float(row["target_weight"]) / total
    return selected
