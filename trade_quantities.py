"""Planning quantities inferred from existing positions, without broker API claims."""

from decimal import Decimal, ROUND_DOWN
import math


def quantity_step(held_quantity, currency):
    """Whole units by default; preserve demonstrated non-INR fractional precision.

    A fractional Indian holding can arise from corporate actions and does not
    establish fractional order support. Other listings use the precision already
    present in the holding (up to eight decimal places), not a guessed broker lot.
    """
    held = float(held_quantity)
    if not math.isfinite(held) or held < 0:
        raise ValueError("Holding quantity must be finite and non-negative")
    normalized = Decimal(str(round(held, 8))).normalize()
    code = str(currency).strip().upper()
    if code == "INR" or len(code) != 3 or not code.isalpha() or code in {"NAN"} or normalized == normalized.to_integral_value():
        return 1.0
    return float(Decimal(1).scaleb(normalized.as_tuple().exponent))


def plan_quantity(gap_quantity, held_quantity, currency, action, days_to_flip=1):
    """Round buys down to the inferred step; size exits without withholding a unit."""
    gap, held, days = map(float, (gap_quantity, held_quantity, days_to_flip))
    if not all(math.isfinite(value) for value in (gap, held, days)):
        raise ValueError("Trade sizing inputs must be finite")
    if gap < 0 or held < 0 or days <= 0 or action not in ("Buy", "Sell"):
        raise ValueError("Invalid trade sizing inputs")
    step = Decimal(str(quantity_step(held, currency)))
    raw = min(gap, held) if action == "Sell" else gap / days
    value = Decimal(str(round(raw, 8)))
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return min(float(units * step), held) if action == "Sell" else float(units * step)


def format_quantity(value):
    """Keep fractional quantities visible in the plan and holding summary."""
    return f"{float(value):,.8f}".rstrip("0").rstrip(".")
