"""Known-cost funding floor, NOT an all-in US investment or tax model.

Tickertape Pro: https://www.tickertape.in/us-stocks/pricing
HDFC FX GST: https://www.hdfc.bank.in/remittance/fees-and-charges
Platform/gateway zero applies only to the user's supplied integrated funding quote.
"""
import math


def fx_gst(amount):
    """GST on the INR value of foreign currency exchanged (HDFC slab formula)."""
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Positive finite currency-exchange amount required")
    if amount <= 100_000:
        return min(180., max(45., amount * .0018))
    if amount <= 1_000_000:
        return min(990., 180. + (amount - 100_000.) * .0009)
    return min(10_800., 990. + (amount - 1_000_000.) * .00018)


def pro_brokerage(order_inr, usd_inr):
    if not all(math.isfinite(x) and x > 0 for x in (order_inr, usd_inr)):
        raise ValueError("Positive order and FX required")
    return min(order_inr * .0015, 25 * usd_inr) * 1.18


def known_cost_floor(maximum_drag=.005):
    """First INR100 grid amount meeting funding GST + uncapped buy brokerage.

    Ignoring the brokerage cap is conservative for this limited estimate.
    Unknown FX spread/statutory fees are NOT asserted to be zero.
    """
    if not math.isfinite(maximum_drag) or not 0 < maximum_drag < 1:
        raise ValueError("Invalid cost threshold")
    for amount in range(100, 1_000_001, 100):
        gst = fx_gst(amount)
        brokerage = (amount - gst) * .0015 * 1.18
        if (gst + brokerage) / amount <= maximum_drag:
            return {"amount_inr": amount, "fx_gst_inr": gst,
                    "buy_brokerage_with_gst_inr": brokerage,
                    "known_drag": (gst + brokerage) / amount,
                    "status": "KNOWN_COSTS_ONLY"}
    return None
