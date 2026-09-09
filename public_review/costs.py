"""Versioned conservative regular-resident NSE delivery cost envelope.

Not a contract note or tax return. No exemptions, loss offsets or fee tax deductions
are credited: this intentionally overestimates capital-gains tax. Instrument tax
classification MUST be supplied explicitly; suffixes are not classifications.
"""
from datetime import date
from decimal import Decimal, ROUND_CEILING
import math

SOURCES = ["https://groww.in/pricing", "https://zerodha.com/charges/",
           "https://www.amfiindia.com/investor/knowledge-center-info?zoneName=TaxRegimeForMutualFunds"]
TARIFF_VERSION = "groww-zerodha-nse-delivery-2026-09-09"
KINDS = {"equity", "equity_etf", "listed_non_equity_etf", "specified_debt_etf"}


def money(x):
    if not math.isfinite(float(x)) or x < 0:
        raise ValueError("Invalid non-negative amount")
    return float(Decimal(str(x)).quantize(Decimal(".01"), rounding=ROUND_CEILING))


def charges(gross, side, kind, policy):
    """One consolidated executed delivery order/ISIN/day, normal funded account.

    Excludes UPI mandate, NRI, MTF, call-and-trade and penalties. DP is charged once
    in this model, not per share. Caller must aggregate same-security same-day sells.
    """
    if side not in {"BUY", "SELL"} or kind not in KINDS:
        raise ValueError("Unsupported side/instrument classification")
    gross = money(gross)
    if not gross:
        return {"brokerage": 0., "dp": 0., "stt": 0., "stamp": 0., "exchange": 0.,
                "sebi": 0., "ipft": 0., "gst": 0., "slippage": 0., "total": 0.}
    # Groww normal plan vs Zerodha zero delivery brokerage; enforce value cap.
    brokerage = min(20., max(5., gross * .001), gross * .025)
    # Higher DP base: Groww male-depository tariff vs Zerodha 13 before GST.
    dp = max(13., 3.5 + (16.5 if gross >= 100 else 0.)) if side == "SELL" else 0.
    stt_rate = (.001 if kind == "equity" else
                .00001 if kind == "equity_etf" and side == "SELL" else 0.)
    parts = {"brokerage": money(brokerage), "dp": money(dp),
             "stt": float(math.ceil(gross * stt_rate)) if stt_rate else 0.,
             "stamp": money(gross * .00015) if side == "BUY" else 0.,
             "exchange": money(gross * .0000297), "sebi": money(gross * .000001),
             "ipft": money(gross * .000001)}
    parts["gst"] = money(.18 * sum(parts[k] for k in ("brokerage", "dp", "exchange", "sebi", "ipft")))
    parts["slippage"] = money(gross * policy["slippage_bps"] / 10000)
    parts["total"] = money(sum(parts.values()))
    return parts


def anniversary(d):
    try:
        return d.replace(year=d.year + 1)
    except ValueError:
        return d.replace(year=d.year + 1, day=28)


def tax_rate(kind, bought, sold, policy):
    if kind not in KINDS:
        raise ValueError("Unclassified instrument")
    bought, sold = date.fromisoformat(str(bought)[:10]), date.fromisoformat(str(sold)[:10])
    # No historical tax regime inference. This version supports new model lots only.
    if bought < date(2026, 4, 1) or sold < bought:
        raise ValueError("Unsupported tax date/regime")
    long_term = sold > anniversary(bought)
    if kind in {"equity", "equity_etf"}:
        base = .125 if long_term else .20
        surcharge = min(policy["surcharge_rate"], .15)
    elif kind == "listed_non_equity_etf" and long_term:
        base, surcharge = .125, min(policy["surcharge_rate"], .15)
    else:
        base, surcharge = policy["slab_rate"], policy["surcharge_rate"]
    return base * (1 + surcharge) * 1.04


def sell_value(quantity, price, lot, day, policy):
    gross = money(quantity * price)
    fees = charges(gross, "SELL", lot["kind"], policy)
    tax = money(max(0., quantity * (price - lot["price"])) *
                tax_rate(lot["kind"], lot["entry_date"], day, policy))
    return {"gross": gross, "fees": fees, "tax": tax,
            "net": round(gross - fees["total"] - tax, 2)}

