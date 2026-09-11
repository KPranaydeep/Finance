"""Deterministic frozen model lots, cash-flow returns and review decisions."""
from datetime import date, datetime, timezone
import hashlib
import json
import math
from .costs import charges, sell_value


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def xirr(flows):
    """Unique conventional-flow XIRR. Ambiguous multi-sign-change flows -> None."""
    merged = {}
    for day, amount in flows:
        d = date.fromisoformat(str(day)[:10])
        if not math.isfinite(amount):
            raise ValueError("Nonfinite cash flow")
        merged[d] = merged.get(d, 0.) + amount
    rows = [(d, v) for d, v in sorted(merged.items()) if abs(v) > 1e-10]
    if len(rows) < 2 or rows[0][1] >= 0 or rows[-1][1] <= 0:
        return None
    signs = [v > 0 for _, v in rows]
    if sum(a != b for a, b in zip(signs, signs[1:])) != 1:
        return None
    t = [(d - rows[0][0]).days / 365 for d, _ in rows]
    # Solve in log(1+r), scaling exponents for numerical stability.
    def value(y):
        e = [-y * a for a in t]
        m = max(e)
        return sum(v * math.exp(z - m) for (_, v), z in zip(rows, e))
    low, high = -30., 30.
    if value(low) * value(high) >= 0:
        return None
    for _ in range(160):
        mid = (low + high) / 2
        if value(mid) > 0:
            low = mid
        else:
            high = mid
    return math.expm1((low + high) / 2)


def freeze(publication, weights, prices, entry_date, capital, kinds, policy, captured_at):
    if not math.isfinite(capital) or capital < 1:
        raise ValueError("Capital must be finite and at least INR 1")
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()) or sum(weights.values()) > 1.000001:
        raise ValueError("Invalid target weights")
    lots, spent = [], 0.
    foreign_weight = sum(weight for ticker, weight in weights.items()
                         if kinds[ticker] == "foreign_us_listing")
    foreign_funding_gst = 0.
    if foreign_weight:
        from public_us_funding import fx_gst
        foreign_funding_gst = fx_gst(capital * foreign_weight)
    for ticker, weight in sorted(weights.items()):
        p = float(prices[ticker])
        if not math.isfinite(p) or p <= 0:
            raise ValueError("Missing/invalid entry price")
        kind = kinds[ticker]  # Explicit owner-approved classification, never guessed.
        budget = capital * weight
        allocated_fx_gst = (foreign_funding_gst * weight / foreign_weight
                            if kind == "foreign_us_listing" else 0.)
        q = int(budget // p)
        while q and q * p + charges(q * p, "BUY", kind, policy)["total"] + allocated_fx_gst > budget:
            q -= 1
        if not q:
            continue
        fee = charges(q * p, "BUY", kind, policy)
        if allocated_fx_gst:
            allocated = round(allocated_fx_gst, 2)
            fee["fx_gst"] = allocated
            fee["total"] = round(fee["total"] + allocated, 2)
        lot = {"ticker": ticker, "quantity": q, "price": p, "kind": kind,
               "entry_date": entry_date, "entry_charges": fee}
        lots.append(lot)
        spent += q * p + fee["total"]
    if not lots:
        raise ValueError("Capital cannot fund any whole-share position after entry costs")
    payload = {"publication_id": publication["publication_id"], "basket_id": publication["basket_id"],
               "portfolio_version": int(publication["portfolio_version"]),
               "published_at": str(publication["published_at"]), "entry_date": entry_date,
               "captured_at": captured_at, "capital": capital, "cash": round(capital - spent, 2),
               "weights": weights, "lots": lots, "entry_policy": policy,
               "foreign_funding_fx_gst": round(sum(
                   lot["entry_charges"].get("fx_gst", 0.) for lot in lots), 2),
               "basis": "RETROSPECTIVE_PUBLICATION_MODEL_NOT_ACTUAL_TRADES",
               "units": "Yahoo split-normalized at capture; subsequent corporate actions require review"}
    payload["baseline_id"] = digest(payload)
    return payload


def evaluate(baseline, prices, day, policy, dividend_flows=None):
    dividends = dividend_flows or []  # Net, dated, retained as basket cash.
    owned = {r["ticker"] for r in baseline["lots"]}
    for flow in dividends:
        if (flow["ticker"] not in owned or not baseline["entry_date"] < flow["date"] <= day or
                not math.isfinite(flow["net"]) or flow["net"] < 0):
            raise ValueError("Invalid dated model distribution")
    rows = []
    for lot in baseline["lots"]:
        price = float(prices[lot["ticker"]])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid current price")
        sale = sell_value(lot["quantity"], price, lot, day, policy)
        outlay = lot["quantity"] * lot["price"] + lot["entry_charges"]["total"]
        div = sum(x["net"] for x in dividends if x["ticker"] == lot["ticker"])
        flows = [(baseline["entry_date"], -outlay)] + [
            (x["date"], x["net"]) for x in dividends if x["ticker"] == lot["ticker"]]
        flows.append((day, sale["net"]))
        rows.append({"ticker": lot["ticker"], "shares": lot["quantity"], "price": price,
                     "outlay": round(outlay, 2), "net_profit": round(sale["net"] + div - outlay, 2),
                     "xirr": xirr(flows), **sale})
    foreign = [row for row, lot in zip(rows, baseline["lots"])
               if lot["kind"] == "foreign_us_listing"]
    if foreign:
        from public_us_funding import fx_gst
        conversion_gst = fx_gst(sum(row["gross"] for row in foreign))
        foreign_gross = sum(row["gross"] for row in foreign)
        for row in foreign:
            share = conversion_gst * row["gross"] / foreign_gross
            row["fees"]["fx_gst"] = round(share, 2)
            row["fees"]["total"] = round(row["fees"]["total"] + share, 2)
            row["net"] = round(row["net"] - share, 2)
            row["net_profit"] = round(row["net_profit"] - share, 2)
            # Recompute the security XIRR after its allocated conversion GST.
            lot = next(lot for lot in baseline["lots"] if lot["ticker"] == row["ticker"])
            security_dividends = [x for x in dividends if x["ticker"] == row["ticker"]]
            row["xirr"] = xirr([(baseline["entry_date"], -row["outlay"])] +
                               [(x["date"], x["net"]) for x in security_dividends] +
                               [(day, row["net"])])
    net = baseline["cash"] + sum(r["net"] for r in rows) + sum(x["net"] for x in dividends)
    gross = baseline["cash"] + sum(r["gross"] for r in rows) + sum(x["net"] for x in dividends)
    return {"date": day, "rows": rows, "net_proceeds": round(net, 2), "gross_value": gross,
            "net_profit": round(net - baseline["capital"], 2),
            "net_total_return": net / baseline["capital"] - 1,
            "xirr": xirr([(baseline["entry_date"], -baseline["capital"]), (day, net)]),
            "days_held": (date.fromisoformat(day) - date.fromisoformat(baseline["entry_date"])).days,
            "exit_costs": sum(r["fees"]["total"] for r in rows), "estimated_tax": sum(r["tax"] for r in rows)}


def decision(metrics, baseline, latest_weights, policy, peak, forecast=None, promised=None, benefit=None):
    """Reviews, never orders. Risk and profit events can coexist."""
    reasons = []
    weights = {r["ticker"]: r["gross"] / metrics["gross_value"] for r in metrics["rows"]}
    drift = max([abs(weights.get(t, 0) - latest_weights.get(t, 0)) for t in set(weights) | set(latest_weights)] or [0.])
    drawdown = metrics["net_proceeds"] / max(peak, metrics["net_proceeds"]) - 1
    if drawdown <= -policy["drawdown_limit"] or max(weights.values(), default=0) > policy["concentration_limit"]:
        reasons.append("RISK_REVIEW")
    if metrics["xirr"] is not None and metrics["xirr"] >= policy["target_xirr"]:
        reasons.append("PROFIT_TAKING_REVIEW")
    crossed = [r["ticker"] for r in metrics["rows"]
               if r.get("xirr") is not None and r["xirr"] >= policy["target_xirr"]]
    if crossed:
        reasons.append("SECURITY_TARGET_REVIEW")
    rebalance = {"status": "NO_MATERIAL_DRIFT", "drift": drift}
    if drift >= policy["drift_limit"]:
        # benefit must be a separately validated comparable annual net estimate,
        # not forecast median extrapolated to one year.
        qualifies = (benefit is not None and benefit.get("validated") is True and
                     benefit.get("net_annual_improvement", -1) >= policy["min_annual_improvement"])
        rebalance["status"] = "BENEFIT_GATE_PASSED" if qualifies else "BENEFIT_NOT_ESTABLISHED"
        if qualifies:
            reasons.append("REBALANCE_REVIEW")
    candidate = (forecast or {}).get("next_review")
    next_review = min(x for x in (candidate, promised) if x) if candidate or promised else None
    if next_review and next_review <= metrics["date"]:
        reasons.append("SCHEDULED_REVIEW_DUE")
    if reasons:
        next_review = min(next_review, metrics["date"]) if next_review else metrics["date"]
    return {"status": reasons[0] if reasons else "NO_TRIGGER_DETECTED", "reasons": reasons,
            "target_crossed_securities": crossed,
            "next_review": next_review, "drawdown": drawdown, "rebalance": rebalance,
            "note": "No trigger is not a safety guarantee. Review signals do not submit trades."}


def compare_exits(baseline, prices, day, policy, metrics):
    """Cost-first mixed integer whole-share sale plans with exact fee reconciliation."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    import numpy as np
    rows = baseline["lots"]
    # Candidate lots compress very large quantities; disclose bounded optimality.
    candidates = []
    for i, lot in enumerate(rows):
        q = lot["quantity"]
        sizes = range(1, q + 1) if q <= 500 else sorted(set([1, q] + [max(1, round(q * k / 500)) for k in range(1, 500)]))
        for n in sizes:
            s = sell_value(n, prices[lot["ticker"]], lot, day, policy)
            if s["net"] > 0:
                candidates.append((i, n, s))
    output = [{"option": "No trade", "cash_raised": 0., "fees": 0., "tax": 0., "orders": []},
              {"option": "Full exit", "cash_raised": metrics["net_proceeds"],
               "fees": metrics["exit_costs"], "tax": metrics["estimated_tax"],
               "orders": [{"ticker": r["ticker"], "shares": r["shares"]} for r in metrics["rows"]]}]
    cash = metrics["net_proceeds"] - sum(r["net"] for r in metrics["rows"])
    for name, target in [("Recover initial capital", baseline["capital"]),
                         ("Withdraw net profit", max(0., metrics["net_profit"]))]:
        if target <= 0:
            output.append({"option": name, "status": "NO_POSITIVE_PROFIT"}); continue
        if target > metrics["net_proceeds"]:
            output.append({"option": name, "status": "INSUFFICIENT_NET_PROCEEDS"}); continue
        if cash >= target:
            output.append({"option": name, "cash_raised": cash, "fees": 0., "tax": 0., "orders": []}); continue
        n = len(candidates)
        a = np.zeros((len(rows) + 1, n))
        c = np.empty(n)
        for j, (i, _, s) in enumerate(candidates):
            a[i, j] = 1
            a[-1, j] = s["net"]
            c[j] = s["fees"]["total"] + s["tax"]
        lower = np.array([0.] * len(rows) + [target - cash])
        upper = np.array([1.] * len(rows) + [np.inf])
        solved = milp(c, integrality=np.ones(n), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(a, lower, upper),
                      options={"time_limit": 8., "mip_rel_gap": 0.})
        if solved.status != 0:
            output.append({"option": name, "status": "OPTIMUM_NOT_PROVEN"}); continue
        chosen = [item for item, v in zip(candidates, solved.x) if v > .5]
        raised = cash + sum(s["net"] for _, _, s in chosen)
        if raised + .001 < target:
            raise ValueError("Sale reconciliation failed")
        output.append({"option": name, "cash_raised": round(raised, 2),
                       "fees": sum(s["fees"]["total"] for _, _, s in chosen),
                       "tax": sum(s["tax"] for _, _, s in chosen),
                       "orders": [{"ticker": rows[i]["ticker"], "shares": q} for i, q, _ in chosen],
                       "status": "MINIMUM_ESTIMATED_FEE_PLUS_TAX_WITHIN_CANDIDATE_GRID"})
    return output
