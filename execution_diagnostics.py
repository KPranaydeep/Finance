"""Explain recorded execution failures without market requests or a new solve."""
import math


SHOCK_PREFIX = "Single-security shock budget exceeded: "


def execution_summary_metric(report, executable_count=0):
    """Count released orders only after validation; failures are a status, not zero activity."""
    if not report:
        return "Plan status", "Needs validation", "Saved plan has no execution check"
    if report.get("passed") is not True:
        breaches = {str(reason)[len(SHOCK_PREFIX):] for reason in report.get("violations") or []
                    if str(reason).startswith(SHOCK_PREFIX)}
        note = (f"{len(breaches)} {'holding' if len(breaches) == 1 else 'holdings'} over cap · draft withheld"
                if breaches else "Required checks failed · draft withheld")
        return "Plan status", "Blocked", note
    return "Executable trades", executable_count, (
        "Checks passed · review orders before execution" if executable_count else
        "Checks passed · no executable orders")


def execution_position_rows(report, allocation_policy=None):
    """Use exact checked caps, or clearly qualified saved ceilings for old reports.

    The saved cap may be higher than a subsequently tightened cap. Never invent
    the newer threshold or recompute history during a display-only migration.
    An invalidated report whose reasons were replaced must not show stale rows.
    """
    tickers = {str(reason)[len(SHOCK_PREFIX):] for reason in report.get("violations") or []
               if str(reason).startswith(SHOCK_PREFIX)}
    exact = {row["ticker"]: row for row in (report.get("allocation_policy") or {}).get("position_breaches") or []}
    saved_caps = ((allocation_policy or {}).get("limits") or {}).get("position_caps") or {}
    rows = []
    for ticker in sorted(tickers):
        row = exact.get(ticker)
        weight = row.get("weight") if row else (report.get("final_weights") or {}).get(ticker)
        cap = row.get("cap") if row else saved_caps.get(ticker)
        try:
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                weight = None
        except (ValueError, TypeError):
            weight = None
        try:
            cap = float(cap)
            if not math.isfinite(cap) or not 0 <= cap <= 1:
                cap = None
        except (ValueError, TypeError):
            cap = None
        # If an old saved ceiling does not explain the recorded breach, the
        # tighter checked cap was not retained. Show unknown, not a false pass.
        if not row and weight is not None and cap is not None and weight <= cap + 1e-7:
            cap = None
        rows.append({"ticker": ticker, "weight": weight, "cap": cap,
                     "excess": weight-cap if weight is not None and cap is not None else None,
                     "cap_basis": "checked" if row else "saved ceiling" if cap is not None else "unavailable"})
    return sorted(rows, key=lambda row: (-(row["excess"] if row["excess"] is not None else -1), row["ticker"]))
