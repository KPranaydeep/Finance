"""One recorded ₹100 outcome for evidence cards and their navigation."""

import math


def value_of_100(outcome):
    value = outcome.get("value_100")
    try:
        if value is None:
            result = outcome.get("ticker_return", outcome.get("return"))
            if result is None or isinstance(result, bool):
                return None
            value = 100 * (1 + float(result))
        if isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def lifecycle_key(row):
    return (str(row["ticker"]), str(row.get("entry_date") or "")[:10],
            str(row.get("exit_date") or "")[:10])


def remember_card_outcome(saved, publication_id, row, metrics):
    """Retain exactly the card's value; return whether navigation needs updating."""
    value = value_of_100(metrics)
    if value is None or not metrics.get("as_of"):
        return False
    key = (str(publication_id), *lifecycle_key(row))
    outcome = {
        "ticker": row["ticker"], "entry_date": row["entry_date"],
        "exit_date": row.get("exit_date"), "value_100": value,
        "return": value / 100 - 1, "data_through": str(metrics["as_of"]),
    }
    if saved.get(key) == outcome:
        return False
    saved[key] = outcome
    while len(saved) > 256:
        del saved[next(iter(saved))]
    return True


def reuse_card_outcomes(rows, summaries, saved, publication_id):
    """A rendered card is authoritative for its matching publication/lifecycle."""
    merged = {lifecycle_key(row): row for row in summaries}
    for row in rows:
        lifecycle = lifecycle_key(row)
        outcome = saved.get((str(publication_id), *lifecycle))
        if outcome is not None:
            merged[lifecycle] = outcome
    return list(merged.values())
