from datetime import date
import json
import numpy as np
import pandas as pd
import pytest
from election_watch import seed_snapshot, monthly_snapshot, upcoming_terms, parse_eci_terms, market_context


class Response:
    def __init__(self, html): self.text = html
    def raise_for_status(self): pass


def test_monthly_failure_retains_verified_snapshot_and_does_not_retry(tmp_path):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("blocked")
    path = tmp_path / "snapshot.json"
    first = monthly_snapshot(path, date(2026,10,30), fetch=fail)
    second = monthly_snapshot(path, date(2026,10,31), fetch=fail)
    assert first["rows"] == second["rows"] == seed_snapshot()["rows"]
    assert first["verified_at"] == "2026-10-30"
    assert len(calls) == 1
    monthly_snapshot(path, date(2026,11,1), fetch=fail)
    assert len(calls) == 2


def test_valid_eci_schema_updates_only_after_success(tmp_path):
    html = pd.DataFrame({"HOUSE/STATE":[f"State {x}" for x in range(31)], "TO":["14.03.2027"]*31,
                         "LOKSABHA SEAT":[2]*31, "ASSEMBLY SEAT":[40]*31, "RAJYA SABHA SEAT":[1]*31}).to_html(index=False)
    result = monthly_snapshot(tmp_path / "snapshot.json", date(2026,11,1), fetch=lambda *a,**k: Response(html))
    assert result["verified_at"] == "2026-11-01"
    assert result["rows"][0]["Term ends"] == "2027-03-14"
    with pytest.raises(ValueError): parse_eci_terms("<table><tr><td>Access denied</td></tr></table>")


def test_watchlist_uses_terms_not_poll_dates_and_separates_counts():
    snapshot = seed_snapshot()
    assert len(snapshot["rows"]) == 31
    rows = upcoming_terms(snapshot, date(2026,10,30))
    assert rows[0]["State"] == "Manipur"
    assert len(rows) == 10
    up = next(r for r in rows if r["State"] == "Uttar Pradesh")
    assert (up["Assembly seats"], up["Lok Sabha allocation"], up["Rajya Sabha allocation"], up["Full-house majority"]) == (403,80,31,202)
    assert up["Election dates"].startswith("Not verified")


def test_market_context_measures_drawdown_without_claiming_event_causality():
    prices = pd.Series(np.r_[np.full(299,100),90.], index=pd.bdate_range("2025-01-01",periods=300))
    result = market_context(prices)
    assert result["drawdown"] == pytest.approx(-.1)
    assert result["as_of"] == str(prices.index[-1].date())
    with pytest.raises(ValueError): market_context(prices.iloc[:20])
