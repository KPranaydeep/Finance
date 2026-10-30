import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytest
from election_reserve import election_reserve_reference, combined_reserve, pack_inr_history, saved_event_reference, history_from_market_cache
from election_schedule import parse_schedule, monthly_schedule


def example():
    dates = mcal.get_calendar("NSE").valid_days("2024-12-01", "2025-06-05").tz_localize(None)
    frame = pd.DataFrame({"INDIA": 0., "US": -.2}, index=dates)
    frame.loc["2025-02-10", "INDIA"] = np.log(.9)
    weights, currencies = {"INDIA": .4, "US": .6}, {"INDIA": "INR", "US": "USD"}
    events = [dict(name="Past", counting_date="2025-02-08", poll_dates=["2025-02-05"], verified=True, source="https://www.eci.gov.in/past"),
              dict(name="Next", counting_date="2025-06-10", poll_dates=["2025-06-07"], verified=True, source="https://www.eci.gov.in/next")]
    return frame, weights, currencies, events


def test_only_inr_risk_scaled_to_total_and_max_not_sum():
    frame, weights, currencies, events = example()
    result = election_reserve_reference(frame, weights, currencies, events, "2025-06-06")
    assert result["status"] == "active_event_stress"
    assert result["event_reserve_fraction"] == pytest.approx(.04)
    assert combined_reserve(.02, result) == pytest.approx(.04)
    assert combined_reserve(.05, result) == .05
    assert result["active_events"][0]["preparation_date"] < "2025-06-10"


def test_distant_or_unverified_event_does_not_activate():
    frame, weights, currencies, events = example()
    events[1]["verified"] = False
    result = election_reserve_reference(frame, weights, currencies, events, "2025-06-06")
    assert result["historical_stress_fraction"] == pytest.approx(.04)
    assert result["event_reserve_fraction"] is None
    assert combined_reserve(.02, result) == .02


def test_missing_dates_not_imputed_and_overlaps_not_counted_twice():
    frame, weights, currencies, events = example()
    events.append(dict(events[0], counting_date="2025-02-09"))
    result = election_reserve_reference(frame, weights, currencies, events, "2025-06-06")
    assert len(result["windows"]) == 1
    result = election_reserve_reference(frame.drop(pd.Timestamp("2025-02-10")), weights, currencies, events, "2025-06-06")
    assert result["status"] == "insufficient_event_history"
    assert result["event_reserve_fraction"] is None


def test_history_roundtrip_and_currency_gap():
    frame, weights, currencies, events = example()
    payload = {"election_reserve_history": pack_inr_history(frame, weights, currencies)}
    assert saved_event_reference(payload, events, "2025-06-06")["event_reserve_fraction"] == pytest.approx(.04)
    assert election_reserve_reference(frame, weights, {}, events, "2025-06-06")["status"] == "incomplete_currency_or_weights"


def test_no_inr_and_ongoing_polling_window():
    frame, weights, currencies, events = example()
    assert election_reserve_reference(frame, weights, {"INDIA":"USD", "US":"USD"}, events, "2025-06-06")["event_reserve_fraction"] == 0
    events[1].update(counting_date="2025-08-20", poll_dates=["2025-06-05"])
    result = election_reserve_reference(frame, weights, currencies, events, "2025-06-06")
    assert result["status"] == "active_event_stress"
    assert result["event_reserve_fraction"] == pytest.approx(.04)


def test_schedule_parser_requires_explicit_poll_and_counting():
    text = "General Election to Legislative Assembly Date of Poll : 05-02-2025 Date of Counting : 08-02-2025"
    assert parse_schedule(text, "https://www.eci.gov.in/test")[0]["counting_date"] == "2025-02-08"
    with pytest.raises(ValueError):
        parse_schedule("Term ends 08-02-2025", "https://www.eci.gov.in/test")


def test_schedule_outage_preserves_baseline_and_is_monthly(tmp_path):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError("offline")
    path = tmp_path / "schedule.json"
    first = monthly_schedule(path, "2026-10-30", fetch=fail)
    second = monthly_schedule(path, "2026-10-31", fetch=fail)
    assert first["events"] == second["events"]
    assert len(first["events"]) == 3 and len(calls) == 2
    assert "refresh_error" in second


def test_old_analysis_recovers_inr_history_from_cache_without_download(tmp_path):
    from market_history_cache import sync_market_history
    frame, weights, currencies, events = example()
    prices = 100*np.exp(frame[["INDIA"]].cumsum())
    cache = tmp_path / "market.duckdb"
    sync_market_history(cache, ["INDIA"], "2024-12-01", "2025-06-06",
                        lambda *args: (prices, pd.DataFrame(), {}))
    payload = {"optimal_allocation":[{"ticker":t,"target_weight":w} for t,w in weights.items()],
               "current_allocation":[{"Yahoo Ticker":t,"Currency":c} for t,c in currencies.items()]}
    history = history_from_market_cache(payload, cache, "2025-06-06")
    result = saved_event_reference(dict(payload,election_reserve_history=history), events, "2025-06-06")
    assert history["source"] == "existing persistent price cache"
    assert history["columns"] == ["INDIA"]
    assert result["event_reserve_fraction"] == pytest.approx(.04)


def test_eci_failure_discovers_explicit_pib_schedule(tmp_path):
    class Response:
        def __init__(self,text): self.text=text
        def raise_for_status(self): pass
    def fetch(url, **kwargs):
        if 'eci.gov.in' in url: raise OSError('406 blocked')
        if 'allRel.aspx' in url:
            return Response('<a href="PressReleasePage.aspx?PRID=12345">Schedule for General Election to Legislative Assembly</a>')
        return Response('General Election to Legislative Assembly <table><tr><td>Date of Poll</td><td>01.03.2027</td></tr><tr><td>Date of Counting</td><td>05.03.2027</td></tr></table>')
    result = monthly_schedule(tmp_path/'schedule.json','2026-10-30',fetch=fetch)
    assert result["fallback_schedules"] == 1
    assert any(e["counting_date"] == "2027-03-05" for e in result["events"])
    assert len(result["events"]) == 4
