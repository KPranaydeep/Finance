import numpy as np
import pandas as pd

from public_planning_prices import latest_planning_prices


def test_partial_quote_failure_retries_only_missing_and_retains_good_prices(monkeypatch):
    calls = []

    def download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        data = ({"A.NS": [100., 101.], "MNST": [np.nan, np.nan]}
                if len(calls) == 1 else {"MNST": [4500., np.nan]})
        return pd.DataFrame(data, index=pd.to_datetime(["2026-10-30", "2026-11-01"])), {
            "A.NS": "INR", "MNST": "USD"}

    monkeypatch.setattr("public_planning_prices.download_inr", download)
    result = latest_planning_prices(("A.NS", "MNST", "MNST"))
    assert [call[0] for call in calls] == [["A.NS", "MNST"], ["MNST"]]
    assert calls[1][1]["period"] == "1mo"
    assert calls[1][1]["auto_adjust"] is False
    assert calls[1][1]["timeout"] == 10
    assert result["A.NS"]["price"] == 101.
    assert result["MNST"] == {"price": 4500., "currency": "INR",
                               "source_currency": "USD", "price_as_of": "2026-10-30"}


def test_failed_retry_does_not_erase_successful_quotes_or_invent_missing(monkeypatch):
    calls = []

    def download(tickers, **kwargs):
        calls.append(tickers)
        if len(calls) == 2:
            raise ValueError("temporary provider failure")
        return pd.DataFrame({"A.NS": [100.], "MNST": [np.nan]},
                            index=pd.to_datetime(["2026-10-30"])), {"A.NS": "INR"}

    monkeypatch.setattr("public_planning_prices.download_inr", download)
    assert set(latest_planning_prices(("A.NS", "MNST"))) == {"A.NS"}
    assert len(calls) == 2


def test_widespread_outage_is_not_retried_repeatedly(monkeypatch):
    calls = []

    def download(tickers, **kwargs):
        calls.append(tickers)
        raise ValueError("provider unavailable")

    monkeypatch.setattr("public_planning_prices.download_inr", download)
    assert latest_planning_prices(tuple(f"SEC{i}" for i in range(26))) == {}
    assert len(calls) == 1


def test_invalid_prices_and_missing_fx_remain_unavailable(monkeypatch):
    monkeypatch.setattr("public_planning_prices.download_inr", lambda *a, **k: (
        pd.DataFrame({"MNST": [np.inf, -1., 0., np.nan]},
                     index=pd.date_range("2026-10-27", periods=4)), {"MNST": "USD"}))
    assert latest_planning_prices(("MNST",)) == {}
    assert latest_planning_prices(()) == {}
