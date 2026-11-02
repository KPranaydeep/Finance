from threading import Event
from concurrent.futures import ThreadPoolExecutor

from public_evidence_history import EvidenceHistoryJobs


def test_slow_history_is_nonblocking_and_repeated_polls_share_one_job():
    jobs = EvidenceHistoryJobs()
    gate, started = Event(), Event()
    calls = []

    def loader():
        calls.append(1)
        started.set()
        gate.wait(5)
        return [{"ticker": "X", "return": .2}], ["Y"]

    try:
        assert jobs.poll("same", loader) == ("loading", [], [])
        assert started.wait(1)
        assert jobs.poll("same", loader) == ("loading", [], [])
        assert calls == [1]
        gate.set()
        jobs.jobs["same"].result(timeout=1)
        assert jobs.poll("same", loader) == ("ready", [{"ticker": "X", "return": .2}], ["Y"])
        assert calls == [1]
    finally:
        gate.set()
        jobs.executor.shutdown(wait=True)


def test_active_jobs_are_bounded_and_do_not_queue_more_downloads():
    jobs, gate = EvidenceHistoryJobs(), Event()

    def loader():
        gate.wait(5)
        return [], []

    try:
        jobs.poll("a", loader)
        jobs.poll("b", loader)
        assert jobs.poll("c", loader) == ("loading", [], [])
        assert set(jobs.jobs) == {"a", "b"}
    finally:
        gate.set()
        jobs.executor.shutdown(wait=True)


def test_failed_job_stays_unavailable_without_retry_storm():
    jobs = EvidenceHistoryJobs()

    def loader():
        raise ValueError("provider unavailable")

    try:
        jobs.poll("failed", loader)
        try:
            jobs.jobs["failed"].result(timeout=1)
        except ValueError:
            pass
        assert jobs.poll("failed", loader) == ("unavailable", [], [])
        assert len(jobs.jobs) == 1
    finally:
        jobs.executor.shutdown(wait=True)


def test_yahoo_batches_cannot_reset_each_others_shared_state(monkeypatch):
    from public_market_data import download
    entered, release, second_started, second_entered = Event(), Event(), Event(), Event()

    def provider(ticker):
        if ticker == "first":
            entered.set()
            release.wait(5)
        else:
            second_entered.set()
        return ticker

    def second_call():
        second_started.set()
        return download("second")

    monkeypatch.setattr("public_market_data.yf.download", provider)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(download, "first")
        assert entered.wait(1)
        second = executor.submit(second_call)
        try:
            assert second_started.wait(1)
            assert not second_entered.wait(.05)
        finally:
            release.set()
        assert first.result(timeout=1) == "first"
        assert second.result(timeout=1) == "second"
