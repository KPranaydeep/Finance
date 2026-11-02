from threading import Event
from concurrent.futures import ThreadPoolExecutor

from public_evidence_history import EvidenceHistoryJobs, _load_summary_batches, summary_progress_signature


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


def test_partial_batches_are_visible_without_waiting_for_the_full_job():
    jobs, gate, published = EvidenceHistoryJobs(), Event(), Event()
    row = {"ticker": "VENUSREM.NS", "entry_date": "2026-09-30", "return": .2}

    def loader(*, publish):
        publish([row], [])
        published.set()
        gate.wait(5)
        return [row], []

    try:
        jobs.poll("partial", loader, progressive=True)
        assert published.wait(1)
        assert jobs.poll("partial", loader, progressive=True) == ("loading", [row], [])
    finally:
        gate.set()
        jobs.executor.shutdown(wait=True)


def test_refresh_preserves_previous_values_and_overlays_new_batches():
    jobs, gate, published = EvidenceHistoryJobs(), Event(), Event()
    old = {"ticker": "GOLDBETA.NS", "entry_date": "2026-09-15", "return": .1}
    other = {"ticker": "VENUSREM.NS", "entry_date": "2026-09-30", "return": .2}
    prefix = ("PUB", (("GOLDBETA.NS", "2026-09-15"), ("VENUSREM.NS", "2026-09-30")))
    old_key, new_key = (*prefix, "2026-10-30T10"), (*prefix, "2026-10-30T11")

    def loader(*, publish):
        publish([{**old, "return": .15}], [])
        published.set()
        gate.wait(5)
        return [], []

    try:
        jobs.poll(old_key, lambda: ([old, other], []))
        jobs.jobs[old_key].result(timeout=1)
        jobs.poll(new_key, loader, progressive=True)
        assert published.wait(1)
        state, rows, failures = jobs.poll(new_key, loader, progressive=True)
        assert state == "refreshing"
        assert {row["ticker"]: row["return"] for row in rows} == {"GOLDBETA.NS": .15, "VENUSREM.NS": .2}
        assert summary_progress_signature(state, rows, []) != summary_progress_signature(state, [old, other], [])
    finally:
        gate.set()
        jobs.executor.shutdown(wait=True)


def test_small_batches_fill_every_security_without_opening_a_card(monkeypatch):
    from public_track_record import load_portfolio_summaries
    calls, progress = [], []
    securities = tuple((f"S{i:02}.NS", "2026-09-15", None, "active" if i >= 2 else "removed", .1)
                       for i in range(18))

    def loader(publication_id, batch, refresh_bucket):
        calls.append(batch)
        return [{"ticker": row[0], "entry_date": row[1], "exit_date": row[2],
                 "return": .2} for row in batch], []

    monkeypatch.setattr(load_portfolio_summaries, "__wrapped__", loader)
    rows, failed = _load_summary_batches("PUB", securities, "window",
                                         publish=lambda rows, failures: progress.append(len(rows)))
    assert [len(batch) for batch in calls] == [8, 8, 2]
    assert calls[0][0][3] == "active"
    assert progress == [8, 16, 18]
    assert {row["ticker"] for row in rows} == {row[0] for row in securities}
    assert failed == []
