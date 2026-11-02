"""Bounded, shared history jobs: market-data delays must not block navigation."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import streamlit as st
from public_evidence_values import lifecycle_key, value_of_100


class EvidenceHistoryJobs:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="public-evidence")
        self.jobs = {}
        self.progress = {}
        self.lock = Lock()

    def _publish(self, key, rows, failures):
        with self.lock:
            self.progress[key] = (list(rows), list(failures))

    def _available(self, key):
        with self.lock:
            current = self.progress.get(key, ([], []))
            previous = []
            # Reuse the previous verified refresh of the exact same lifecycle
            # set while a new refresh runs; do not blank previously known values.
            if isinstance(key, tuple) and len(key) == 3:
                for old_key, future in reversed(list(self.jobs.items())):
                    if (isinstance(old_key, tuple) and old_key != key
                            and old_key[:2] == key[:2] and old_key[2] <= key[2]
                            and future.done()):
                        try:
                            rows, failures = future.result()
                            if rows:
                                previous = rows
                                break
                        except Exception:
                            continue
            merged = {lifecycle_key(row): row for row in previous}
            merged.update({lifecycle_key(row): row for row in current[0]})
        return "refreshing" if previous else "loading", list(merged.values()), current[1]

    def poll(self, key, loader, *args, progressive=False):
        with self.lock:
            future = self.jobs.get(key)
            if future is None:
                # Do not queue unlimited downloads when versions/windows change.
                if sum(not job.done() for job in self.jobs.values()) >= 2:
                    return "loading", [], []
                for old_key in list(self.jobs):
                    if len(self.jobs) < 8:
                        break
                    if self.jobs[old_key].done():
                        del self.jobs[old_key]
                        self.progress.pop(old_key, None)
                kwargs = {"publish": lambda rows, failed: self._publish(key, rows, failed)} if progressive else {}
                future = self.executor.submit(loader, *args, **kwargs)
                self.jobs[key] = future
        if not future.done():
            return self._available(key)
        try:
            summaries, failures = future.result()
            return "ready", summaries, failures
        except Exception:
            state, rows, failures = self._available(key)
            return "unavailable", rows, failures


@st.cache_resource(show_spinner=False)
def history_jobs():
    """Cache progressive batches and their previous completed outcomes (v2)."""
    return EvidenceHistoryJobs()


def _load_summary_batches(publication_id, securities, refresh_bucket, *, publish):
    from public_track_record import load_portfolio_summaries

    # Current holdings first. Each batch reuses the same card calculation,
    # without rendering images or waiting for every historical exit to finish.
    ordered = sorted(securities, key=lambda row: (row[3] == "removed", row[0]))
    summaries, failures = [], []
    loader = load_portfolio_summaries.__wrapped__
    for start in range(0, len(ordered), 8):
        batch = tuple(ordered[start:start + 8])
        try:
            rows, failed = loader(publication_id, batch, refresh_bucket)
            summaries.extend(rows)
            failures.extend(failed)
        except Exception:
            failures.extend(row[0] for row in batch)
        publish(summaries, failures)
    return summaries, failures


def poll_portfolio_summaries(publication_id, securities, refresh_bucket):
    key = (publication_id, securities, refresh_bucket)
    return history_jobs().poll(key, _load_summary_batches, publication_id, securities,
                               refresh_bucket, progressive=True)


@st.fragment(run_every="3s")
def watch_summary_progress(publication_id, securities, refresh_bucket, expected):
    state, rows, failures = poll_portfolio_summaries(publication_id, securities, refresh_bucket)
    if summary_progress_signature(state, rows, failures) != expected:
        st.rerun()


def summary_progress_signature(state, rows, failures):
    return (state, tuple(sorted((*lifecycle_key(row), str(row.get("data_through") or ""),
                                 value_of_100(row)) for row in rows)), tuple(sorted(failures)))
