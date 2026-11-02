"""Bounded, shared history jobs: market-data delays must not block navigation."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import streamlit as st


class EvidenceHistoryJobs:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="public-evidence")
        self.jobs = {}
        self.lock = Lock()

    def poll(self, key, loader, *args):
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
                future = self.executor.submit(loader, *args)
                self.jobs[key] = future
        if not future.done():
            return "loading", [], []
        try:
            summaries, failures = future.result()
            return "ready", summaries, failures
        except Exception:
            return "unavailable", [], []


@st.cache_resource(show_spinner=False)
def history_jobs():
    return EvidenceHistoryJobs()


def poll_portfolio_summaries(publication_id, securities, refresh_bucket):
    from public_track_record import load_portfolio_summaries

    key = (publication_id, securities, refresh_bucket)
    # The worker runs only pure market-data calculation; no Streamlit calls.
    loader = load_portfolio_summaries.__wrapped__
    return history_jobs().poll(key, loader, publication_id, securities, refresh_bucket)
