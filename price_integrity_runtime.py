"""Bounded evidence I/O; completed checks survive reruns and process restarts.

Only provider evidence is cached here, never holdings or optimizer results.
Timeouts/missing evidence remain unknown. Workers never render Streamlit UI.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import pandas as pd


# Shared across runs: a provider stall cannot create an unbounded worker backlog.
_WORKERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="price-evidence")


def integrity_progress_label(details):
    phase = str(details.get("integrity_phase", "evidence"))
    done = int(details.get("integrity_done", 0))
    total = int(details.get("integrity_total", 0))
    cached = int(details.get("integrity_cached", 0))
    elapsed = float(details.get("integrity_elapsed", 0))
    budget = float(details.get("integrity_budget", 60))
    return (f"{phase} {done:,}/{total:,} · {cached:,} cached · "
            f"{elapsed:.0f}s / {budget:.0f}s request budget")


class EvidenceCache:
    def __init__(self, path, version, start, end):
        self.path = Path(path) if path else None
        self.scope = (version, start, end)
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(self.path, timeout=1) as db:
                    db.execute("CREATE TABLE IF NOT EXISTS evidence "
                               "(key TEXT PRIMARY KEY, expires REAL, payload TEXT)")
                    db.execute("DELETE FROM evidence WHERE expires < ?", (time.time(),))
            except (OSError, sqlite3.Error):
                self.path = None

    def key(self, kind, ticker):
        return hashlib.sha256(json.dumps((*self.scope, kind, ticker)).encode()).hexdigest()

    def read(self, kind, tickers):
        found = {}
        if not self.path:
            return found
        try:
            with sqlite3.connect(self.path, timeout=1) as db:
                for ticker in tickers:
                    row = db.execute("SELECT payload FROM evidence WHERE key=? AND expires>?",
                                     (self.key(kind, ticker), time.time())).fetchone()
                    if row:
                        try:
                            value = json.loads(row[0])
                            found[ticker] = None if value is None else {
                                k: pd.Series(v["values"], index=pd.to_datetime(v["dates"]))
                                if isinstance(v, dict) and "dates" in v else v
                                for k, v in value.items()
                            }
                        except (ValueError, TypeError, KeyError, AttributeError):
                            pass  # A malformed cache is never validation evidence.
        except sqlite3.Error:
            pass
        return found

    def write(self, kind, ticker, value):
        if not self.path:
            return
        try:
            payload = None if value is None else {
                k: {"dates": [str(x) for x in v.index],
                    "values": [None if pd.isna(x) else float(x) for x in v]}
                if isinstance(v, pd.Series) else v for k, v in value.items()
            }
            with sqlite3.connect(self.path, timeout=1) as db:
                db.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?)",
                           (self.key(kind, ticker), time.time() + (86400 if value else 900),
                            json.dumps(payload, allow_nan=False)))
        except (OSError, sqlite3.Error, ValueError, TypeError):
            pass


def bounded_evidence(tickers, fetch_batch, *, cache, kind, budget_seconds=60,
                     batch_size=3, on_progress=None):
    """Return completed evidence by deadline, without waiting for stalled calls.

At most four jobs are submitted at once. In-flight calls can finish into the
cache after the deadline; they cannot mutate the returned report. Cached failures
expire after 15 minutes, successes after 24 hours. No per-ticker retry cascade.
"""
    tickers = tuple(dict.fromkeys(tickers))
    results = cache.read(kind, tickers)
    cached = len(results)
    todo = [x for x in tickers if x not in results]
    jobs = iter(tuple(todo[i:i + batch_size]) for i in range(0, len(todo), batch_size))
    started = time.monotonic()
    deadline = started + max(0, budget_seconds)
    pending = {}

    def notify():
        if on_progress:
            try:
                on_progress({"integrity_phase": kind, "integrity_done": len(results),
                             "integrity_total": len(tickers), "integrity_cached": cached,
                             "integrity_elapsed": time.monotonic() - started,
                             "integrity_budget": budget_seconds,
                             "integrity_deferred": len(tickers) - len(results)})
            except Exception:
                pass

    def run(batch):
        try:
            values = fetch_batch(batch) or {}
        except Exception:
            values = {}
        completed = {ticker: values.get(ticker) for ticker in batch}
        for ticker, value in completed.items():
            cache.write(kind, ticker, value)
        return completed

    def fill():
        while len(pending) < 4 and time.monotonic() < deadline:
            batch = next(jobs, None)
            if batch is None:
                break
            pending[_WORKERS.submit(run, batch)] = batch

    notify()
    fill()
    while pending and time.monotonic() < deadline:
        done, _ = wait(pending, timeout=min(1, max(0, deadline - time.monotonic())),
                       return_when=FIRST_COMPLETED)
        for future in done:
            pending.pop(future)
            results.update(future.result())
        notify()  # Heartbeat even when the provider has not answered.
        fill()
    for future in pending:
        future.cancel()
    notify()
    return results
