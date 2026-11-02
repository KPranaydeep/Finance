"""Serialize public Yahoo batches, whose installed client shares batch state."""

from threading import RLock

import yfinance as yf


_download_lock = RLock()


def download(*args, **kwargs):
    # yfinance 0.2.x resets shared._DFS per call, then threaded downloads wait
    # for its batch size. Concurrent public-page calls can corrupt that state
    # and strand the wait loop. Keep each complete batch together.
    with _download_lock:
        return yf.download(*args, **kwargs)
