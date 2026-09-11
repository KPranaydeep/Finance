"""Gap-safe shared daily returns: no filling or multi-session one-day returns."""
import numpy as np
import pandas as pd
from . import market


def common_history(histories, as_of, policy):
    raw = pd.DataFrame({t: h.Close for t, h in histories.items()}).sort_index()
    if raw.empty or raw.index.duplicated().any():
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.where(np.isfinite(raw) & (raw > 0))
    # Restrict to the common listing/availability period, not the oldest member.
    starts = [raw[t].first_valid_index() for t in raw]
    if any(d is None for d in starts):
        raise ValueError("INSUFFICIENT_COMMON_HISTORY")
    start = max(starts)
    dates = [str(d.date()) for d in market.calendar(start, as_of, policy).index]
    frame = raw.reindex(dates)
    if frame.empty or frame.iloc[-1].isna().any():
        raise ValueError("STALE_OR_INCOMPLETE_MARKET_HISTORY")
    changes = frame.pct_change(fill_method=None)
    valid = changes.notna().all(axis=1) & np.isfinite(changes).all(axis=1)
    returns = changes.loc[valid].copy()
    returns.attrs["session_positions"] = {d: i for i, d in enumerate(dates)}
    details = {"start": start, "end": as_of, "usable_daily_returns": len(returns),
               "missing_sessions": frame.index[frame.isna().any(axis=1)].tolist(),
               "method": "complete-adjacent-session-pairs-no-fill"}
    return frame.dropna(), returns, details
