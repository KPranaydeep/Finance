"""Align end-of-day portfolio NAV with the VT adjusted-close INR proxy."""
import numpy as np
import pandas as pd

LABEL = "Global stocks — VT (INR)"


def daily_series(series):
    result = pd.to_numeric(series, errors="coerce").astype(float).copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index)).tz_localize(None).normalize()
    if result.index.duplicated().any():
        raise ValueError("Duplicate daily benchmark observations")
    return result.where(np.isfinite(result) & (result > 0)).dropna().sort_index()


def compare_world_benchmark(nav_rows, vt_adjusted, usd_inr):
    """Use only dates observed in all three sources; never fill missing prices."""
    nav = pd.DataFrame(nav_rows)
    if nav.empty:
        return pd.DataFrame()
    portfolio = daily_series(pd.Series(nav["nav"].to_numpy(),index=nav["nav_date"]))
    joined = pd.concat([portfolio.rename("Portfolio"),
                        daily_series(vt_adjusted).rename("VT"),
                        daily_series(usd_inr).rename("FX")],axis=1,join="inner").dropna()
    if len(joined) < 2:
        return pd.DataFrame()
    values = pd.DataFrame({"Portfolio — estimated net":joined["Portfolio"],
                           LABEL:joined["VT"] * joined["FX"]})
    return values.div(values.iloc[0]).mul(100)
