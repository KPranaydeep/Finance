"""Calendar-horizon historical scenarios and immutable outcome matching."""
from datetime import date, timedelta
import numpy as np
import pandas as pd

HORIZON_DAYS = 28
MINIMUM_NAV_ROWS = 126
METHOD = "historical-calendar-block-28d-v1"
VERSION = "forecast-v3-28calendar"


def calendar_outlook(rows, forecast_date):
    """Use complete four-week historical blocks aligned to issuance weekday.

    Overlapping blocks preserve serial dependence inside each scenario; they
    are not independent trials. Thresholds are operational minimums, not
    evidence that six months is statistically sufficient in every regime.
    """
    origin = pd.Timestamp(forecast_date).normalize()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame["nav_date"] = pd.to_datetime(frame["nav_date"])
    frame = frame.sort_values("nav_date")
    frame = frame.loc[frame.nav_date <= origin]
    if frame.nav_date.duplicated().any():
        raise ValueError("Forecast NAV dates must be unique")
    values = pd.to_numeric(frame["nav"], errors="coerce").astype(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Forecast NAV must be finite and positive")
    if len(frame) < MINIMUM_NAV_ROWS:
        return None
    history = pd.Series(values.to_numpy(), index=frame.nav_date)
    calendar = history.resample("D").ffill()
    starts = calendar.index[(calendar.index.weekday == origin.weekday()) &
                            (calendar.index + pd.Timedelta(days=HORIZON_DAYS) <= calendar.index[-1])]
    if len(starts) < 20:
        return None
    ends = starts + pd.Timedelta(days=HORIZON_DAYS)
    outcomes = calendar.loc[ends].to_numpy() / calendar.loc[starts].to_numpy() - 1
    q05,q25,q50,q75,q95 = np.quantile(outcomes,[.05,.25,.5,.75,.95])
    return dict(horizon_days=HORIZON_DAYS, horizon_unit="calendar_days", method=METHOD,
                sample_start=history.index[0].date().isoformat(),
                sample_end=history.index[-1].date().isoformat(), observation_count=len(history),
                scenario_count=len(outcomes), overlapping_scenarios=True,
                reference_date=history.index[-1].date().isoformat(), reference_nav=float(history.iloc[-1]),
                target_date=(origin.date()+timedelta(days=HORIZON_DAYS)).isoformat(),
                median_return=float(q50),lower_50=float(q25),upper_50=float(q75),
                lower_90=float(q05),upper_90=float(q95),
                probability_positive=float(np.mean(outcomes>0)),
                probability_negative=float(np.mean(outcomes<0)),
                probability_loss_gt_threshold=float(np.mean(outcomes<-.05)),loss_threshold=-.05)


def calendar_realization(payload, observations):
    """Wait for data through expiry; use last close on/before expiry."""
    target = date.fromisoformat(payload["target_date"])
    ordered = sorted((pd.Timestamp(row["nav_date"]).date(),float(row["nav"])) for row in observations)
    if not ordered or ordered[-1][0] < target:
        return None
    eligible = [row for row in ordered if row[0] <= target]
    if not eligible:
        return None
    return (date.fromisoformat(payload["reference_date"]),float(payload["reference_nav"])),eligible[-1]
