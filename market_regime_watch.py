"""Causal change flags from observed prices, using River's ADWIN detector."""
import numpy as np
import pandas as pd


def detect_market_changes(closes, recent_sessions=30):
    try:
        import river
        from river.drift import ADWIN
    except ImportError:
        return {"status": "unavailable", "reason": "Install the declared river dependency to enable change detection"}
    prices = pd.to_numeric(closes, errors="coerce").dropna().sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.loc[np.isfinite(prices) & prices.gt(0)]
    returns = prices.pct_change(fill_method=None).dropna()
    warmup = 60
    if len(returns) < warmup + 60:
        return {"status": "insufficient_history", "reason": "At least 120 daily returns required"}
    # Bounded features for ADWIN. Scale uses ONLY the initial warmup, never
    # future observations. Later samples cannot change historical flags.
    scale = max(float(returns.iloc[:warmup].abs().median()), 1e-6)
    detectors = {name: ADWIN(delta=.001, clock=1, min_window_length=30, grace_period=60)
                 for name in ("Return direction", "Return magnitude")}
    events = []
    for position, (when, value) in enumerate(returns.items()):
        features = {"Return direction": 1.0 if value > 0 else 0.0 if value < 0 else .5,
                    "Return magnitude": abs(float(value)) / (abs(float(value)) + scale)}
        for name, detector in detectors.items():
            detector.update(features[name])
            if detector.drift_detected:
                previous = returns.iloc[max(0,position-59):max(0,position-29)]
                recent = returns.iloc[max(0,position-29):position+1]
                if name == "Return magnitude":
                    comparison = float(recent.abs().mean() - previous.abs().mean())
                    description = "Volatility pressure increased" if comparison > 0 else "Volatility pressure decreased"
                else:
                    comparison = float((recent > 0).mean() - (previous > 0).mean())
                    description = "Positive-return frequency increased" if comparison > 0 else "Positive-return frequency decreased"
                events.append({"Detected on": str(when)[:10], "Indicator": name, "Change": description,
                               "Observation": position, "ADWIN window": int(detector.width)})
    recent_events = [event for event in events if event["Observation"] >= len(returns)-recent_sessions]
    return {"status": "recent_change" if recent_events else "no_recent_change",
            "as_of": str(returns.index[-1])[:10], "observations": len(returns),
            "recent_sessions": recent_sessions, "events": events, "recent_events": recent_events,
            "library": f"river {river.__version__} · ADWIN",
            "method": "Sequential direction and bounded return-magnitude features; delta 0.001 per stream, 30-observation minimum subwindows. Flags date detection, not the start of a regime. Serial dependence and repeated monitoring mean delta is not a calibrated market-change probability."}
