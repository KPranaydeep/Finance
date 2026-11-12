"""Explain the saved calculator's actual conventions, without recomputing metrics."""
import math
import re


def finite_number(value, *, nonnegative=False):
    try:
        if isinstance(value,bool):
            return None
        number=float(value)
        return number if math.isfinite(number) and (not nonnegative or number>=0) else None
    except (ValueError,TypeError,OverflowError):
        return None


def image_metric_config(payload):
    config=payload.get("portfolio_metrics_config") or {}
    if not config and not payload.get("portfolio_metrics_as_of"):
        config=payload.get("review_analysis_context") or (payload.get("settings") or {}).get("optimizer_config") or {}
    return config


def saved_image_metrics(payload):
    current=payload.get("current_stats") or {};optimal=payload.get("optimal_stats") or {}
    horizons={int(m.group(1)) for key in optimal for m in [re.search(r"Block-Bootstrap .*\((\d+) Sessions\)",str(key))] if m}
    horizon=30 if 30 in horizons else next(iter(horizons)) if len(horizons)==1 else None
    h=str(horizon) if horizon else "?"
    metrics=(
        ("ANNUAL LOG RETURN","Annual Return",False,False,"Historical rate · not CAGR"),
        ("ANNUAL VOLATILITY","Annual Volatility",False,True,"Upside and downside variability"),
        ("SHARPE-LIKE · APP","Sharpe Ratio",True,False,"Risk-adjusted comparison · approximation"),
        (f"MEDIAN · {h} SESSIONS",f"Block-Bootstrap Median Return ({h} Sessions)",False,False,"Middle resampled outcome"),
        ("ES 95% · 1 SESSION","Historical ES 95% (1 Session)",False,True,"Historical worst-tail average loss"),
        (f"ES 95% · {h} SESSIONS",f"Block-Bootstrap ES 95% ({h} Sessions)",False,True,"Resampled worst-tail average loss"),
        (f"TAIL SD 95% · {h} SESSIONS",f"Block-Bootstrap Tail Loss SD 95% ({h} Sessions)",False,True,"Dispersion within the loss tail"),
    )
    result=[]
    for label,key,ratio,nonnegative,note in metrics:
        values=[]
        for stats in (current,optimal):
            value=finite_number(stats.get(key),nonnegative=nonnegative)
            if ratio and finite_number(stats.get("Annual Volatility"),nonnegative=True) in (None,0):
                value=None
            values.append("Unavailable" if value is None else f"{value:.2f}" if ratio else f"{value:.2%}")
        result.append((label,*values,note))
    config=image_metric_config(payload)
    rf=finite_number(config.get("risk_free_rate_annual"))
    n=finite_number(config.get("trading_days_per_year"),nonnegative=True)
    result.append(("SAVED RISK-FREE INPUT",None,"Unavailable" if rf is None else f"{rf:.3%}",
                   f"{n:g} sessions/year" if n else "Annualization not recorded"))
    return tuple(result),horizon


def definition_ribbon(horizon, *, recalculated=False):
    h=f"{horizon}-session" if horizon else "Multi-session (horizon unrecorded)"
    return (
        ("Annual log return", "Mean weighted daily log return × sessions/year. Historical log-rate; not CAGR, XIRR or a future-return promise."),
        ("Volatility", "Daily log-return standard deviation × square root of sessions/year. Measures swings in both directions; not a loss bound."),
        ("Sharpe-like (app approximation)", "(Annual log-return rate − annual risk-free input) ÷ annual log volatility. Conventional Sharpe uses matching excess returns; this app mixes conventions."),
        (f"Median · {h}", "Middle resampled simple return: half the sampled outcomes lie on either side. A scenario summary, not a calibrated forecast."),
        ("Expected shortfall · ES 95%", "Average loss in the worst approximately 5% of outcomes. One-session uses history; multi-session uses resampling. Not VaR or maximum drawdown."),
        ("Tail-loss standard deviation · SD", "Sample standard deviation of non-negative losses within the worst approximately 5% tail. Describes tail dispersion, not total portfolio volatility."),
        ("Moving-block bootstrap", "Resamples short adjacent stretches of history to retain some short-range dependence. Unseen regimes and gaps may be absent."),
        ("Allocation and comparison", "Image weights retain their original base. Selected current and optimal mixes are each normalized to 100% for statistics, using INR-valued history. In-sample comparisons can overstate improvement." if recalculated else "Weights share the stated capital base. Statistics come from the saved analysis; rounded targets can differ from fitted weights. In-sample comparisons can overstate improvement."),
    )
