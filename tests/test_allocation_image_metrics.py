from copy import deepcopy
from allocation_image_metrics import saved_image_metrics, definition_ribbon
from saved_allocation_image import saved_weights_image_spec


def test_saved_statistics_are_transcribed_without_recalculation_or_invented_data():
    payload={"current_stats":{"Annual Return":.465,"Annual Volatility":.1645,"Sharpe Ratio":2.33},
             "optimal_stats":{"Annual Return":.4838,"Annual Volatility":.1171,"Sharpe Ratio":3.43,
                "Block-Bootstrap Median Return (30 Sessions)":.0407,
                "Block-Bootstrap ES 95% (30 Sessions)":.0413,
                "Block-Bootstrap Tail Loss SD 95% (30 Sessions)":.0156},
             "portfolio_metrics_config":{"risk_free_rate_annual":.08225,"trading_days_per_year":252}}
    before=deepcopy(payload);metrics,horizon=saved_image_metrics(payload)
    assert payload==before and horizon==30
    assert metrics[0][1:3]==("46.50%","48.38%")
    assert metrics[2][1:3]==("2.33","3.43")
    assert metrics[4][1:3]==("Unavailable","Unavailable")
    assert metrics[5][1:3]==("Unavailable","4.13%")
    assert metrics[-1][2:]==("8.225%","252 sessions/year")


def test_missing_nonfinite_negative_risk_and_zero_volatility_are_not_fabricated():
    metrics,horizon=saved_image_metrics({"optimal_stats":{"Annual Return":-.2,"Annual Volatility":0,
        "Sharpe Ratio":0,"Historical ES 95% (1 Session)":-.01,
        "Block-Bootstrap ES 95% (30 Sessions)":float('nan')}})
    assert metrics[0][2]=="-20.00%" and metrics[1][2]=="0.00%"
    assert metrics[2][2]=="Unavailable" and metrics[4][2]=="Unavailable" and metrics[5][2]=="Unavailable"


def test_legacy_horizon_is_preserved_and_current_horizon_is_not_mixed():
    metrics,h=saved_image_metrics({"current_stats":{"Block-Bootstrap ES 95% (30 Sessions)":.08},
                                 "optimal_stats":{"Block-Bootstrap ES 95% (20 Sessions)":.04}})
    assert h==20 and metrics[5][0]=="ES 95% · 20 SESSIONS"
    assert metrics[5][1:3]==("Unavailable","4.00%")
    assert "20-session" in definition_ribbon(h)[3][0]


def test_refreshed_metrics_do_not_inherit_a_stale_risk_free_assumption():
    payload={"portfolio_metrics_as_of":"2026-11-11","review_analysis_context":{"risk_free_rate_annual":.05,"trading_days_per_year":252}}
    assert saved_image_metrics(payload)[0][-1][2]=="Unavailable"
    payload["portfolio_metrics_config"]={"risk_free_rate_annual":0,"trading_days_per_year":252}
    assert saved_image_metrics(payload)[0][-1][2]=="0.000%"


def test_image_dates_distinguish_saved_weights_from_refreshed_statistics():
    spec=saved_weights_image_spec({"saved_at":"2026-11-09T00:00:00Z","portfolio_metrics_as_of":"2026-11-11",
        "history":{"valid_end":"2026-11-08"},"portfolio_metrics_refresh_error":"Earlier readings retained",
        "optimal_allocation":[{"ticker":"EXAMPLE.NS","target_weight":1}]})
    assert spec.date_slug=="2026-11-09" and spec.history_label=="11 Nov 2026" and spec.stats_retained


def test_definitions_identify_the_app_conventions_and_limits():
    text=" ".join(t+" "+d for t,d in definition_ribbon(30))
    for phrase in ("not CAGR","Conventional Sharpe","not a calibrated forecast","Not VaR or maximum drawdown",
                   "short-range dependence","In-sample comparisons","rounded targets"):
        assert phrase in text


def test_educational_labels_match_the_actual_portfolio_calculator_conventions():
    import numpy as np
    import pandas as pd
    import pytest
    import portfolio_optimizer_core as core
    rng=np.random.default_rng(124)
    history=pd.DataFrame(rng.normal(.0007,.012,(252,2)),columns=["A","B"])
    weights=np.array([.4,.6]);series=history@weights
    stats=core.portfolio_stats(weights,history)
    n=core.portfolio_stats.__globals__["TRADING_DAYS_PER_YEAR"]
    rf=core.portfolio_stats.__globals__["RISK_FREE_RATE_ANNUAL"]
    assert stats["Annual Return"]==pytest.approx(series.mean()*n)
    assert stats["Annual Volatility"]==pytest.approx(series.std()*np.sqrt(n))
    assert stats["Sharpe Ratio"]==pytest.approx((series.mean()*n-rf)/(series.std()*np.sqrt(n)))
    metrics,h=saved_image_metrics({"optimal_stats":stats})
    assert metrics[0][2]==f'{stats["Annual Return"]:.2%}' and h==30
