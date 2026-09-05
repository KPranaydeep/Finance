from datetime import date

import numpy as np
import pytest

from public_portfolio_trust import bootstrap_outlook, performance_metrics, versioned_model_nav, xirr
from public_portfolio_publications import validate_constituents, verify_trust_audit


def test_xirr_single_investment():
    assert xirr([(date(2020,1,1),-1000),(date(2021,1,1),1100)]) == pytest.approx(0.10, abs=3e-4)


def test_xirr_multiple_contributions_irregular_dates():
    value = xirr([(date(2020,1,1),-1000),(date(2020,4,15),-500),(date(2021,7,1),1800)])
    assert value is not None and value > 0


def test_xirr_withdrawal_and_terminal_value():
    value = xirr([(date(2020,1,1),-1000),(date(2020,7,1),200),(date(2021,1,1),900)])
    assert value is not None and value > 0


def test_xirr_contribution_then_withdrawal():
    assert xirr([(date(2020,1,1),-1000),(date(2020,6,1),1200)]) is not None


def test_xirr_zero_and_near_zero():
    assert xirr([(date(2020,1,1),-1000),(date(2021,1,1),1000)]) == pytest.approx(0, abs=1e-10)


def test_xirr_negative_return():
    assert xirr([(date(2020,1,1),-1000),(date(2021,1,1),800)]) < 0


def test_xirr_known_multiple_flow_case():
    flows=[(date(2018,1,1),-10000),(date(2018,6,1),-2500),(date(2019,2,1),4000),(date(2020,1,1),11000)]
    rate=xirr(flows)
    assert rate is not None
    from public_portfolio_trust import xnpv, _dated_flows
    assert xnpv(rate,_dated_flows(flows)) == pytest.approx(0, abs=1e-5)


@pytest.mark.parametrize("flows", [[],[(date(2020,1,1),-1)],[(date(2020,1,1),1),(date(2021,1,1),2)],[(date(2020,1,1),-1),(date(2021,1,1),-2)]])
def test_xirr_insufficient_or_invalid(flows):
    assert xirr(flows) is None


def test_performance_math():
    rows=[{"nav_date":"2024-01-01","nav":100},{"nav_date":"2024-01-02","nav":110},{"nav_date":"2024-01-03","nav":99}]
    result=performance_metrics(rows)
    assert result["total_return"] == pytest.approx(-0.01)
    assert result["maximum_drawdown"] == pytest.approx(-0.10)
    assert result["best_day"] == pytest.approx(0.10)
    assert result["worst_day"] == pytest.approx(-0.10)


def test_bootstrap_is_reproducible_and_bounded():
    returns=np.tile([-.01,.0,.012,.004,-.003],30)
    dates=[date(2024,1,1)]*len(returns)
    a=bootstrap_outlook(returns,dates,seed=42,simulations=1000)
    b=bootstrap_outlook(returns,dates,seed=42,simulations=1000)
    assert a == b and a.horizon_days == 14
    assert a.lower_90 <= a.lower_50 <= a.median_return <= a.upper_50 <= a.upper_90
    assert 0 <= a.probability_positive <= 1 and 0 <= a.probability_negative <= 1


def test_forecast_requires_history():
    assert bootstrap_outlook([0.01]*20,[date(2024,1,1)]*20) is None


def test_weights_and_duplicates():
    assert len(validate_constituents([{"ticker":"a.ns","target_weight":.9}],.1)) == 1
    with pytest.raises(ValueError): validate_constituents([{"ticker":"A","target_weight":.5},{"ticker":"a","target_weight":.5}],0)
    with pytest.raises(ValueError): validate_constituents([{"ticker":"A","target_weight":.8}],0)


def test_versioned_nav_switches_immutable_portfolio_versions():
    import pandas as pd
    prices=pd.DataFrame({"A":[100,110,110],"B":[100,100,120]},index=pd.date_range("2024-01-01",periods=3))
    publications=[
        {"publication_id":"P1","as_of":"2024-01-01","weights":{"A":1.0}},
        {"publication_id":"P2","as_of":"2024-01-03","weights":{"B":1.0}},
    ]
    result=versioned_model_nav(prices,publications)
    assert result.iloc[0]["nav"] == pytest.approx(110)
    assert result.iloc[1]["nav"] == pytest.approx(132)
    assert result.iloc[1]["publication_id"] == "P2"


def test_audit_multi_basket_isolation_and_tampering():
    assert verify_trust_audit([],"A")[0] is False
    wrong=[{"basket_id":"B","sequence_number":1,"previous_hash":None,"event_hash":"x","payload_json":{}}]
    assert verify_trust_audit(wrong,"A")[0] is False
