from datetime import date

import numpy as np
import pytest

from public_portfolio_trust import allocation_turnover, bootstrap_outlook, common_price_window, fingerprint, forecast_calibration, performance_metrics, round_weights_to_whole_percent, versioned_model_nav, xirr
from public_portfolio_publications import validate_constituents, verify_trust_audit
from public_release_checks import inspect_public_data
from public_lumpsum_allocator import allocate_public_lumpsum, estimate_minimum_entry_capital
from production_smoke_test import current_forecast_is_due, filter_configured_backfill_findings
from public_portfolio_config import load_public_portfolio_config


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


def test_whole_percent_rounding_is_exact_and_deterministic():
    source=[{"ticker":"B","target_weight":.3333},{"ticker":"A","target_weight":.3333},{"ticker":"C","target_weight":.3334}]
    rounded=round_weights_to_whole_percent(source)
    assert rounded == [
        {"ticker":"C","target_weight":.34},
        {"ticker":"A","target_weight":.33},
        {"ticker":"B","target_weight":.33},
    ]
    assert sum(row["target_weight"] for row in rounded) == pytest.approx(1.0)


def test_zero_percent_positions_are_removed():
    source=[{"ticker":"BIG","target_weight":.999},{"ticker":"TINY","target_weight":.001}]
    assert round_weights_to_whole_percent(source) == [{"ticker":"BIG","target_weight":1.0}]


def test_versioned_nav_switches_immutable_portfolio_versions():
    import pandas as pd
    prices=pd.DataFrame({"A":[100,110,110],"B":[100,100,120]},index=pd.date_range("2024-01-01",periods=3))
    publications=[
        {"publication_id":"P1","as_of":"2024-01-01","weights":{"A":1.0}},
        {"publication_id":"P2","as_of":"2024-01-03","weights":{"B":1.0}},
    ]
    result=versioned_model_nav(prices,publications)
    assert result.iloc[0]["nav"] == pytest.approx(100)
    assert result.iloc[0]["daily_return"] == 0
    assert result.iloc[1]["nav"] == pytest.approx(110)
    assert result.iloc[2]["gross_nav"] == pytest.approx(110)
    assert result.iloc[2]["net_nav"] == pytest.approx(110*(1-.0022))
    assert result.iloc[2]["nav"] == result.iloc[2]["net_nav"]
    assert result.iloc[2]["turnover"] == pytest.approx(1.0)
    assert result.iloc[2]["estimated_drag"] == pytest.approx(.0022)
    assert result.iloc[2]["publication_id"] == "P2"


def test_common_price_window_uses_shared_constituent_history_and_requested_cap():
    prices=pd.DataFrame(
        {"A":[100,101,102,103,104,105],"B":[None,None,50,51,None,53],"OLD":[9,9,9,9,9,9]},
        index=pd.date_range("2024-01-01",periods=6,freq="B"),
    )
    result=common_price_window(prices,["A","B"],maximum_trading_days=2)
    assert list(result.columns) == ["A","B"]
    assert list(result.index) == list(prices.index[-3:])
    assert result.iloc[1]["B"] == 51


def test_common_price_window_stops_at_earliest_last_observation():
    prices=pd.DataFrame(
        {"A":[100,101,102,103],"B":[50,51,None,None]},
        index=pd.date_range("2024-01-01",periods=4,freq="B"),
    )
    result=common_price_window(prices,["A","B"],maximum_trading_days=10)
    assert result.index[-1] == prices.index[1]


def test_allocation_turnover_includes_cash_and_ignores_unchanged_weights():
    assert allocation_turnover({"A":.6,"B":.4},{"A":.6,"B":.4}) == 0
    assert allocation_turnover({"A":1.0},{"B":1.0}) == 1.0
    assert allocation_turnover({"A":.5},{"A":1.0}) == pytest.approx(.5)


def test_audit_multi_basket_isolation_and_tampering():
    assert verify_trust_audit([],"A")[0] is False
    wrong=[{"basket_id":"B","sequence_number":1,"previous_hash":None,"event_hash":"x","payload_json":{}}]
    assert verify_trust_audit(wrong,"A")[0] is False


def _audit_row(sequence, previous, payload, basket="A"):
    envelope={"basket_id":basket,"sequence_number":sequence,"entity_type":"publication",
              "entity_id":f"P{sequence}","event_type":"PUBLISHED","payload":payload}
    return {"basket_id":basket,"sequence_number":sequence,"previous_hash":previous or None,
            "payload_json":envelope,"event_hash":fingerprint({"previous_hash":previous,"event":envelope})}


def test_audit_valid_chain_and_corruption_detection():
    first=_audit_row(1,"",{"version":1})
    second=_audit_row(2,first["event_hash"],{"version":2})
    assert verify_trust_audit([first,second],"A")[0] is True
    changed=[dict(first),dict(second)]
    changed[1]["payload_json"]={**changed[1]["payload_json"],"payload":{"version":999}}
    assert verify_trust_audit(changed,"A")[0] is False
    broken=[dict(first),dict(second,previous_hash="bad-link")]
    assert verify_trust_audit(broken,"A")[0] is False
    wrong_hash=[dict(first,event_hash="0"*64),second]
    assert verify_trust_audit(wrong_hash,"A")[0] is False


def test_calibration_threshold_and_metrics():
    forecasts=[{"forecast_id":f"F{i}","median_return":.01,"lower_50":0,"upper_50":.02,"lower_90":-.02,"upper_90":.04} for i in range(20)]
    realizations=[{"forecast_id":f"F{i}","actual_return":.015} for i in range(20)]
    assert forecast_calibration(forecasts[:19],realizations)["sufficient"] is False
    result=forecast_calibration(forecasts,realizations)
    assert result["coverage_50"]==1 and result["coverage_90"]==1
    assert result["directional_accuracy"]==1


def test_public_data_security_inspection():
    assert inspect_public_data({"database_url":"postgresql://secret"})
    assert inspect_public_data({"path":"C:\\Users\\Someone\\private.csv"})
    assert inspect_public_data({"portfolio_version":"P001","weights":[.4,.6]}) == []


def test_small_lumpsum_uses_deterministic_starter_plan_without_zero_orders():
    target=[{"ticker":"A","target_weight":.5},{"ticker":"B","target_weight":.3},{"ticker":"C","target_weight":.2}]
    prices={"A":{"price":800},"B":{"price":240},"C":{"price":60}}
    first=allocate_public_lumpsum(target,prices,1000)
    second=allocate_public_lumpsum(target,prices,1000)
    assert first == second
    assert first["mode"] == "STARTER"
    assert first["orders"] and all(row["quantity"] > 0 for row in first["orders"])
    assert first["invested_inr"] + first["residual_cash_inr"] == pytest.approx(1000)
    assert first["invested_inr"] <= 1000


def test_lumpsum_reports_missing_prices_and_rejects_invalid_amount():
    target=[{"ticker":"A","target_weight":.5},{"ticker":"B","target_weight":.5}]
    result=allocate_public_lumpsum(target,{"A":100},1000)
    assert result["missing_prices"] == ["B"]
    with pytest.raises(ValueError): allocate_public_lumpsum(target,{"A":100},0)


def test_starter_subset_limits_orders_to_highest_target_weights():
    target=[{"ticker":"A","target_weight":.4},{"ticker":"B","target_weight":.3},
            {"ticker":"C","target_weight":.2},{"ticker":"D","target_weight":.1}]
    result=allocate_public_lumpsum(target,{"A":100,"B":80,"C":40,"D":20},10_000,starter_max_assets=2)
    assert result["mode"] == "STARTER_SUBSET"
    assert result["coverage"] == 2
    assert {row["ticker"] for row in result["orders"]} == {"A","B"}


def test_minimum_entry_uses_prices_costs_coverage_and_tracking():
    target=[{"ticker":"A","target_weight":.25},{"ticker":"B","target_weight":.25},
            {"ticker":"C","target_weight":.25},{"ticker":"D","target_weight":.25}]
    estimate=estimate_minimum_entry_capital(target,{"A":800,"B":240,"C":60,"D":120})
    assert estimate["constituent_count"] == 4
    assert estimate["minimum_price"] == 60 and estimate["maximum_price"] == 800
    assert estimate["coverage"] == 4
    assert estimate["estimated_execution_drag"] <= estimate["assumptions"]["maximum_execution_drag"]
    assert estimate["tracking_error_pp"] <= estimate["assumptions"]["maximum_tracking_error_pp"]
    assert estimate["minimum_capital_inr"] >= estimate["weighted_affordability_floor"]
    assert estimate["minimum_viable_starter_inr"] <= estimate["minimum_capital_inr"]
    starter=estimate["starter"]
    assumptions=estimate["assumptions"]
    assert starter["coverage"] >= assumptions["starter_required_coverage"]
    assert starter["invested_ratio"] >= assumptions["starter_minimum_invested_ratio"]
    assert starter["maximum_position_weight"] <= assumptions["starter_maximum_position_weight"]
    assert starter["target_weight_coverage"] >= assumptions["starter_minimum_target_coverage"]
    assert starter["estimated_execution_drag"] <= assumptions["maximum_execution_drag"]
    assert "bare_minimum_capital_inr" not in estimate


def test_forecast_smoke_check_only_becomes_strict_when_due_for_current_version():
    trust={"current":{"publication_id":"PUB-3"},"forecasts":[]}
    assert not current_forecast_is_due([{}]*60,trust)
    assert current_forecast_is_due([{}]*61,trust)
    trust["forecasts"]=[{"publication_id":"PUB-1"}]
    assert current_forecast_is_due([{}]*61,trust)
    trust["forecasts"].append({"publication_id":"PUB-3"})
    assert not current_forecast_is_due([{}]*61,trust)


def test_backfill_configuration_defaults_closed_and_validates_bounds(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODEL_BACKFILL_TRADING_DAYS",raising=False)
    assert load_public_portfolio_config().model_backfill_trading_days == 0
    monkeypatch.setenv("PUBLIC_MODEL_BACKFILL_TRADING_DAYS","120")
    assert load_public_portfolio_config().model_backfill_trading_days == 120
    monkeypatch.setenv("PUBLIC_MODEL_BACKFILL_TRADING_DAYS","-1")
    with pytest.raises(RuntimeError,match="BACKFILL"):
        load_public_portfolio_config()


def test_smoke_allows_only_configured_backfill_provenance_marker():
    trust={
        "active_forecasts":[{"forecast_json":{"history_source":"DEVELOPMENT_BACKFILL"}}],
        "forecasts":[{"forecast_json":{"history_source":"DEVELOPMENT_BACKFILL"}}],
    }
    expected=[
        "Non-production marker at $.active_forecasts[0].forecast_json.history_source",
        "Non-production marker at $.forecasts[0].forecast_json.history_source",
        "Non-production marker at $.current.strategy_version",
    ]
    assert filter_configured_backfill_findings(
        expected,trust,backfill_trading_days=120
    ) == ["Non-production marker at $.current.strategy_version"]


def test_smoke_backfill_exception_closes_at_zero_and_requires_exact_value():
    finding="Non-production marker at $.forecasts[0].forecast_json.history_source"
    trust={"forecasts":[{"forecast_json":{"history_source":"DEVELOPMENT_BACKFILL"}}]}
    assert filter_configured_backfill_findings([finding],trust,backfill_trading_days=0) == [finding]
    trust["forecasts"][0]["forecast_json"]["history_source"]="DEVELOPMENT_OTHER"
    assert filter_configured_backfill_findings([finding],trust,backfill_trading_days=120) == [finding]
