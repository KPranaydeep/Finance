import json

import pytest

from portfolio_optimizer_config import load_optimizer_config


def valid_config():
    return {
        "config_version": "test-config-v1",
        "risk_free_rate_annual": 0.08225,
        "trading_days_per_year": 250,
        "max_weight_per_asset": 0.5,
        "history_start_date": "2000-01-01",
        "momentum_filter": {
            "enabled": True,
            "method_version": "robust-momentum-v1",
            "maximum_exclusion_fraction": 0.20,
            "lookback_sessions": [63, 126, 252],
            "skip_recent_sessions": 21,
            "stability_checkpoint_sessions": [0, 21, 42],
            "trend_lookback_sessions": 200,
            "minimum_negative_horizons": 2,
            "annualization_sessions": 250,
            "volatility_floor_annual": 0.05,
            "apply_exclusion_to_owned_holdings": False,
        },
    }


def write(tmp_path, payload):
    path = tmp_path / "optimizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_and_normalizes_valid_configuration(tmp_path):
    result = load_optimizer_config(write(tmp_path, valid_config()))
    assert result["risk_free_rate_annual"] == 0.08225
    assert result["max_weight_per_asset"] == 0.5
    assert result["history_start_date"] == "2000-01-01"
    assert result["momentum_filter"]["maximum_exclusion_fraction"] == 0.20
    assert result["momentum_filter"]["method_version"] == "robust-momentum-v1"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("risk_free_rate_annual", -0.01, "RISK_FREE_RATE"),
        ("risk_free_rate_annual", 1.0, "RISK_FREE_RATE"),
        ("trading_days_per_year", 0, "TRADING_DAYS"),
        ("max_weight_per_asset", 1.1, "MAX_WEIGHT"),
        ("history_start_date", "not-a-date", "HISTORY_START"),
    ],
)
def test_rejects_invalid_values(tmp_path, key, value, message):
    payload = valid_config()
    payload[key] = value
    with pytest.raises(ValueError, match=message):
        load_optimizer_config(write(tmp_path, payload))


def test_rejects_unknown_or_missing_keys(tmp_path):
    payload = valid_config()
    payload["silent_new_knob"] = True
    with pytest.raises(ValueError, match="KEYS_INVALID"):
        load_optimizer_config(write(tmp_path, payload))


def test_rejects_momentum_exclusion_above_twenty_percent(tmp_path):
    payload = valid_config()
    payload["momentum_filter"]["maximum_exclusion_fraction"] = 0.21
    with pytest.raises(ValueError, match="MOMENTUM_MAXIMUM_EXCLUSION"):
        load_optimizer_config(write(tmp_path, payload))


def test_requires_owned_holding_protection(tmp_path):
    payload = valid_config()
    payload["momentum_filter"]["apply_exclusion_to_owned_holdings"] = True
    with pytest.raises(ValueError, match="OWNED_PROTECTION_REQUIRED"):
        load_optimizer_config(write(tmp_path, payload))


def test_ui_free_optimizer_uses_the_shared_configuration():
    import portfolio_optimizer_core as core

    assert core._CORE["RISK_FREE_RATE_ANNUAL"] == 0.08225
    assert core._CORE["TRADING_DAYS_PER_YEAR"] == 250
    assert core._CORE["MAX_WEIGHT_PER_ASSET"] == 0.5
    assert core._CORE["DEFAULT_HISTORY_START_DATE"] == "2000-01-01"
    assert core._CORE["ALLOCATION_POLICY"]["enabled"]


def test_old_external_config_does_not_silently_enable_new_risk_assumptions(tmp_path):
    result = load_optimizer_config(write(tmp_path,valid_config()))
    assert result["allocation_policy"]["enabled"] is False


def test_explicit_allocation_policy_is_validated(tmp_path):
    from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
    payload=valid_config()
    payload["allocation_policy"]=dict(DEFAULT_ALLOCATION_POLICY)
    result=load_optimizer_config(write(tmp_path,payload))
    assert result["allocation_policy"]["single_name_loss_budget"] == .02
    payload["allocation_policy"]["confidence"] = 1.
    with pytest.raises(ValueError,match="ALLOCATION_POLICY"):
        load_optimizer_config(write(tmp_path,payload))


@pytest.mark.parametrize("value",[-20.,101.,float("nan"),float("inf")])
def test_event_adapter_rejects_invalid_drawdown_magnitudes(value):
    from public_basket_optimizer_adapter import _validated_settings
    with pytest.raises(ValueError,match="max_drawdown_input_pct"):
        _validated_settings({"settings":{"max_drawdown_input_pct":value}})


def test_event_adapter_uses_same_positive_drawdown_magnitude_as_core():
    from public_basket_optimizer_adapter import _validated_settings
    assert _validated_settings({"settings":{"max_drawdown_input_pct":20}})["internal_max_dd"] == .2
