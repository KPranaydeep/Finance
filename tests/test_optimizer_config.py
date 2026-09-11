import json

import pytest

from portfolio_optimizer_config import load_optimizer_config


def valid_config():
    return {
        "config_version": "test-config-v1",
        "risk_free_rate_annual": 0.112,
        "trading_days_per_year": 250,
        "max_weight_per_asset": 0.5,
        "history_start_date": "2000-01-01",
    }


def write(tmp_path, payload):
    path = tmp_path / "optimizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_and_normalizes_valid_configuration(tmp_path):
    result = load_optimizer_config(write(tmp_path, valid_config()))
    assert result["risk_free_rate_annual"] == 0.112
    assert result["max_weight_per_asset"] == 0.5
    assert result["history_start_date"] == "2000-01-01"


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


def test_ui_free_optimizer_uses_the_shared_configuration():
    import portfolio_optimizer_core as core

    assert core._CORE["RISK_FREE_RATE_ANNUAL"] == 0.112
    assert core._CORE["TRADING_DAYS_PER_YEAR"] == 250
    assert core._CORE["MAX_WEIGHT_PER_ASSET"] == 0.5
    assert core._CORE["DEFAULT_HISTORY_START_DATE"] == "2000-01-01"
