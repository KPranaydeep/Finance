from copy import deepcopy
from contextlib import nullcontext
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer_core as core
import fixed_weight_rebalance
from fixed_weight_rebalance import holdings_fingerprint, prepare_fixed_universe, rebuild_saved_plan, target_fingerprint


@pytest.fixture(autouse=True)
def legacy_policy_fixture(monkeypatch):
    # These small synthetic portfolios characterize plan refresh/precision.
    # Risk-budget activation has dedicated multi-asset integration tests.
    monkeypatch.setattr(fixed_weight_rebalance,"ALLOCATION_POLICY",{"enabled":False})


def fixtures():
    rows = [{"Symbol": t, "Yahoo Ticker": t, "Currency": "USD", "Quantity": q,
             "Average Price": 8, "FX to INR": 1, "Latest Price INR": 10, "Weight": q / 10}
            for t, q in [("A", 8), ("B", 2), ("OLD", 3)]]
    payload = {"saved_at": "2026-11-01T10:00:00+00:00", "run_id": "fixed-run",
               "optimal_allocation": [{"ticker": "A", "target_weight": .6}, {"ticker": "B", "target_weight": .4}],
               "current_allocation": rows, "settings": {"days_to_flip": 1}}
    owned = pd.DataFrame([dict(rows[0], Quantity=4.), dict(rows[1], Quantity=4.),
                          {"Symbol": "NEW", "Yahoo Ticker": "NEW", "Currency": "USD", "Quantity": 2., "Average Price": 9}])
    rng = np.random.default_rng(883)
    history = pd.DataFrame(rng.normal(.0005, .01, size=(300, 3)),
                           index=pd.bdate_range("2025-01-01", periods=300), columns=["NEW", "B", "A"])
    return payload, owned, history


def run(payload, owned, history, **kwargs):
    return rebuild_saved_plan(payload, owned, history, {"A": 10, "B": 10, "NEW": 10}, {"USD": 1},
                               plan_builder=core.rebalance_plan_multi, stats_builder=core.portfolio_stats, **kwargs)


def test_current_quantities_replace_old_positions_and_unowned_exits_disappear():
    payload, owned, history = fixtures()
    before = deepcopy(payload)
    # Any accidental optimizer call makes this financial regression fail.
    with patch.object(core, "run_portfolio_analysis_multi", side_effect=AssertionError("optimizer called")):
        candidate, allocation, plan = run(payload, owned, history)
    assert candidate["optimal_allocation"] == before["optimal_allocation"]
    assert candidate["run_id"] == before["run_id"] and candidate["saved_at"] == before["saved_at"]
    assert payload == before
    assert set(allocation["Yahoo Ticker"]) == {"A", "B", "NEW"}
    # The legacy 60% target is preserved as research, but cannot bypass the 50% cap.
    assert plan.empty and not candidate["execution_validation"]["passed"]
    indexed = pd.DataFrame(candidate["proposed_rebalancing_plan"]).set_index("Yahoo Ticker")
    assert indexed.loc["A", "Executable Quantity"] <= 1
    assert indexed.loc["NEW", "Action"] == "Sell" and indexed.loc["NEW", "Executable Quantity"] == 2
    assert indexed.loc["NEW", "Optimal Weight"] == 0
    assert candidate["current_portfolio_value"] == 100
    assert candidate["rebalance_plan_context"]["optimizer_run"] is False


def test_previously_owned_target_bought_again_starts_with_zero_current_quantity():
    payload, owned, _ = fixtures()
    universe, targets = prepare_fixed_universe(payload, owned.loc[owned["Yahoo Ticker"].ne("A")])
    assert universe.set_index("Yahoo Ticker").loc["A", "Quantity"] == 0
    assert targets == {"A": .6, "B": .4}


def test_paused_security_is_valued_but_not_traded_or_renormalized_out_of_target():
    payload, owned, history = fixtures()
    candidate, allocation, plan = run(payload, owned, history, paused=["NEW"])
    assert "NEW" in set(allocation["Yahoo Ticker"])
    assert "NEW" not in set(plan["Yahoo Ticker"])
    assert candidate["optimal_allocation"] == payload["optimal_allocation"]


def test_refresh_propagates_saved_frozen_holdings_and_optional_profit_controls():
    payload,owned,history=fixtures()
    payload["history"]={"price_integrity_frozen_owned_weights":{"A":.8}}
    calls=[]
    def builder(*args,**kwargs):
        calls.append(kwargs)
        return core.rebalance_plan_multi(*args,**kwargs)
    settings={"enabled":True,"minimum_net_profit_inr":8079.}
    rebuild_saved_plan(payload,owned,history,{"A":10.,"B":10.,"NEW":10.},{"USD":1.},
        plan_builder=builder,stats_builder=core.portfolio_stats,profit_exception_settings=settings)
    assert calls[0]["frozen_weights"]=={"A":.4}  # Current marked weight, not old .8.
    assert calls[0]["profit_exception_settings"]==settings


def test_fractional_exits_preserve_current_held_quantity():
    payload, owned, history = fixtures()
    owned.loc[owned["Yahoo Ticker"].eq("NEW"), "Quantity"] = .37
    _, _, plan = run(payload, owned, history)
    assert plan.set_index("Yahoo Ticker").loc["NEW", "Executable Quantity"] == pytest.approx(.37)


def test_missing_target_quote_fails_without_mutating_saved_evidence():
    payload, owned, history = fixtures()
    original = deepcopy(payload)
    with pytest.raises(ValueError, match="Missing or invalid current"):
        rebuild_saved_plan(payload, owned, history, {"A": 10, "NEW": 10}, {"USD": 1},
                           plan_builder=core.rebalance_plan_multi, stats_builder=core.portfolio_stats)
    assert payload == original


def test_invalid_weights_do_not_become_an_implicitly_normalized_new_target():
    payload, owned, _ = fixtures()
    payload["optimal_allocation"][0]["target_weight"] = .5
    with pytest.raises(ValueError, match="Complete saved optimal weights"):
        prepare_fixed_universe(payload, owned)


def test_fingerprints_detect_quantity_changes_but_ignore_order_and_market_marks():
    payload, owned, _ = fixtures()
    altered = owned.iloc[::-1].copy()
    altered["Latest Price INR"] = 999
    assert holdings_fingerprint(owned) == holdings_fingerprint(altered)
    altered.loc[altered.index[0], "Quantity"] += 1
    assert holdings_fingerprint(owned) != holdings_fingerprint(altered)
    assert target_fingerprint(payload) == target_fingerprint(deepcopy(payload))


def test_short_history_resizes_exits_without_pretending_risk_checks_pass():
    payload, owned, history = fixtures()
    payload["current_stats"] = {"old": 1}
    candidate, _, plan = run(payload, owned, history.iloc[:3])
    assert "NEW" in set(pd.DataFrame(candidate["proposed_rebalancing_plan"])["Yahoo Ticker"])
    assert candidate["current_stats"] == {"old": 1}
    assert "Earlier dated metrics retained" in candidate["portfolio_metrics_refresh_error"]
    assert "insufficient" in candidate["executable_rebalance_review"]["decision"].lower()


def configure_saved_refresh(monkeypatch):
    """Exercise the app's actual save path with synthetic data and no network/DB."""
    payload, owned, history = fixtures()
    ns = core.update_saved_rebalance_plan.__globals__
    saved = []
    ranking = pd.DataFrame({"Yahoo Ticker": ["A", "B", "NEW"], "Security": ["A", "B", "NEW"],
                            "Expected net scenario return (%)": [4., 2., 1.],
                            "Probability of positive net return (%)": [50., 50., 50.],
                            "Return × positive-probability score (%)": [2., 1., .5]})
    replacements = {
        "_app_run_log": lambda *a, **k: None,
        "build_current_allocation_from_db": lambda owner: (owned.copy(), []),
        "load_master_holdings": lambda owner: owned.copy(),
        "load_latest_analysis": lambda owner: payload,
        "refresh_security_scenarios": lambda *a, **k: (ranking.copy(), history, {}),
        "get_latest_price_map": lambda tickers: {t: 10 for t in tickers},
        "get_fx_to_inr_map": lambda currencies: {"USD": 1},
        "get_db_connection": lambda: nullcontext(None),
        "active_trade_deferrals": lambda *a: pd.DataFrame(),
        "save_latest_analysis": lambda p, owner: saved.append(p) or p,
        "run_portfolio_analysis_multi": lambda *a, **k: pytest.fail("optimizer called"),
        "prepare_saved_risk_repair": lambda *a, **k: {"status":"blocked"},
    }
    for name, value in replacements.items():
        monkeypatch.setitem(ns, name, value)
    return payload, owned, ns, saved


def test_app_refresh_persists_new_plan_and_marks_research_weights(monkeypatch):
    payload, _, ns, saved = configure_saved_refresh(monkeypatch)
    result = ns["update_saved_rebalance_plan"](payload, "synthetic-owner")
    assert saved == [result]
    assert result["optimal_allocation"] == payload["optimal_allocation"]
    assert result["rebalance_plan_context"]["optimizer_run"] is False
    research = pd.DataFrame(result["security_scenario_ranking"]).set_index("Yahoo Ticker")
    assert research.loc["A", "Current Weight (%)"] == 40
    assert research.loc["A", "Optimized Weight (%)"] == 60
    assert research.loc["NEW", "Current Weight (%)"] == 20
    assert set(pd.DataFrame(result["security_exit_ranking"])["Yahoo Ticker"]) == {"NEW"}


def test_app_refresh_rejects_concurrent_holdings_edit_before_saving(monkeypatch):
    payload, owned, ns, saved = configure_saved_refresh(monkeypatch)
    changed = owned.copy()
    changed.loc[0, "Quantity"] += 1
    reads = iter([owned, changed])
    monkeypatch.setitem(ns, "load_master_holdings", lambda owner: next(reads))
    with pytest.raises(ValueError, match="Holdings changed during refresh"):
        ns["update_saved_rebalance_plan"](payload, "synthetic-owner")
    assert not saved


def test_app_refresh_rejects_newer_optimizer_result_before_saving(monkeypatch):
    payload, _, ns, saved = configure_saved_refresh(monkeypatch)
    monkeypatch.setitem(ns, "load_latest_analysis", lambda owner: dict(payload, run_id="new-run"))
    with pytest.raises(ValueError, match="newer optimization"):
        ns["update_saved_rebalance_plan"](payload, "synthetic-owner")
    assert not saved


def test_app_refresh_does_not_silently_drop_invalid_owned_listing(monkeypatch):
    payload, owned, ns, saved = configure_saved_refresh(monkeypatch)
    monkeypatch.setitem(ns, "build_current_allocation_from_db", lambda owner: (owned.iloc[:2], ["NEW"]))
    with pytest.raises(ValueError, match="Some owned holdings"):
        ns["update_saved_rebalance_plan"](payload, "synthetic-owner")
    assert not saved
