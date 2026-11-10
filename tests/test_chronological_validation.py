import json
import time

import numpy as np
import pandas as pd
import pytest

from chronological_validation import (ResearchPause, comparison_summary, experiment_id, experiment_lock, atomic_save, training_seed,
                                       fold_windows, run_validation, simulate_fold)
from chronological_pipeline import closed_session_marks, make_pipeline_fitter
from portfolio_ui import empty_trade_plan_message


def inputs():
    days = pd.bdate_range("2024-01-01", periods=130)
    prices = pd.DataFrame({"A": 100 * np.exp(np.arange(130) * .001),
                           "B": 100 * np.exp(np.arange(130) * -.0005)}, index=days)
    volume = prices * 100
    metadata = [{"Yahoo Ticker": t, "Currency": "INR", "Exchange": "NSE"} for t in prices]
    settings = {"minimum_train": 60, "test_sessions": 30, "folds": 2, "cost_bps": 22.}
    return prices, volume, metadata, settings


def test_outer_fit_receives_only_prefix_and_future_outcomes_cannot_change_choice(tmp_path):
    prices, volumes, metadata, settings = inputs()
    seen = []

    def fit(train, volume, meta, initial, strategy, trials, save, deadline, progress):
        seen.append((strategy, train.index[-1], list(train.columns), float(train["A"].iloc[-1])))
        assert volume.index.equals(train.index)
        return {"weights": {"A": 1.}, "status": "selected", "pair": [50, 50]}

    result = run_validation(prices, volumes, metadata, settings, fit, tmp_path / "original.json")
    changed = prices.copy()
    changed.iloc[70:, 0] *= 2
    first = seen[:2]
    seen.clear()
    second = run_validation(changed, volumes, metadata, settings, fit, tmp_path / "altered.json")
    assert seen[:2] == first
    assert result["status"] == second["status"] == "complete"
    assert result["folds"][0]["net_return"] != second["folds"][0]["net_return"]
    assert result["summary"]["auto_promote"] is False


def test_next_close_execution_does_not_capture_overnight_move_before_the_fill():
    days = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"A": [100., 100., 110.], "B": [100., 200., 200.]}, index=days)
    result = simulate_fold(prices, {"A": 1.}, {"B": 1.}, 0.)
    assert result["net_return"] == pytest.approx(0.)
    assert result["execution_date"] == str(days[1].date())
    assert result["gross_turnover"] == pytest.approx(2.)
    costed = simulate_fold(prices, {"A": 1.}, {"B": 1.}, 100.)
    assert costed["cost_fraction"] == pytest.approx(.02 / 1.01)
    assert costed["net_return"] == pytest.approx(-.02 / 1.01)
    assert list(result["ending_weights"]) == ["B"]
    # A security already fully exited does not require later valuation marks.
    prices.loc[days[2], "A"] = np.nan
    assert simulate_fold(prices, {"A": 1.}, {"B": 1.}, 0.)["net_return"] == 0.


def test_resume_preserves_completed_folds_and_trial_work_across_instances(tmp_path):
    prices, volumes, metadata, settings = inputs()
    checkpoint = tmp_path / "resume.json"
    calls = []

    def fit(train, volume, meta, initial, strategy, trials, save, deadline, progress):
        if strategy == "risk-aware" and "partial" not in trials:
            trials["partial"] = {"done": True}
            save()
            raise ResearchPause()
        calls.append((str(train.index[-1]), strategy))
        return {"weights": initial, "status": "selected"}

    partial = run_validation(prices, volumes, metadata, settings, fit, checkpoint)
    assert partial["status"] == "paused" and len(partial["folds"]) == 1
    assert json.loads(checkpoint.read_text())["trials"]["1:risk-aware"]["partial"]["done"]
    while partial["status"] != "complete":
        partial = run_validation(prices, volumes, metadata, settings, fit, checkpoint)
    assert len(calls) == len(set(calls)) == 4
    before = list(calls)
    run_validation(prices, volumes, metadata, settings, fit, checkpoint)
    assert calls == before


def test_changed_evidence_or_config_does_not_overwrite_old_checkpoint(tmp_path):
    prices, volumes, metadata, settings = inputs()
    checkpoint = tmp_path / "fixed.json"
    result = run_validation(prices, volumes, metadata, settings, lambda *a: {"weights": a[3]}, checkpoint)
    original = checkpoint.read_bytes()
    altered = prices.copy()
    altered.iloc[-1, 0] += 1
    with pytest.raises(ValueError, match="changed"):
        run_validation(altered, volumes, metadata, settings, lambda *a: {}, checkpoint)
    assert checkpoint.read_bytes() == original
    assert experiment_id(prices, volumes, metadata, settings) != experiment_id(prices, volumes, metadata, dict(settings, cost_bps=30))


def test_missing_selected_or_owned_test_price_withholds_instead_of_dropping_loser(tmp_path):
    prices, volumes, metadata, settings = inputs()
    prices.iloc[75, 0] = np.nan
    state = run_validation(prices, volumes, metadata, settings,
                           lambda *a: {"weights": {"A": 1.}, "status": "selected"}, tmp_path / "missing.json")
    assert state["status"] == "blocked"
    assert state["folds"] == []
    assert "missing" in state["error"].lower()
    assert state["summary"]["status"] == "incomplete"


def test_no_feasible_fit_is_held_and_reported_without_hiding_the_fold(tmp_path):
    prices, volumes, metadata, settings = inputs()
    state = run_validation(prices, volumes, metadata, settings,
                           lambda *a: {"status": "held", "reason": "infeasible"}, tmp_path / "held.json")
    assert state["status"] == "complete" and len(state["folds"]) == 4
    assert all(row["fit_status"] == "held" and row["reason"] == "infeasible" for row in state["folds"])
    assert state["summary"]["strategies"]["risk-aware"]["held_folds"] == 2
    assert state["summary"]["decision"].startswith("No consistent")


def test_expanding_folds_are_nonoverlapping_and_chronological():
    windows = fold_windows(pd.bdate_range("2024-01-01", periods=342))
    assert windows == [(252, 282), (282, 312), (312, 342)]
    with pytest.raises(ValueError, match="Need"):
        fold_windows(pd.bdate_range("2024-01-01", periods=341))


def test_closed_day_carry_never_fills_missing_open_quote_or_earlier_start():
    days = pd.date_range("2026-01-01", periods=7)
    raw = pd.Series([10., 11., np.nan], index=days[[1, 2, 5]])
    opened = days[[0, 1, 2, 5, 6]]
    marked = closed_session_marks(raw, days, opened)
    assert pd.isna(marked.iloc[0])
    assert marked.iloc[3] == marked.iloc[4] == 11.
    assert pd.isna(marked.iloc[5]) and pd.isna(marked.iloc[6])


def test_empty_trade_plan_explains_actual_cause():
    assert "needs validation" in empty_trade_plan_message(30, {})
    assert "Funding" in empty_trade_plan_message(30, {"passed": False, "violations": ["Funding unavailable"]})
    assert "All 3" in empty_trade_plan_message(3, {"passed": True})
    assert "paused" not in empty_trade_plan_message(0)


def test_fx_cannot_use_a_future_first_rate_for_an_earlier_price():
    from scalable_universe_preselection import convert_candidate_history_to_inr
    import portfolio_optimizer_core as core
    dates = pd.bdate_range("2026-01-01", periods=4)
    prices = pd.DataFrame({"US": [10.,11.,12.,13.]}, index=dates)
    fx = pd.Series([90.,91.], index=dates[2:])
    converted, _ = convert_candidate_history_to_inr(prices, {"US": "USD"}, {"USD": fx})
    assert converted["US"].iloc[:2].isna().all()
    assert converted["US"].iloc[2] == 1080.
    full = core.convert_price_history_to_inr(prices, (("US","USD"),), fx_histories={"USD":fx})
    assert full["US"].iloc[:2].isna().all()
    prefix, omitted = convert_candidate_history_to_inr(prices.iloc[:2], {"US":"USD"}, {"USD":fx.loc[:dates[1]]})
    assert "US" not in prefix and omitted == ["USD"]


def test_zero_runtime_does_not_start_a_fit_and_keeps_a_checkpoint(tmp_path):
    prices, volumes, metadata, settings = inputs()
    checkpoint = tmp_path / "paused.json"
    state = run_validation(prices, volumes, metadata, settings,
                           lambda *a: pytest.fail("fit started after deadline"), checkpoint, runtime_seconds=0)
    assert state["status"] == "paused" and checkpoint.exists()


def test_two_windows_cannot_advance_the_same_experiment_concurrently(tmp_path):
    path = tmp_path / "experiment.json"
    with experiment_lock(path):
        with pytest.raises(ValueError, match="already running"):
            with experiment_lock(path):
                pytest.fail("Second writer acquired the run lock")
    with experiment_lock(path):
        pass  # Releasing/terminating the first run does not leave a stale lock.


def test_atomic_checkpoint_survives_transient_windows_reader_lock(monkeypatch, tmp_path):
    import chronological_validation as module
    real_replace = module.os.replace
    calls = []

    def brief_lock(source, target):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("Transient reader")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", brief_lock)
    target = tmp_path / "atomic.json"
    atomic_save(target, {"complete": 3})
    assert json.loads(target.read_text()) == {"complete": 3} and len(calls) == 2


def test_cached_panel_keeps_missing_open_fx_unknown_and_handles_quoted_pence(monkeypatch):
    import chronological_pipeline as module
    import portfolio_optimizer_core as core
    days = pd.bdate_range("2026-01-05", periods=5)
    cached = pd.DataFrame({"A.NS": [100.]*5, "B": [10.]*5, "C.L": [100.]*5,
                           "INR=X": [90.,90.,np.nan,90.,90.], "GBPINR=X": [100.]*5}, index=days)
    monkeypatch.setattr(module, "load_cached_market_history", lambda *a: (cached, cached*100))
    universe = pd.DataFrame([
        {"Yahoo Ticker":"A.NS","Exchange":"NSE","Currency":"INR"},
        {"Yahoo Ticker":"B","Exchange":"NYQ","Currency":"USD"},
        {"Yahoo Ticker":"C.L","Exchange":"LSE","Currency":"GBp"}])
    panel, _, _, report = module.cached_research_panel("unused", universe, days[-1],
                                                       currency_unit=core._currency_major_and_unit_factor)
    assert pd.isna(panel.loc[days[2], "B"])
    assert panel.loc[days[1], "B"] == 900.
    assert panel.loc[days[1], "C.L"] == 100.  # 100 pence = GBP1, not GBP100.
    assert report["included"] == 3 and report["omitted"] == 0 and report["network_requests"] == 0


def test_common_starting_basket_uses_training_liquidity_and_keeps_listing_representation():
    days = pd.bdate_range("2025-01-01", periods=80)
    names = [f"US{i:02}" for i in range(70)] + ["IN.NS", "BENCH"]
    prices = pd.DataFrame(100., index=days, columns=names)
    volumes = pd.DataFrame(10., index=days, columns=names)
    volumes["US69"] = 10000.
    metadata = [{"Yahoo Ticker": t, "Exchange":"NYSE", "Currency":"USD"} for t in names[:-2]] + [
        {"Yahoo Ticker":"IN.NS", "Exchange":"NSE", "Currency":"INR"},
        {"Yahoo Ticker":"BENCH", "Exchange":"NYSE", "Currency":"USD", "research_role":"benchmark"}]
    initial = training_seed(prices, volumes, metadata, 60)
    assert len(initial) == 50 and "IN.NS" in initial and "US69" in initial
    assert "BENCH" not in initial and sum(initial.values()) == pytest.approx(1.)


def test_production_funnel_and_recursive_search_are_offline_and_trial_fits_reused(monkeypatch):
    import portfolio_optimizer_core as core
    ns = core.search_universal_shortlist_caps.__globals__
    for name in ("build_current_allocation_from_db", "extend_allocation_with_universal_candidates",
                 "sync_market_history", "_yf_download_quiet", "download_fx_history_to_inr"):
        monkeypatch.setitem(ns, name, lambda *a, **k: pytest.fail("Live data/holdings used by a historical fit"))
    monkeypatch.setitem(ns, "optimize_portfolio_max_return_given_daily_risk", lambda r, **k: np.repeat(1/len(r.columns), len(r.columns)))
    rng = np.random.default_rng(4)
    train = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.001,.008,(310,60)), axis=0)),
                         index=pd.bdate_range("2024-01-01", periods=310), columns=[f"T{i:02}" for i in range(60)])
    volume = pd.DataFrame(100000., index=train.index, columns=train.columns)
    metadata = [{"Yahoo Ticker": t, "Currency": "INR", "Exchange": "NSE"} for t in train]
    settings = {"trials_per_fold": 3, "solver_seconds": 1., "cost_bps": 22., "search": {
        "minimum_cap": 50, "maximum_cap": 100, "step": 50, "adaptive": True, "max_dd": .9,
        "target_volatility": None, "drop_bottom_pct": 0., "redundancy_corr_threshold": .99,
        "exact_optimizer_asset_caps": [30,50], "minimum_trading_days": 252}}
    initial = {t: 1/30 for t in train.columns[:30]}
    trials = {}
    fitter = make_pipeline_fitter(core, settings)
    first = fitter(train, volume, metadata, initial, "incumbent", trials, lambda: None, time.monotonic()+30, None)
    assert first["status"] == "selected" and 1 <= len(trials) <= 3
    risk_trials = {}
    risk = fitter(train, volume, metadata, initial, "risk-aware", risk_trials, lambda: None, time.monotonic()+30, None)
    assert risk["status"] == "selected"
    assert all(r["policy_constraints_applied"] for r in risk_trials.values() if r["weights"])
    assert core.chronological_research_config()["allocation_policy"]["enabled"] is True
    saved = json.dumps(trials)
    monkeypatch.setattr(core, "optimize_prepared_return_matrix", lambda *a, **k: pytest.fail("Completed trial refit"))
    second = fitter(train, volume, metadata, initial, "incumbent", trials, lambda: None, time.monotonic()+30, None)
    assert first == second and json.dumps(trials) == saved


def test_research_report_displays_net_results_and_scope_without_promoting():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).with_name("chronological_validation_ui_fixture.py"))).run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["+4.50%", "+5.00%"]
    assert any("No consistent improvement" in i.value for i in app.info)
    assert any("No automatic promotion" in i.value for i in app.caption)


def test_research_controls_do_not_load_universe_or_fit_on_page_render(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import chronological_validation_ui as ui
    monkeypatch.setattr(ui, "research_universe", lambda *a: pytest.fail("Universe loaded on page render"))
    source = "from chronological_validation_ui import render_chronological_validation\nrender_chronological_validation('synthetic', " + repr(str(tmp_path / "db.sqlite")) + ", " + repr(str(tmp_path / "cache.duckdb")) + ", {})"
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert app.button[0].label == "Run / resume chronological test"
