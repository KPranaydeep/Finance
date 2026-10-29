import ast
from pathlib import Path
from types import SimpleNamespace


SOURCE_PATH = Path("portfolio_rebalancer_database.py")


def _load_apply_pair():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_best_search_pair"
    )
    namespace = {"st": SimpleNamespace(session_state={})}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<apply-pair>", "exec"),
        namespace,
    )
    return namespace["_apply_best_search_pair"], namespace["st"].session_state


def test_raw_peak_button_updates_both_warm_start_coordinates():
    apply_pair, state = _load_apply_pair()

    apply_pair(8300, 550)

    assert state["shortlist_cap_search_start_x"] == 8300
    assert state["shortlist_cap_search_start_y"] == 550
    assert state["universal_preselection_cap"] == 8300
    assert state["exact_optimizer_asset_cap"] == 550


def test_live_widgets_are_not_overwritten_from_past_best_on_every_rerun():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "past_best = best_feasible_result" not in source
    assert 'if "shortlist_cap_search_start_x" not in st.session_state:' in source
    assert 'if "shortlist_cap_search_start_y" not in st.session_state:' in source


def test_latest_checkpoint_raw_peak_precedes_cross_checkpoint_fallback():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    current_peak = 'best_feasible_result(restored_checkpoint["results"])'

    assert current_peak in source
    assert source.index(current_peak) < source.index("or saved_best_result", source.index(current_peak))


def test_grid_edits_preserve_results_from_same_analysis_snapshot():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_search_results_are_compatible"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<compatible>", "exec"),
        namespace,
    )
    compatible = namespace["_search_results_are_compatible"]
    previous = {
        "max_dd": 0.05,
        "target_volatility": None,
        "drop_bottom_pct": 0.2,
        "history_buffer_days": 0,
        "redundancy_corr_threshold": 0.8,
        "minimum_trading_days": 252,
        "snapshot_date_ist": "2026-10-14",
        "optimizer_config_version": "portfolio-optimizer-config-v3",
        "risk_free_rate_annual": 0.08225,
        "step": 50,
    }
    current = {**previous, "step": 100, "starting_cap": 8300}

    assert compatible(previous, current)
    assert not compatible(
        previous, {**current, "snapshot_date_ist": "2026-10-15"}
    )
    assert not compatible(
        previous, {**current, "risk_free_rate_annual": 0.112}
    )


def test_saturated_adaptive_search_hands_raw_peak_to_final_optimization():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert 'search_rows[-1].get("Search saturated")' in source
    assert "highest_return_pair = best_feasible_result(search_rows)" in source
    assert "automatic_prepared_analysis = validation_matrix_cache.get(" in source
    assert "run_btn = True" in source
    assert 'st.session_state["pending_automatic_best_pair"] =' not in source
