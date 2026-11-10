"""Exercise the real click callback without running the private app/optimizer."""
import ast
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from shortlist_search_checkpoint import load_latest_checkpoint, save_checkpoint
from two_dimensional_search import best_feasible_result


SOURCE = Path("portfolio_rebalancer_database.py")


def row(x, y, value, status="Feasible"):
    return {"Shortlist cap": x, "Maximum assets": y, "Annual Return": value,
            "Status": status}


def load_callback(connection=None, checkpoint=None, error=None, streamlit=None):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_start_optimization_at_raw_peak")
    state = {"universal_preselection_cap": 400, "exact_optimizer_asset_cap": 300,
             "shortlist_cap_search_start_x": 7700, "shortlist_cap_search_start_y": 600,
             "shortlist_cap_search_results": [row(9000, 900, .99)]}
    calls = []

    @contextmanager
    def database():
        if error:
            raise error
        yield connection

    def latest(conn, owner):
        calls.append(("load", owner))
        return load_latest_checkpoint(conn, owner) if conn is not None else checkpoint

    st = streamlit or SimpleNamespace(session_state=state)
    namespace = {"st": st, "sqlite3": sqlite3,
                 "get_db_connection": database, "load_latest_checkpoint": latest,
                 "best_feasible_result": best_feasible_result,
                 "_app_run_log": lambda event, **kw: calls.append((event, kw)),
                 "_start_optimization_plan_timer": lambda owner: calls.append(("timer", owner))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<raw-peak>", "exec"), namespace)
    return namespace[node.name], st.session_state, calls


def test_click_uses_latest_profile_raw_peak_not_old_or_session_winners():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # The all-time winner, other profile and near-equal robust alternative must
    # never override the raw peak of this profile's latest displayed checkpoint.
    save_checkpoint(conn, "alice", {"day": 1}, [row(9400, 1800, .92)])
    save_checkpoint(conn, "alice", {"day": 2}, [row(8300, 600, .512), row(8350, 550, .513)])
    save_checkpoint(conn, "bob", {"day": 3}, [row(9000, 2000, 1.0)])
    callback, state, calls = load_callback(conn)
    preserved = {k: v for k, v in state.items() if k.startswith("shortlist_")}
    callback("alice")
    assert (state["universal_preselection_cap"], state["exact_optimizer_asset_cap"]) == (8350, 550)
    assert {k: v for k, v in state.items() if k.startswith("shortlist_")} == preserved
    assert state["optimization_raw_peak_selection"]["source"] == "saved_raw_peak"
    assert calls[0] == ("load", "alice")
    assert calls[-1] == ("timer", "alice")
    assert calls[1][1]["search_started"] is False
    conn.close()


@pytest.mark.parametrize("rows", [[], [row(7700, 550, .5, "Infeasible")],
    [row(7700, 550, float("inf")), row(7700, 550, float("nan"))],
    [row(0, 550, .5), row(7700, float("inf"), .5)]])
def test_no_usable_saved_peak_preserves_manual_controls(rows):
    callback, state, calls = load_callback(checkpoint={"results": rows})
    callback("alice")
    assert (state["universal_preselection_cap"], state["exact_optimizer_asset_cap"]) == (400, 300)
    assert state["optimization_raw_peak_selection"]["source"] == "configured_controls"
    assert calls[-1] == ("timer", "alice")


def test_unavailable_checkpoint_does_not_destroy_controls_or_start_search():
    callback, state, calls = load_callback(error=sqlite3.OperationalError("unavailable"))
    callback("alice")
    assert state["universal_preselection_cap"] == 400
    assert state["exact_optimizer_asset_cap"] == 300
    assert state["optimization_raw_peak_selection"]["load_error"] == "OperationalError"
    assert calls[-1] == ("timer", "alice")


def test_smallest_supported_exact_limit_and_deterministic_raw_ties():
    callback, state, _ = load_callback(checkpoint={"results": [row(7800, 100, .5), row(7700, 50, .5)]})
    callback("alice")
    assert (state["universal_preselection_cap"], state["exact_optimizer_asset_cap"]) == (7700, 50)


def test_main_button_wired_to_callback_before_any_normal_run():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    button = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == "button"
                  and n.args and isinstance(n.args[0], ast.Constant)
                  and n.args[0].value == "Run optimization")
    options = {kw.arg: kw.value for kw in button.keywords}
    assert options["on_click"].id == "_start_optimization_at_raw_peak"
    assert options["args"].elts[0].id == "CURRENT_USER"


def test_streamlit_click_applies_xy_before_widget_construction(tmp_path):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("tests/raw_peak_ui_fixture.py").run()
    assert not app.exception
    app.button(key="run_optimization_btn_main").click().run()
    assert not app.exception
    assert app.number_input(key="universal_preselection_cap").value == 8350
    assert app.number_input(key="exact_optimizer_asset_cap").value == 550
    assert app.number_input(key="shortlist_cap_search_start_x").value == 7700
    assert app.info[0].value == "Using saved raw-return peak **X 8,350 × Y 550**. Refitting on current data."
