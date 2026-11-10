from contextlib import contextmanager
from copy import deepcopy
import sqlite3

import pytest

from allocation_attempts import load_workspace, save_controls, save_attempt, failure_summary
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY


def test_controls_and_failures_are_separate_and_profile_scoped():
    conn = sqlite3.connect(":memory:")
    policy = dict(DEFAULT_ALLOCATION_POLICY)
    save_controls(conn, "a", policy)
    failed = {"status": "unavailable", "reason": "Owned history missing", "stage": "current_allocation_inputs"}
    save_attempt(conn, "a", failed)
    restored = load_workspace(conn, "a")
    timestamp = restored["attempt"]["recorded_at"]
    assert restored["controls"] == policy
    save_controls(conn, "a", dict(policy, single_name_loss_budget=.03))
    assert load_workspace(conn, "a")["attempt"]["recorded_at"] == timestamp
    assert load_workspace(conn, "b") == {}
    with pytest.raises(ValueError):
        save_controls(conn, "a", dict(policy, group_weight_limit=2.))
    assert load_workspace(conn, "a")["controls"]["group_weight_limit"] == .25
    conn.close()


def test_failure_categories_do_not_call_input_errors_risk_infeasibility():
    assert failure_summary({"status": "unavailable", "reason": "Missing owned history"}).startswith("Allocation inputs unavailable:")
    assert "time budget" in failure_summary({"status": "time_limit"})
    assert "No particular constraint has been diagnosed" in failure_summary({})


def test_controls_apply_without_optimizer_and_survive_new_session(tmp_path):
    from streamlit.testing.v1 import AppTest
    path = str(tmp_path / "fixture.sqlite")
    source = '''import sqlite3
from contextlib import contextmanager
import streamlit as st
from allocation_policy_ui import render_policy_controls, render_last_attempt
from allocation_policy_config import DEFAULT_ALLOCATION_POLICY
@contextmanager
def connect():
    conn = sqlite3.connect(PATH)
    try:
        yield conn
    finally:
        conn.close()
p = render_policy_controls(DEFAULT_ALLOCATION_POLICY, 'synthetic', connect)
st.metric('Applied loss budget', p['single_name_loss_budget'])
render_last_attempt('synthetic', connect)
'''.replace("PATH", repr(path))
    app = AppTest.from_string(source, default_timeout=10).run()
    assert not app.exception
    assert app.metric[0].value == "0.02"
    app.number_input[0].set_value(3.)
    app.run()
    assert app.metric[0].value == "0.02"  # Draft form values have not been applied.
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "0.03"
    second = AppTest.from_string(source, default_timeout=10).run()
    assert second.number_input[0].value == 3.
    assert any("6.00%" in item.value for item in second.markdown)
    conn = sqlite3.connect(path)
    save_attempt(conn, "synthetic", {"status":"unavailable", "reason":"Owned history missing", "stage":"current_allocation_inputs"})
    conn.close()
    second.run()
    assert any("Allocation inputs unavailable" in item.value for item in second.warning)
    disabled_source = source.replace("render_policy_controls(DEFAULT_ALLOCATION_POLICY,", "render_policy_controls(dict(DEFAULT_ALLOCATION_POLICY, enabled=False),")
    disabled = AppTest.from_string(disabled_source, default_timeout=10).run()
    assert not disabled.exception
    assert not list(disabled.number_input)
    assert any("disabled" in item.value for item in disabled.info)
