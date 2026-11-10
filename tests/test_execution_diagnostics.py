from copy import deepcopy
import json

import pytest
from execution_diagnostics import execution_position_rows, execution_summary_metric, SHOCK_PREFIX


def legacy_report():
    return {"passed": False, "violations": [SHOCK_PREFIX+"EXAMPLE", "Funding check failed"],
            "final_weights": {"EXAMPLE": .12}}


def test_summary_distinguishes_blocked_draft_holdings_from_executable_orders():
    report = {"passed":False, "violations":[SHOCK_PREFIX+f"S{i}" for i in range(8)]}
    report["violations"].append(SHOCK_PREFIX+"S0")  # Same holding is counted once.
    assert execution_summary_metric(report, 0) == ("Plan status", "Blocked", "8 holdings over cap · draft withheld")
    assert execution_summary_metric(report, 12)[1] == "Blocked"  # A stale count cannot imply approval.
    assert execution_summary_metric({"passed":False,"violations":["Funding failed"]})[1] == "Blocked"
    assert execution_summary_metric({})[1] == "Needs validation"
    assert execution_summary_metric({"passed":True}, 7)[:2] == ("Executable trades", 7)
    assert execution_summary_metric({"passed":True}, 0) == ("Executable trades", 0, "Checks passed · no executable orders")


def test_old_evidence_is_usable_without_a_solve_and_upper_ceiling_is_qualified():
    report = legacy_report()
    policy = {"limits": {"position_caps": {"EXAMPLE": .04}}}
    before = deepcopy((report, policy))
    rows = execution_position_rows(report, policy)
    assert rows == [{"ticker": "EXAMPLE", "weight": .12, "cap": .04,
                     "excess": pytest.approx(.08), "cap_basis": "saved ceiling"}]
    assert (report, policy) == before


def test_exact_checked_cap_takes_priority_over_saved_ceiling_and_survives_json():
    report = legacy_report()
    report["allocation_policy"] = {"position_breaches": [
        {"ticker": "EXAMPLE", "weight": .12, "cap": .03, "excess": .09}]}
    rows = execution_position_rows(json.loads(json.dumps(report)), {"limits": {"position_caps": {"EXAMPLE": .04}}})
    assert rows[0]["cap"] == .03 and rows[0]["excess"] == pytest.approx(.09)
    assert rows[0]["cap_basis"] == "checked"
    # A holdings/pause change replaces the reasons, invalidating old evidence.
    report["violations"] = ["Holdings changed"]
    assert execution_position_rows(report) == []


@pytest.mark.parametrize("cap", [None, float('nan'), float('inf'), .2, -1, 'missing'])
def test_legacy_missing_or_non_explanatory_ceiling_does_not_invent_a_cap(cap):
    rows = execution_position_rows(legacy_report(), {"limits": {"position_caps": {"EXAMPLE": cap}}})
    assert rows[0]["weight"] == .12
    assert rows[0]["cap"] is rows[0]["excess"] is None
    assert rows[0]["cap_basis"] == "unavailable"


def test_ui_legacy_refresh_explains_sell_breach_formats_percentages_and_keeps_other_failures():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("tests/execution_conflict_ui_fixture.py", default_timeout=10).run()
    assert not app.exception
    assert app.metric[0].label == "Plan status" and app.metric[0].value == "Blocked"
    assert any("1 holding over cap" in item.value for item in app.caption)
    assert any("No new allocation was optimized" in item.value for item in app.warning)
    assert not any("Blocked while not recorded" in item.value or "allocator solve" in item.value for item in app.caption)
    assert any("keeps these weights" in item.value for item in app.caption)
    assert any("partial sale" in item.value for item in app.caption)
    assert any("retained upper ceilings" in item.value for item in app.caption)
    assert any("Funding check failed" in item.value for item in app.markdown)
    assert not any(SHOCK_PREFIX in item.value for item in app.markdown)
    table = app.dataframe[0].value
    assert table.iloc[0]["After draft (%)"] == 12.
    assert table.iloc[0]["Recorded cap (%)"] == 4.
    assert table.iloc[0]["Excess (pp)"] == pytest.approx(8.)
    draft = app.dataframe[1].value
    assert draft.iloc[0]["Current allocation (%)"] == 20.
    assert draft.iloc[0]["Saved allocation (%)"] == 12.
    assert draft.iloc[0]["Proposed quantity"] == 8.
    assert not list(app.button)  # Display cannot trigger an optimizer or a trade.


def test_prior_failed_refresh_is_not_described_as_a_new_failed_optimization():
    from streamlit.testing.v1 import AppTest
    source = '''from portfolio_ui import render_allocation_policy
render_allocation_policy({'status':'failed_recheck', 'candidate':{'passed':False},
    'limits':{'history_through':'2026-11-01'}, 'constraints_applied':False})
'''
    app = AppTest.from_string(source).run()
    assert not app.exception
    assert any("No new allocation was optimized" in item.value for item in app.warning)
    assert any("optimizer not run" in item.value for item in app.caption)


def test_non_shock_failure_and_success_preserve_their_messages():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file("tests/allocation_integrity_ui_fixture.py").run()
    assert not app.exception
    assert any("Missing final holding history" in item.value for item in app.markdown)
    app.segmented_control[0].set_value("Passed").run()
    assert not app.exception
    assert len(app.success) == 1 and not list(app.dataframe)
