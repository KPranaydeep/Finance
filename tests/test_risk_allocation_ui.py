from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_risk_policy_panel_distinguishes_selected_retained_and_withheld():
    app=AppTest.from_file(str(Path(__file__).with_name("risk_allocation_ui_fixture.py"))).run()
    assert not app.exception
    assert any("Risk-aware allocator selected" in p.value for p in app.markdown)
    assert any("1.20%" in p.value and "3.50%" in p.value for p in app.caption)
    assert app.metric[0].value == "30"
    assert not app.expander[0].proto.expanded
    app.segmented_control[0].set_value("Incumbent").run()
    assert not app.exception
    assert any("Existing allocator retained" in p.value for p in app.markdown)
    app.segmented_control[0].set_value("Withheld").run()
    assert not app.exception
    assert "No allocation passed" in app.warning[0].value
    assert app.metric[0].value == "30"
