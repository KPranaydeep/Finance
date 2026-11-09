from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_withheld_plan_preserves_analysis_and_exposes_proposal_without_exception():
    app = AppTest.from_file(str(Path(__file__).with_name("allocation_integrity_ui_fixture.py"))).run()
    assert not app.exception
    assert app.metric[0].value == "15"
    assert "Plan withheld" in app.warning[0].value
    assert app.expander[0].label == "Withheld proposal — review only"
    app.segmented_control[0].set_value("Passed").run()
    assert not app.exception and app.success
    assert app.metric[0].value == "15"
    app.segmented_control[0].set_value("Older plan").run()
    assert not app.exception
    assert "without rerunning optimization" in app.info[0].value
