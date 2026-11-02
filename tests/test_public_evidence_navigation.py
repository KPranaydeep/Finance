from streamlit.testing.v1 import AppTest

from public_evidence_navigation import evidence_option_label, rank_evidence_rows


ROWS = [
    {"ticker": "LARGE.NS", "target_weight": .12, "entry_date": "2026-10-01", "status": "active"},
    {"ticker": "SMALL", "target_weight": .03, "entry_date": "2026-10-12", "status": "active"},
]


def picker_app(rows=ROWS):
    script = f'''
import streamlit as st
from public_evidence_navigation import render_evidence_picker
rows = st.session_state.get("replacement_rows", {rows!r})
scope = st.session_state.get("scope", "Current holdings")
selected = render_evidence_picker(rows, scope)
st.session_state["selected_ticker"] = selected["ticker"] if selected else None
'''
    return AppTest.from_string(script).run()


def test_named_options_and_adjacent_buttons_select_the_actual_security():
    app = picker_app()
    assert not app.exception
    assert app.selectbox[0].options == ["Portfolio overview",
        "LARGE.NS · allocation 12% · since 01 Oct 2026", "SMALL · allocation 3% · since 12 Oct 2026"]
    assert app.button[0].disabled
    assert app.button[1].label == "Next: LARGE.NS"
    app.button[1].click().run()
    assert app.session_state["selected_ticker"] == "LARGE.NS"
    assert app.button[0].label == "Previous: Portfolio overview"
    app.selectbox[0].select("SMALL").run()
    assert app.session_state["selected_ticker"] == "SMALL"
    assert app.button[1].disabled
    app.button[0].click().run()
    assert app.session_state["selected_ticker"] == "LARGE.NS"


def test_publication_reordering_keeps_selection_and_removed_symbol_resets_safely():
    app = picker_app()
    app.selectbox[0].select("SMALL").run()
    app.session_state["replacement_rows"] = list(reversed(ROWS))
    app.run()
    assert app.session_state["selected_ticker"] == "SMALL"
    app.session_state["replacement_rows"] = ROWS[:1]
    app.run()
    assert not app.exception
    assert app.session_state["selected_ticker"] is None


def test_current_and_exited_views_keep_independent_selection():
    app = picker_app()
    app.selectbox[0].select("SMALL").run()
    app.session_state["scope"] = "Exited holdings"
    app.session_state["replacement_rows"] = [{"ticker": "EXIT", "target_weight": .05,
                                              "status": "removed", "exit_date": "2026-10-20"}]
    app.run()
    assert app.selectbox[0].options[1] == "EXIT · last allocation 5% · exited 20 Oct 2026"
    assert app.session_state["selected_ticker"] is None
    app.session_state["scope"] = "Current holdings"
    app.session_state["replacement_rows"] = ROWS
    app.run()
    assert app.session_state["selected_ticker"] == "SMALL"


def test_empty_view_has_overview_without_navigation_errors():
    app = picker_app([])
    assert not app.exception
    assert app.selectbox[0].options == ["Portfolio overview"]
    assert all(button.disabled for button in app.button)
    assert "date unavailable" in evidence_option_label({"ticker": "X", "entry_date": None})


def test_returns_sort_numerically_highest_first_with_missing_after_losses():
    rows = [{"ticker": ticker, "entry_date": "2026-10-01", "target_weight": .1}
            for ticker in ("LOSS", "MISSING", "GAIN", "BEST", "INVALID")]
    summaries = [{**row, "return": value} for row, value in zip(rows, (-.2, None, .1, 1.2, float("nan")))]
    ranked = rank_evidence_rows(rows, summaries)
    assert [row["ticker"] for row in ranked] == ["BEST", "GAIN", "LOSS", "INVALID", "MISSING"]
    assert "₹100 → ₹220.00" in evidence_option_label(ranked[0])
    assert "₹100 → ₹80.00" in evidence_option_label(ranked[2])
    assert "₹100 → —" in evidence_option_label(ranked[-1])
    assert all("evidence_return" not in row for row in rows)


def test_stale_return_from_a_different_lifecycle_is_not_reused():
    summaries = [{"ticker": "LARGE.NS", "entry_date": "2026-09-01", "return": 4.}]
    ranked = rank_evidence_rows(ROWS, summaries)
    assert all(row["evidence_return"] is None for row in ranked)
