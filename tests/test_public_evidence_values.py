from streamlit.testing.v1 import AppTest

from public_evidence_navigation import rank_evidence_rows
from public_evidence_values import remember_card_outcome, reuse_card_outcomes, value_of_100


ROW = {"ticker": "A.NS", "entry_date": "2026-09-15", "target_weight": .11}


def test_card_value_is_reused_exactly_without_a_second_return_estimate():
    saved = {}
    metrics = {"value_100": 145.75, "ticker_return": .4575, "as_of": "2026-10-30"}
    assert remember_card_outcome(saved, "PUB", ROW, metrics)
    assert not remember_card_outcome(saved, "PUB", ROW, metrics)
    summaries = [{**ROW, "return": .10}]
    outcomes = reuse_card_outcomes([ROW], summaries, saved, "PUB")
    ranked = rank_evidence_rows([ROW], outcomes)
    assert ranked[0]["evidence_value_100"] == value_of_100(metrics) == 145.75
    assert saved[("PUB", "A.NS", "2026-09-15", "")]["data_through"] == "2026-10-30"


def test_saved_card_is_not_reused_for_another_publication_or_lifecycle():
    saved = {}
    remember_card_outcome(saved, "PUB", ROW, {"ticker_return": .3, "as_of": "2026-10-30"})
    assert reuse_card_outcomes([ROW], [], saved, "NEXT") == []
    assert reuse_card_outcomes([{**ROW, "entry_date": "2026-10-01"}], [], saved, "PUB") == []
    assert reuse_card_outcomes([{**ROW, "exit_date": "2026-10-30"}], [], saved, "PUB") == []


def test_missing_values_are_not_invented_and_old_card_metrics_still_work():
    assert value_of_100({"ticker_return": .25}) == 125.
    assert value_of_100({"return": -.5}) == 50.
    assert value_of_100({"value_100": 0}) == 0
    for outcome in ({}, {"return": None}, {"return": True}, {"value_100": float("nan")},
                    {"value_100": -1}, {"return": float("inf")}):
        assert value_of_100(outcome) is None


def test_picker_updates_once_from_cached_card_and_keeps_its_selection():
    script = f'''
import streamlit as st
from public_evidence_navigation import rank_evidence_rows, render_evidence_picker
from public_evidence_values import remember_card_outcome, reuse_card_outcomes, value_of_100
rows=[{ROW!r}, {{'ticker':'B','entry_date':'2026-09-15','target_weight':.2}}]
summaries=[{{**rows[0],'return':.1}},{{**rows[1],'return':.2}}]
saved=st.session_state.setdefault('outcomes',{{}})
st.session_state.setdefault('track_record_security_Current holdings','A.NS')
st.session_state['passes']=st.session_state.get('passes',0)+1
ranked=rank_evidence_rows(rows,reuse_card_outcomes(rows,summaries,saved,'PUB'))
selected=render_evidence_picker(ranked,'Current holdings')
metrics={{'ticker_return':.8,'as_of':'2026-10-30'}}
if selected and remember_card_outcome(saved,'PUB',selected,metrics):
    st.rerun()
st.session_state['rendered_value']=value_of_100(metrics)
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert app.session_state["passes"] == 2
    assert app.selectbox[0].value == "A.NS"
    assert "A.NS · ₹100 → ₹180.00" in app.selectbox[0].options[1]
    assert app.session_state["rendered_value"] == 180.
    app.run()
    assert app.session_state["passes"] == 3
