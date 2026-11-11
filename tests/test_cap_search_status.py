from copy import deepcopy
from contextlib import contextmanager
import ast
from pathlib import Path
import sqlite3

import pytest

from cap_search_status import scan_age_text, render_scan_age
from shortlist_search_checkpoint import save_checkpoint, load_latest_checkpoint


def checkpoint(stamp="2026-01-23T12:00:00+00:00", **kwargs):
    return dict(status="complete", updated_at=stamp,
                results=[{"Shortlist cap":7700,"Maximum assets":550,"Status":"Feasible"}], **kwargs)


def test_age_counts_exchange_closes_not_weekends_holidays_or_open_session():
    saved=checkpoint()  # Friday after close; Monday 26 Jan is an NSE holiday.
    assert "0 NSE trading sessions old" in scan_age_text(saved,now="2026-01-27T09:59:59+00:00")
    assert "1 NSE trading session old" in scan_age_text(saved,now="2026-01-27T10:00:00+00:00")
    assert "23 Jan 2026, 17:30 IST" in scan_age_text(saved,now="2026-01-27T10:00:00+00:00")


def test_same_session_age_and_equivalent_timezone_are_consistent():
    saved=checkpoint("2026-01-27T14:00:00+05:30")
    assert "0 NSE trading sessions old" in scan_age_text(saved,now="2026-01-27T14:59:00+05:30")
    assert scan_age_text(saved,now="2026-01-27T15:30:00+05:30")==scan_age_text(saved,now="2026-01-27T10:00:00Z")
    assert "1 NSE trading session old" in scan_age_text(saved,now="2026-01-27T10:00:00Z")


def test_reuse_save_cannot_make_original_discovery_fresh_and_data_date_stays_distinct():
    saved=checkpoint("2026-01-28T12:00:00Z",analysis_cutoff="2026-01-22")
    saved["results"][0]["Search skipper context"]={"last_discovery_at_utc":"2026-01-23T12:00:00Z"}
    before=deepcopy(saved)
    text=scan_age_text(saved,now="2026-01-28T12:00:00Z")
    assert "2 NSE trading sessions old" in text and "23 Jan" in text
    assert "Price history through 2026-01-22" in text
    assert saved==before
    saved["status"]="running"
    text=scan_age_text(saved,now="2026-01-28T12:00:00Z")
    assert "Partial scan checkpoint" in text and "0 NSE trading sessions old" in text
    assert "1 pairs saved; resume available" in text


@pytest.mark.parametrize("stamp",[None,"bad","2026-01-23", "2026-01-29T12:00:00Z"])
def test_invalid_naive_or_future_dates_never_claim_freshness(stamp):
    text=scan_age_text(checkpoint(stamp),now="2026-01-28T12:00:00Z")
    assert "age unavailable" in text and "sessions old" not in text


def test_no_scan_and_read_failure_are_explicit_and_read_only():
    assert scan_age_text(None)=="No saved cap scan yet."
    class Slot:
        def caption(self,text):self.text=text
    slot=Slot()
    @contextmanager
    def unavailable():
        raise sqlite3.OperationalError("unavailable")
        yield
    render_scan_age(slot,"synthetic",unavailable)
    assert "age unavailable" in slot.text
    conn=sqlite3.connect(":memory:");conn.row_factory=sqlite3.Row
    save_checkpoint(conn,"alice",{},checkpoint()["results"],status="complete")
    save_checkpoint(conn,"bob",{},checkpoint()["results"],status="complete")
    conn.execute("UPDATE shortlist_search_checkpoints SET updated_at=? WHERE owner=?",("2026-01-23T12:00:00Z","alice"))
    conn.commit()
    before=conn.total_changes
    render_scan_age(slot,"alice",lambda:conn,now="2026-01-28T12:00:00Z")
    assert "2 NSE trading sessions old" in slot.text and conn.total_changes==before
    assert load_latest_checkpoint(conn,"alice")["updated_at"]=="2026-01-23T12:00:00Z"
    conn.close()


def test_real_opening_action_panel_contains_single_search_button_and_persisted_age():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file("tests/raw_peak_ui_fixture.py",default_timeout=10).run()
    assert not app.exception
    button=app.button(key="run_shortlist_cap_search_btn")
    assert button.label=="Run / resume cap search" and not button.disabled
    assert any("1 NSE trading session old" in item.value for item in app.caption)
    assert not any(b.key=="run_shortlist_cap_search_btn" for b in app.sidebar.button)
    button.click().run()
    assert not app.exception and app.session_state["fixture_cap_search_clicked"]
    app.session_state["fixture_advanced_search"]=False
    app.run()
    assert app.button(key="run_shortlist_cap_search_btn").disabled
    assert any("Enable Advanced" in c.value for c in app.caption)
    # Ensure the production action remains in the first reserved main-page block.
    tree=ast.parse(Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8"))
    action=next(n for n in tree.body if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Name) and i.context_expr.id=="run_analysis_placeholder" for i in n.items))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="render_cap_search_action" for n in ast.walk(action))
