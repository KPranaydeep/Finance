import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

from trade_deferrals import active_trade_deferrals


def editor(database):
    app = AppTest.from_file(str(Path(__file__).with_name('optimal_weight_pause_ui_fixture.py')))
    app.session_state['fixture_db'] = str(database)
    app.run()
    assert not app.exception
    return app


def pauses(database):
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        return active_trade_deferrals(conn, 'synthetic-owner')


def test_pause_from_weight_without_trade_persists_after_restart(tmp_path):
    database = tmp_path / 'test.sqlite'
    app = editor(database)
    assert app.dataframe[0].value['Yahoo Ticker'].tolist() == ['HOLD.NS', 'NOORDER']
    app.session_state['saved-test_pause_editor'] = {
        'edited_rows': {1: {'Pause 90 days': True}}, 'added_rows': [], 'deleted_rows': [],
    }
    app.button[0].click().run()
    assert not app.exception
    saved = pauses(database)
    assert saved['yahoo_ticker'].tolist() == ['NOORDER']
    assert saved['requested_action'].tolist() == ['Allocation']
    expiry = saved.iloc[0]['expires_at']
    # A new application session reads the same persistent security pause.
    restarted = editor(database)
    table = restarted.dataframe[0].value.set_index('Yahoo Ticker')
    assert table.loc['NOORDER', 'Paused until'] == datetime.fromisoformat(expiry).astimezone(
        ZoneInfo('Asia/Kolkata')).strftime('%d %b %Y, %H:%M IST')
    assert table.loc['HOLD.NS', 'Paused until'] == 'Not paused'
    assert table.loc['NOORDER', 'Optimal Weight'] == .4
    assert table.loc['HOLD.NS', 'Optimal Weight'] == .6
    restarted.session_state['saved-test_pause_editor'] = {
        'edited_rows': {1: {'Pause 90 days': True}}, 'added_rows': [], 'deleted_rows': [],
    }
    restarted.button[0].click().run()
    assert not restarted.exception and restarted.warning
    assert pauses(database).iloc[0]['expires_at'] == expiry


def test_empty_selection_does_not_write_and_two_scopes_have_unique_forms(tmp_path):
    database = tmp_path / 'test.sqlite'
    app = editor(database)
    app.button[0].click().run()
    assert not app.exception and app.warning
    assert pauses(database).empty
    app.session_state['show_second_view'] = True
    app.run()
    assert not app.exception
    assert len(app.button) == 2
