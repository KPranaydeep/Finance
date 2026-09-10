import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class UiTests(unittest.TestCase):
    def app(self,mode):
        app=AppTest.from_file(str(Path(__file__).with_name('review_ui_fixture.py')),default_timeout=20)
        app.session_state['fixture_mode']=mode
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_panel_renders_net_metrics(self):
        app=self.app('normal')
        self.assertEqual(len(app.metric),4)
        self.assertEqual(app.metric[0].value,'Not validated')
        self.assertTrue(any('Alert delivery' in w.value for w in app.warning))

    def test_stale_hides_actionable_values(self):
        app=self.app('stale')
        self.assertEqual(len(app.metric),0)
        self.assertTrue(any('stale' in w.value for w in app.warning))

    def test_failure_hides_previous_metrics(self):
        app=self.app('failure')
        self.assertEqual(len(app.metric),0)
        self.assertTrue(any('Cannot assess' in w.value for w in app.warning))

    def test_unconfigured_is_read_only_message(self):
        app=self.app('empty')
        self.assertTrue(app.info)
        self.assertEqual(len(app.metric),0)

    def test_recovery_heartbeat_clears_prior_failure(self):
        app=self.app('recovered')
        self.assertEqual(len(app.metric),4)

    def test_latest_selected_and_older_selectable(self):
        app=self.app('second')
        self.assertEqual(app.selectbox[0].value,0)
        self.assertEqual(len(app.metric),0)
        app.selectbox[0].set_value(1).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric),4)
