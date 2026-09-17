import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class UiTests(unittest.TestCase):
    def test_durable_preview_suppresses_transient_fresh_error(self):
        from public_review.forecast import TIMING_MODEL
        from public_review.ui import has_durable_preview
        events = [
            {"kind": "BASELINE", "baseline_id": "B1",
             "payload": {"baseline_id": "B1", "publication_id": "P1"}},
            {"kind": "PREVIEW", "baseline_id": "B1",
             "payload": {"forecast": {"timing_model": TIMING_MODEL}}},
        ]
        self.assertTrue(has_durable_preview(events, "P1"))
        self.assertFalse(has_durable_preview(events, "P2"))

    def test_legacy_preview_does_not_suppress_fresh_error(self):
        from public_review.ui import has_durable_preview
        events = [
            {"kind": "BASELINE", "baseline_id": "B1",
             "payload": {"baseline_id": "B1", "publication_id": "P1"}},
            {"kind": "PREVIEW", "baseline_id": "B1", "payload": {
                "forecast": {"research_candidate": "2026-09-15"}}},
        ]
        self.assertFalse(has_durable_preview(events, "P1"))

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

    def test_security_estimates_default_to_numeric_probability_order(self):
        from public_review.ui import sort_security_estimates
        rows = [
            {"ticker": "LOW.NS", "probability": .09,
             "review_date": "2026-09-16", "target_weight": .40},
            {"ticker": "HIGH-LATE.NS", "probability": .24,
             "review_date": "2026-09-18", "target_weight": .10},
            {"ticker": "HIGH-EARLY.NS", "probability": .24,
             "review_date": "2026-09-17", "target_weight": .05},
            {"ticker": "UNKNOWN.NS", "probability": None,
             "review_date": None, "target_weight": .45},
        ]
        ordered = sort_security_estimates(
            rows, "Probability by date (high to low)")
        self.assertEqual(
            [row["ticker"] for row in ordered],
            ["HIGH-EARLY.NS", "HIGH-LATE.NS", "LOW.NS", "UNKNOWN.NS"],
        )

    def test_security_estimates_can_sort_by_review_date(self):
        from public_review.ui import sort_security_estimates
        rows = [
            {"ticker": "LATE.NS", "crossing_probability": .80,
             "review_date": "2026-09-18"},
            {"ticker": "EARLY.NS", "crossing_probability": .20,
             "review_date": "2026-09-16"},
            {"ticker": "UNDATED.NS", "crossing_probability": None},
        ]
        ordered = sort_security_estimates(
            rows, "Review date (earliest first)")
        self.assertEqual(
            [row["ticker"] for row in ordered],
            ["EARLY.NS", "LATE.NS", "UNDATED.NS"],
        )
