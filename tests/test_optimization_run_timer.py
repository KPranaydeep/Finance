import unittest
from datetime import datetime, timezone

from optimization_run_timer import (
    abort_run_timer,
    finish_run_timer,
    format_elapsed,
    serializable_timer,
    start_run_timer,
)


class OptimizationRunTimerTests(unittest.TestCase):
    def test_finish_measures_click_to_download(self):
        started = start_run_timer(
            monotonic_now=100.0,
            wall_now=datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc),
        )
        finished = finish_run_timer(
            started,
            "lumpsum_buy_orders.html",
            monotonic_now=225.25,
            wall_now=datetime(2026, 10, 2, 10, 2, 5, tzinfo=timezone.utc),
        )
        self.assertEqual(finished["status"], "finished")
        self.assertEqual(finished["elapsed_seconds"], 125.25)
        self.assertEqual(finished["download_file"], "lumpsum_buy_orders.html")
        self.assertEqual(format_elapsed(finished["elapsed_seconds"]), "2 m 05.2 s")

    def test_finish_is_idempotent(self):
        started = start_run_timer(monotonic_now=10.0)
        first = finish_run_timer(started, "first.csv", monotonic_now=15.0)
        second = finish_run_timer(first, "second.csv", monotonic_now=30.0)
        self.assertEqual(second["download_file"], "first.csv")
        self.assertEqual(second["elapsed_seconds"], 5.0)

    def test_abort_is_distinct_from_download(self):
        started = start_run_timer(monotonic_now=5.0)
        stopped = abort_run_timer(started, "invalid holdings", monotonic_now=8.5)
        self.assertEqual(stopped["status"], "stopped_without_download")
        self.assertEqual(stopped["elapsed_seconds"], 3.5)
        self.assertIsNone(stopped["download_file"])

    def test_serialized_state_excludes_process_clock(self):
        state = start_run_timer(monotonic_now=42.0)
        serialized = serializable_timer(state)
        self.assertNotIn("monotonic_start", serialized)
        self.assertEqual(serialized["status"], "running")


if __name__ == "__main__":
    unittest.main()
