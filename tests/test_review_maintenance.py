import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from check_public_review_maintenance import main
from public_review.maintenance import policy_maintenance_status


POLICY = {
    "tariff_verified_on": "2026-09-09",
    "tariff_max_age_days": 90,
    "calendar_verified_through": "2026-12-31",
}


class MaintenanceTests(unittest.TestCase):
    def test_current_horizons(self):
        status = policy_maintenance_status(POLICY, today=date(2026, 10, 1))
        self.assertEqual(status["status"], "current")
        self.assertEqual(status["tariff"]["valid_through"], "2026-12-08")

    def test_warning_before_tariff_expiry(self):
        status = policy_maintenance_status(POLICY, today=date(2026, 11, 15))
        self.assertEqual(status["status"], "attention")
        self.assertEqual(status["tariff"]["state"], "review_due")

    def test_expired_tariff_blocks_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps(POLICY), encoding="utf-8")
            self.assertEqual(
                main(["--policy", str(path), "--as-of", "2026-12-09"]), 1
            )

    def test_calendar_has_independent_horizon(self):
        policy = dict(POLICY, tariff_max_age_days=180)
        status = policy_maintenance_status(policy, today=date(2026, 12, 15))
        self.assertEqual(status["status"], "attention")
        self.assertEqual(status["tariff"]["state"], "current")
        self.assertEqual(status["calendar"]["state"], "review_due")


if __name__ == "__main__":
    unittest.main()
