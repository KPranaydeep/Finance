import unittest

import numpy as np

from portfolio_risk_metrics import (
    empirical_expected_shortfall,
    moving_block_bootstrap_expected_shortfall,
)


class PortfolioRiskMetricsTests(unittest.TestCase):
    def test_empirical_expected_shortfall_uses_worst_losses(self):
        returns = np.array([-0.20, -0.10, -0.02, 0.01, 0.03, 0.04])
        self.assertAlmostEqual(
            empirical_expected_shortfall(returns, confidence=0.80),
            0.15,
        )

    def test_expected_shortfall_never_reports_negative_loss(self):
        self.assertEqual(
            empirical_expected_shortfall([0.01, 0.02, 0.03], confidence=0.95),
            0.0,
        )

    def test_block_bootstrap_is_deterministic_and_non_negative(self):
        log_returns = np.log1p(
            np.array([0.01, -0.02, 0.015, -0.03, 0.005] * 80)
        )
        first = moving_block_bootstrap_expected_shortfall(log_returns)
        second = moving_block_bootstrap_expected_shortfall(log_returns)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)

    def test_invalid_confidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "confidence"):
            empirical_expected_shortfall([-0.1, 0.1], confidence=1.0)


if __name__ == "__main__":
    unittest.main()
