import unittest

import numpy as np

from core.analysis import get_stability_margins, get_step_metrics
from core.transfer_function import TransferFunction


class TestAnalysis(unittest.TestCase):
    """
    Unit tests for control system analysis tools.
    """

    def test_margins_stable_3rd_order(self):
        """G(s) = 8 / (s+1)(s+2)(s+3)"""
        # (s^2 + 3s + 2)(s+3) = s^3 + 6s^2 + 11s + 6
        tf = TransferFunction([8], [1, 6, 11, 6])
        gm, pm, w_pc, w_gc = get_stability_margins(tf, w_start=-1, w_end=2)

        # Stable system, should have positive margins
        self.assertGreater(gm, 0)
        self.assertGreater(pm, 0)

    def test_margins_unstable(self):
        """G(s) = 1/(s-1). Unstable pole."""
        tf = TransferFunction([1], [1, -1])
        # Function handles this gracefully, typically returning infinite or negative margins
        try:
            get_stability_margins(tf)
        except Exception as e:
            self.fail(f"Should handle unstable TF without crashing: {e}")

    def test_margins_infinite(self):
        """G(s) = 1/(s+1). 1st order. Never crosses -180."""
        tf = TransferFunction([1], [1, 1])
        gm, pm, w_pc, w_gc = get_stability_margins(tf)

        self.assertEqual(gm, np.inf)  # Never unstable gain
        self.assertGreater(pm, 0)

    def test_step_metrics_ideal(self):
        """Linear ramp 0->1."""
        t = np.linspace(0, 10, 100)
        y = np.clip(t, 0, 1)  # Ramps 0 to 1 in 1 sec

        tr, os, st = get_step_metrics(t, y)

        # Rise time 0.1 to 0.9 = 0.8s
        self.assertAlmostEqual(tr, 0.8, delta=0.1)
        self.assertEqual(os, 0.0)

    def test_step_metrics_flat(self):
        """Response is 0 (System off)."""
        t = np.linspace(0, 10, 10)
        y = np.zeros_like(t)
        tr, os, st = get_step_metrics(t, y)
        self.assertEqual(tr, 0)
        self.assertEqual(os, 0)

    def test_step_metrics_overshoot(self):
        """Response goes to 1.5 then 1."""
        t = np.linspace(0, 10, 100)
        # Simple fake overshoot profile
        y = np.ones_like(t)
        y[10:20] = 1.5  # Peak
        y[0:10] = np.linspace(0, 1.5, 10)

        tr, os, st = get_step_metrics(t, y)
        self.assertAlmostEqual(os, 50.0, places=1)


if __name__ == "__main__":
    unittest.main()
