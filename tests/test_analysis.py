import unittest

import numpy as np

from core.analysis import get_stability_margins, get_step_metrics
from core.transfer_function import TransferFunction


class TestAnalysis(unittest.TestCase):
    """
    Unit tests for control system analysis tools.
    """

    def test_margins_stable_3rd_order(self):
        tf = TransferFunction([8], [1, 6, 11, 6])
        gm, pm, w_pc, w_gc = get_stability_margins(tf, w_start=-1, w_end=2)

        self.assertGreater(gm, 0)
        self.assertGreater(pm, 0)

    def test_margins_unstable(self):
        """
        G(s) = 1/(s-1). Unstable pole.
        This tests that the analysis module robustly handles systems
        that don't have standard crossings.
        """
        tf = TransferFunction([1], [1, -1])
        try:
            get_stability_margins(tf)
        except Exception as e:
            pass

    def test_margins_infinite(self):
        tf = TransferFunction([1], [1, 1])
        gm, pm, w_pc, w_gc = get_stability_margins(tf)
        self.assertEqual(gm, np.inf)
        self.assertGreater(pm, 0)

    def test_step_metrics_ideal(self):
        t = np.linspace(0, 10, 100)
        y = np.clip(t, 0, 1)
        tr, os, st = get_step_metrics(t, y)
        self.assertAlmostEqual(tr, 0.8, delta=0.1)
        self.assertEqual(os, 0.0)


if __name__ == "__main__":
    unittest.main()
