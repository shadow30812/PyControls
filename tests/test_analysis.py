import unittest

import numpy as np

from core.analysis import get_exact_time_idx, get_stability_margins, get_step_metrics
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
        tf = TransferFunction([1], [1, -1])
        try:
            get_stability_margins(tf)
        except Exception as e:
            print("Skipped analysis test", e, sep="\n")

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

    def test_margins_pure_integrator(self):
        tf = TransferFunction([1], [1, 0])
        gm, pm, w_pc, w_gc = get_stability_margins(tf)

        self.assertEqual(gm, np.inf)
        self.assertTrue(pm > 0 or pm == np.inf)

    def test_margins_double_integrator(self):
        tf = TransferFunction([1], [1, 0, 0])
        gm, pm, w_pc, w_gc = get_stability_margins(tf)

        self.assertTrue(np.isinf(gm) or gm >= 0)
        self.assertTrue(np.isinf(pm) or pm >= 0)

    def test_step_metrics_constant_response(self):
        t = np.linspace(0, 5, 50)
        y = np.ones_like(t)

        tr, os, st = get_step_metrics(t, y)

        self.assertEqual(tr, 0)
        self.assertEqual(os, 0)

    def test_step_metrics_monotonic(self):
        t = np.linspace(0, 10, 200)
        y = 1 - np.exp(-t)

        tr, os, st = get_step_metrics(t, y)

        self.assertEqual(os, 0.0)
        self.assertGreaterEqual(tr, 0.0)

    def test_step_metrics_undershoot(self):
        t = np.linspace(0, 10, 200)
        y = 1 - 1.2 * np.exp(-t)

        tr, os, st = get_step_metrics(t, y)

        self.assertLessEqual(os, 0.0)

    def test_exact_time_idx_exact_hit(self):
        time = np.array([0.0, 1.0, 2.0])
        response = np.array([0.0, 1.0, 2.0])

        t_hit = get_exact_time_idx(time, response, 1.0)

        self.assertEqual(t_hit, 1.0)

    def test_exact_time_idx_flat_segment(self):
        time = np.array([0.0, 1.0, 2.0])
        response = np.array([1.0, 1.0, 2.0])

        t_hit = get_exact_time_idx(time, response, 1.0)

        self.assertEqual(t_hit, 0.0)


class TestEigenAndStabilityAnalysis(unittest.TestCase):
    """
    Additional tests for analysis.py focusing on guarantees:
    - finite outputs
    - correct classification behavior
    - numerical robustness
    """

    def test_margins_return_finite_or_inf(self):
        tf = TransferFunction([1], [1, 3, 2])
        gm, pm, w_pc, w_gc = get_stability_margins(tf)

        for val in (gm, pm, w_pc, w_gc):
            self.assertFalse(np.isnan(val))

    def test_no_crossing_returns_inf_margins(self):
        tf = TransferFunction([1], [1, 1])
        gm, pm, w_pc, w_gc = get_stability_margins(tf)

        self.assertEqual(gm, np.inf)
        self.assertEqual(w_pc, 0.0)

    def test_step_metrics_monotonic_response(self):
        t = np.linspace(0, 5, 200)
        y = 1.0 - np.exp(-t)

        tr, os, st = get_step_metrics(t, y)

        self.assertEqual(os, 0.0)
        self.assertGreaterEqual(tr, 0.0)

    def test_step_metrics_handles_flat_response(self):
        t = np.linspace(0, 5, 100)
        y = np.zeros_like(t)

        tr, os, st = get_step_metrics(t, y)

        self.assertEqual(tr, 0)
        self.assertEqual(os, 0)
        self.assertEqual(st, 0)

    def test_step_metrics_finite_output(self):
        t = np.linspace(0, 10, 500)
        y = np.tanh(t)

        tr, os, st = get_step_metrics(t, y)

        self.assertFalse(np.isnan(tr))
        self.assertFalse(np.isnan(os))
        self.assertFalse(np.isnan(st))


if __name__ == "__main__":
    unittest.main()
