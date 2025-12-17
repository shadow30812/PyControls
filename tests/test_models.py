import unittest

import numpy as np

from core.exceptions import DimensionMismatchError
from core.state_space import StateSpace
from core.transfer_function import TransferFunction


class TestModels(unittest.TestCase):
    """
    Unit Tests for TransferFunction and StateSpace containers.
    """

    def test_tf_init_padding(self):
        tf = TransferFunction([1], [1, 2, 3])
        self.assertEqual(tf.num.dtype, float)
        self.assertEqual(len(tf.num), 1)

    def test_tf_evaluate(self):
        tf = TransferFunction([1], [1, 1])
        self.assertEqual(tf.evaluate(0), 1.0)
        self.assertEqual(tf.evaluate(-1), np.inf)

    def test_tf_to_state_space_simple(self):
        tf = TransferFunction([1], [1, 2])
        A, B, C, D = tf.to_state_space()
        self.assertEqual(A.shape, (1, 1))
        self.assertEqual(B[0, 0], 1.0)

    def test_ss_init_valid(self):
        A = np.eye(2)
        B = np.zeros((2, 1))
        C = np.zeros((1, 2))
        D = np.zeros((1, 1))
        ss = StateSpace(A, B, C, D)
        self.assertEqual(ss.n_states, 2)

    def test_ss_init_invalid_shapes(self):
        with self.assertRaises(DimensionMismatchError):
            StateSpace(
                np.ones((2, 3)), np.zeros((2, 1)), np.zeros((1, 2)), np.zeros((1, 1))
            )

        with self.assertRaises(DimensionMismatchError):
            StateSpace(np.eye(2), np.zeros((3, 1)), np.zeros((1, 2)), np.zeros((1, 1)))

        with self.assertRaises(DimensionMismatchError):
            StateSpace(np.eye(2), np.zeros((2, 1)), np.zeros((1, 3)), np.zeros((1, 1)))

    def test_ss_freq_response_integrator(self):
        ss = StateSpace([[0]], [[1]], [[1]], [[0]])
        mags, phases = ss.get_frequency_response([1.0])
        self.assertAlmostEqual(phases[0], -90.0)


class TestStateSpaceExtended(unittest.TestCase):
    """Extended, non-fragile contract tests for StateSpace."""

    def test_mimo_frequency_response_shapes(self):
        A = np.zeros((2, 2))
        B = np.eye(2)
        C = np.eye(2)
        D = np.zeros((2, 2))
        ss = StateSpace(A, B, C, D)

        omega = np.linspace(0.1, 10.0, 5)
        mags, phases = ss.get_frequency_response(omega, input_idx=1, output_idx=0)

        self.assertEqual(mags.shape, omega.shape)
        self.assertEqual(phases.shape, omega.shape)

    def test_frequency_response_invalid_indices(self):
        A = np.zeros((1, 1))
        B = np.ones((1, 1))
        C = np.ones((1, 1))
        D = np.zeros((1, 1))
        ss = StateSpace(A, B, C, D)

        with self.assertRaises(ValueError):
            ss.get_frequency_response([1.0], input_idx=2)

        with self.assertRaises(ValueError):
            ss.get_frequency_response([1.0], output_idx=2)

    def test_frequency_response_d_matrix_only(self):
        A = np.zeros((1, 1))
        B = np.zeros((1, 1))
        C = np.zeros((1, 1))
        D = np.array([[2.0]])
        ss = StateSpace(A, B, C, D)

        mags, phases = ss.get_frequency_response([0.5, 1.0])

        self.assertTrue(np.allclose(mags, 20 * np.log10(2.0)))
        self.assertTrue(np.allclose(phases, 0.0))

    def test_frequency_response_singularity_handling(self):
        A = np.array([[0.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        ss = StateSpace(A, B, C, D)

        mags, phases = ss.get_frequency_response([0.0])

        self.assertTrue(np.isinf(mags[0]))
        self.assertTrue(np.isfinite(phases[0]))


class TestTransferFunctionExtended(unittest.TestCase):
    """Extended, non-fragile contract tests for TransferFunction."""

    def test_repr_contains_original_coefficients(self):
        num = [1, 2]
        den = [3, 4, 5]
        tf = TransferFunction(num, den)
        rep = repr(tf)

        self.assertIn(str(num), rep)
        self.assertIn(str(den), rep)

    def test_evaluate_matches_polyval(self):
        num = [1, 0]
        den = [1, 1]
        tf = TransferFunction(num, den)

        s = 1 + 1j
        expected = np.polyval(num, s) / np.polyval(den, s)
        self.assertAlmostEqual(tf.evaluate(s), expected)

    def test_evaluate_zero_denominator(self):
        tf = TransferFunction([1.0], [0.0])
        val = tf.evaluate(1.0)
        self.assertEqual(val, np.inf)

    def test_bode_response_consistency(self):
        tf = TransferFunction([1.0], [1.0, 1.0])
        omega = np.array([0.1, 1.0, 10.0])

        mags, phases = tf.bode_response(omega)

        for i, w in enumerate(omega):
            s = 1j * w
            resp = tf.evaluate(s)
            self.assertAlmostEqual(mags[i], 20 * np.log10(abs(resp)))
            self.assertAlmostEqual(phases[i], np.degrees(np.angle(resp)))

    def test_to_state_space_shapes(self):
        tf = TransferFunction([1.0], [1.0, 2.0, 3.0])
        A, B, C, D = tf.to_state_space()

        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(B.shape, (2, 1))
        self.assertEqual(C.shape, (1, 2))
        self.assertTrue(np.isscalar(D) or np.shape(D) == ())

    def test_to_state_space_companion_structure(self):
        tf = TransferFunction([1.0], [1.0, 2.0, 3.0])
        A, _, _, _ = tf.to_state_space()

        self.assertEqual(A[0, 1], 1.0)
        self.assertTrue(np.allclose(A[-1, :], [-3.0, -2.0]))


if __name__ == "__main__":
    unittest.main()
