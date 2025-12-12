import unittest

import numpy as np

from core.state_space import StateSpace
from core.transfer_function import TransferFunction


class TestCoreModels(unittest.TestCase):
    """
    Unit Tests for TransferFunction and StateSpace containers.
    """

    # --- TransferFunction Tests ---

    def test_tf_init_padding(self):
        """Ensure arrays are stored as numpy floats."""
        tf = TransferFunction([1], [1, 2, 3])
        self.assertEqual(tf.num.dtype, float)
        self.assertEqual(len(tf.num), 1)

    def test_tf_evaluate(self):
        tf = TransferFunction([1], [1, 1])  # 1/(s+1)
        self.assertEqual(tf.evaluate(0), 1.0)
        self.assertEqual(tf.evaluate(-1), np.inf)  # Pole

    def test_tf_to_state_space_simple(self):
        # 1/(s+2) -> x_dot = -2x + u, y = x
        tf = TransferFunction([1], [1, 2])
        A, B, C, D = tf.to_state_space()

        self.assertEqual(A.shape, (1, 1))
        self.assertEqual(A[0, 0], -2.0)
        self.assertEqual(B[0, 0], 1.0)

    def test_tf_to_state_space_proper(self):
        # (s+3)/(s^2 + 2s + 1)
        tf = TransferFunction([1, 3], [1, 2, 1])
        A, B, C, D = tf.to_state_space()

        self.assertEqual(A.shape, (2, 2))
        # D should be 0 (strictly proper)
        self.assertEqual(D, 0.0)

    def test_tf_to_state_space_biproper(self):
        # (2s+1)/(s+1) -> 2 + (-1)/(s+1) -> D=2
        tf = TransferFunction([2, 1], [1, 1])
        A, B, C, D = tf.to_state_space()
        self.assertEqual(D, 2.0)

    def test_bode_response_shape(self):
        tf = TransferFunction([1], [1, 1])
        w = np.linspace(0.1, 10, 50)
        mags, phases = tf.bode_response(w)
        self.assertEqual(len(mags), 50)
        self.assertEqual(len(phases), 50)

    # --- StateSpace Tests ---

    def test_ss_init_valid(self):
        A = np.eye(2)
        B = np.zeros((2, 1))
        C = np.zeros((1, 2))
        D = np.zeros((1, 1))
        ss = StateSpace(A, B, C, D)
        self.assertEqual(ss.n_states, 2)

    def test_ss_init_invalid_shapes(self):
        # Non-square A
        with self.assertRaises(ValueError):
            StateSpace(
                np.ones((2, 3)), np.zeros((2, 1)), np.zeros((1, 2)), np.zeros((1, 1))
            )

        # B mismatch
        with self.assertRaises(ValueError):
            StateSpace(np.eye(2), np.zeros((3, 1)), np.zeros((1, 2)), np.zeros((1, 1)))

        # C mismatch
        with self.assertRaises(ValueError):
            StateSpace(np.eye(2), np.zeros((2, 1)), np.zeros((1, 3)), np.zeros((1, 1)))

    def test_ss_freq_response_integrator(self):
        """H(s) = 1/s. A=0, B=1, C=1, D=0."""
        ss = StateSpace([[0]], [[1]], [[1]], [[0]])
        # w=1 -> 1/j -> -90 deg
        mags, phases = ss.get_frequency_response([1.0])
        self.assertAlmostEqual(phases[0], -90.0)

    def test_ss_freq_response_error_indices(self):
        ss = StateSpace([[0]], [[1]], [[1]], [[0]])
        with self.assertRaises(ValueError):
            ss.get_frequency_response([1], input_idx=5)
        with self.assertRaises(ValueError):
            ss.get_frequency_response([1], output_idx=5)


if __name__ == "__main__":
    unittest.main()
