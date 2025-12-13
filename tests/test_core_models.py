import unittest

import numpy as np

from core.exceptions import DimensionMismatchError
from core.state_space import StateSpace
from core.transfer_function import TransferFunction


class TestCoreModels(unittest.TestCase):
    """
    Unit Tests for TransferFunction and StateSpace containers.
    """

    # --- TransferFunction Tests ---

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

    # --- StateSpace Tests ---

    def test_ss_init_valid(self):
        A = np.eye(2)
        B = np.zeros((2, 1))
        C = np.zeros((1, 2))
        D = np.zeros((1, 1))
        ss = StateSpace(A, B, C, D)
        self.assertEqual(ss.n_states, 2)

    def test_ss_init_invalid_shapes(self):
        """Catches DimensionMismatchError."""
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


if __name__ == "__main__":
    unittest.main()
