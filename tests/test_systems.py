import unittest

import numpy as np

from core.transfer_function import TransferFunction
from systems.dc_motor import DCMotor


class TestDCMotor(unittest.TestCase):
    """
    Unit Tests for the DC Motor system implementation.
    """

    def setUp(self):
        self.params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1, "L": 0.5}
        self.motor = DCMotor(**self.params)
        self.ss = self.motor.get_state_space()

    def test_ss_matrix_values(self):
        """Check explicit values of A matrix based on physics."""
        J, b, K, R, L = 0.01, 0.1, 0.01, 1, 0.5

        expected_A = np.array([[-b / J, K / J], [-K / L, -R / L]])
        np.testing.assert_array_almost_equal(self.ss.A, expected_A)

    def test_ss_dimensions(self):
        self.assertEqual(self.ss.A.shape, (2, 2))
        self.assertEqual(self.ss.B.shape, (2, 2))
        self.assertEqual(self.ss.C.shape, (2, 2))
        self.assertEqual(self.ss.D.shape, (2, 2))

    def test_stability_eigenvalues(self):
        eigenvalues = np.linalg.eigvals(self.ss.A)
        self.assertTrue(np.all(eigenvalues.real < 0))

    def test_physics_forward_motion(self):
        A, B = self.ss.A, self.ss.B
        u = np.array([[10], [0]])  # 10V
        x_ss = -np.linalg.inv(A) @ B @ u
        self.assertGreater(x_ss[0, 0], 0)

    def test_closed_loop_tf_validity(self):
        tf = self.motor.get_closed_loop_tf(Kp=5, Ki=10, Kd=0)
        dc_gain = abs(tf.evaluate(1e-9))
        self.assertAlmostEqual(dc_gain, 1.0, places=3)

    def test_augmented_state_space_structure(self):
        ss_aug = self.motor.get_augmented_state_space()
        np.testing.assert_array_equal(ss_aug.A[2, :], [0, 0, 0])
        self.assertEqual(ss_aug.B[0, 0], 0)

    def test_param_estimation_dynamics_shape(self):
        """Test standard scalar input shape."""
        f = self.motor.get_parameter_estimation_func()
        x = np.ones((4, 1))
        u = np.ones((1, 1))
        dx = f(x, u)
        self.assertEqual(dx.shape, (4, 1))

    def test_param_estimation_dynamics_vectorization(self):
        """
        NEW: Test if the dynamics function supports broadcasting.
        This is crucial for the vectorized EKF Jacobian calculation.
        """
        f = self.motor.get_parameter_estimation_func()

        # Create a batch of 10 state vectors (4, 10)
        # e.g., simulating 10 different particles at once
        x_batch = np.ones((4, 10))
        u_batch = np.ones((1, 1))  # Scalar control applied to all

        dx_batch = f(x_batch, u_batch)

        # Output should maintain the batch dimension (4, 10)
        self.assertEqual(dx_batch.shape, (4, 10))

        # Verify calculation is consistent across batch
        first_col = dx_batch[:, 0]
        last_col = dx_batch[:, -1]
        np.testing.assert_array_almost_equal(first_col, last_col)


if __name__ == "__main__":
    unittest.main()
