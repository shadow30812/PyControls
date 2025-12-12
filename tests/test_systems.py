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
        # A = [[-b/J, K/J], [-K/L, -R/L]]
        J, b, K, R, L = 0.01, 0.1, 0.01, 1, 0.5

        expected_A = np.array([[-b / J, K / J], [-K / L, -R / L]])
        # Expected: [[-10, 1], [-0.02, -2]]
        np.testing.assert_array_almost_equal(self.ss.A, expected_A)

    def test_ss_dimensions(self):
        self.assertEqual(self.ss.A.shape, (2, 2))
        self.assertEqual(self.ss.B.shape, (2, 2))  # [V, Load]
        self.assertEqual(self.ss.C.shape, (2, 2))
        self.assertEqual(self.ss.D.shape, (2, 2))

    def test_stability_eigenvalues(self):
        """Passive motor must be stable (real part of poles < 0)."""
        eigenvalues = np.linalg.eigvals(self.ss.A)
        self.assertTrue(np.all(eigenvalues.real < 0))

    def test_physics_forward_motion(self):
        """Positive voltage -> Positive speed (Steady State)."""
        A, B = self.ss.A, self.ss.B
        u = np.array([[10], [0]])  # 10V, 0 Load
        x_ss = -np.linalg.inv(A) @ B @ u

        speed = x_ss[0, 0]
        current = x_ss[1, 0]

        self.assertGreater(speed, 0)
        self.assertGreater(current, 0)

    def test_physics_load_response(self):
        """Positive Load Torque -> Negative Speed change (opposing motion)."""
        A, B = self.ss.A, self.ss.B
        u = np.array([[0], [1.0]])  # 0V, 1Nm Load opposing
        x_ss = -np.linalg.inv(A) @ B @ u

        speed = x_ss[0, 0]
        # Speed should be negative (backwards) or less than zero
        self.assertLess(speed, 0)

    def test_closed_loop_tf_validity(self):
        """Verify Closed Loop TF generation."""
        # Unity Gain PI
        tf = self.motor.get_closed_loop_tf(Kp=5, Ki=10, Kd=0)

        # Evaluate near DC (epsilon) to avoid singular integrator issues
        dc_gain = abs(tf.evaluate(1e-9))
        self.assertAlmostEqual(dc_gain, 1.0, places=3)

        # Check order. Motor (2nd) + PI (1st) = 3rd order denominator
        # However, pole-zero cancellation might occur.
        # Generally den should be size 4 (s^3, s^2, s^1, s^0)
        self.assertEqual(len(tf.den), 4)

    def test_augmented_state_space_structure(self):
        """Verify structure of augmented model for EKF."""
        ss_aug = self.motor.get_augmented_state_space()

        # A_aug structure:
        # [A_std   B_dist]
        # [0       0     ]

        # Check bottom row is zeros (random walk assumption)
        np.testing.assert_array_equal(ss_aug.A[2, :], [0, 0, 0])

        # Check B_aug: Input only affects current derivative (row 1)
        # B_aug = [0; 1/L; 0]
        self.assertEqual(ss_aug.B[0, 0], 0)
        self.assertEqual(ss_aug.B[2, 0], 0)
        self.assertGreater(ss_aug.B[1, 0], 0)

    def test_param_estimation_dynamics_shape(self):
        f = self.motor.get_parameter_estimation_func()
        x = np.ones((4, 1))
        u = np.ones((1, 1))
        dx = f(x, u)
        self.assertEqual(dx.shape, (4, 1))
        # Parameters j, b have 0 derivative
        self.assertEqual(dx[2, 0], 0)
        self.assertEqual(dx[3, 0], 0)


if __name__ == "__main__":
    unittest.main()
