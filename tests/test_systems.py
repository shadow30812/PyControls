import unittest

import numpy as np

from systems.dc_motor import DCMotor
from systems.pendulum import InvertedPendulum, LQRLoopTransferFunction


class TestDCMotor(unittest.TestCase):
    """
    Unit Tests for the DC Motor system implementation.
    """

    def setUp(self):
        self.params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1, "L": 0.5}
        self.motor = DCMotor(**self.params)
        self.ss = self.motor.get_state_space()

    def test_ss_matrix_values(self):
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
        u = np.array([[10], [0]])
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
        f = self.motor.get_parameter_estimation_func()
        x = np.ones((4, 1))
        u = np.ones((1, 1))
        dx = f(x, u)
        self.assertEqual(dx.shape, (4, 1))

    def test_param_estimation_dynamics_vectorization(self):
        f = self.motor.get_parameter_estimation_func()

        x_batch = np.ones((4, 10))
        u_batch = np.ones((1, 1))

        dx_batch = f(x_batch, u_batch)

        self.assertEqual(dx_batch.shape, (4, 10))

        first_col = dx_batch[:, 0]
        last_col = dx_batch[:, -1]
        np.testing.assert_array_almost_equal(first_col, last_col)

    def test_open_loop_tf_orders(self):
        tf = self.motor.get_open_loop_tf(Kp=1.0, Ki=2.0, Kd=3.0)
        self.assertGreaterEqual(len(tf.den), len(tf.num))

    def test_disturbance_tf_shape(self):
        tf = self.motor.get_disturbance_tf(Kp=1.0, Ki=1.0, Kd=0.0)
        self.assertGreater(len(tf.den), 0)
        self.assertGreater(len(tf.num), 0)

    def test_mpc_model_shapes(self):
        A_d, B_d = self.motor.get_mpc_model(dt=0.1)
        self.assertEqual(A_d.shape, (2, 2))
        self.assertEqual(B_d.shape, (2, 1))

    def test_nonlinear_dynamics_shapes(self):
        f, h = self.motor.get_nonlinear_dynamics()
        x = np.array([0.1, 0.0])
        u = 1.0
        x_next = f(x, u, dt=0.01)
        y = h(x)
        self.assertEqual(x_next.shape, (2,))
        self.assertEqual(y.shape, (2,))


class TestInvertedPendulum(unittest.TestCase):
    """
    Exhaustive but non-fragile contract tests for the InvertedPendulum system.
    """

    def setUp(self):
        self.pendulum = InvertedPendulum()

    def test_linear_matrix_shapes(self):
        self.assertEqual(self.pendulum.A.shape, (4, 4))
        self.assertEqual(self.pendulum.B.shape, (4, 1))

    def test_upright_equilibrium_instability(self):
        eigvals = np.linalg.eigvals(self.pendulum.A)
        self.assertTrue(np.any(eigvals.real > 0))

    def test_state_space_dimensions(self):
        ss = self.pendulum.get_state_space()
        self.assertEqual(ss.A.shape, (4, 4))
        self.assertEqual(ss.B.shape, (4, 1))
        self.assertEqual(ss.C.shape, (4, 4))

    def test_augmented_state_space_structure(self):
        ss_aug = self.pendulum.get_augmented_state_space()
        self.assertEqual(ss_aug.A.shape, (5, 5))
        self.assertEqual(ss_aug.B.shape, (5, 1))
        self.assertTrue(np.all(ss_aug.A[4, :] == 0.0))

    def test_parameter_estimation_shape(self):
        f = self.pendulum.get_parameter_estimation_func()
        x = np.ones((6, 1))
        u = np.ones((1, 1))
        dx = f(x, u)
        self.assertEqual(dx.shape, (6, 1))

    def test_parameter_estimation_vectorization(self):
        f = self.pendulum.get_parameter_estimation_func()
        x = np.ones((6, 10))
        u = np.ones((1, 1))
        dx = f(x, u)
        self.assertEqual(dx.shape, (6, 10))
        np.testing.assert_array_equal(dx[4], 0.0)
        np.testing.assert_array_equal(dx[5], 0.0)

    def test_measurement_mapping(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = self.pendulum.measurement(x)
        np.testing.assert_array_equal(y, [1.0, 3.0])

    def test_measurement_jacobian_constant(self):
        H = self.pendulum.measurement_jacobian(None)
        self.assertEqual(H.shape, (2, 4))
        self.assertEqual(H[0, 0], 1.0)
        self.assertEqual(H[1, 2], 1.0)

    def test_discrete_dynamics_shape(self):
        x = np.zeros(4)
        x_next = self.pendulum.dynamics(x, u=0.0, dt=0.01)
        self.assertEqual(x_next.shape, (4,))

    def test_continuous_dynamics_shape(self):
        x = np.zeros(4)
        dx = self.pendulum.dynamics_continuous(x, u=0.0)
        self.assertEqual(dx.shape, (4,))

    def test_ukf_dynamics_interface(self):
        f, h = self.pendulum.get_nonlinear_dynamics()
        x = np.array([0.1, 0.0])
        x_next = f(x, u=0.0, dt=0.01)
        y = h(x)
        self.assertEqual(x_next.shape, (2,))
        self.assertEqual(y.shape, (1,))

    def test_mpc_model_callable(self):
        f = self.pendulum.get_mpc_model(dt=0.05)
        x = np.zeros(4)
        u = np.zeros(1)
        x_next = f(x, u, dt=0.05)
        self.assertEqual(x_next.shape, (4,))


class TestLQRLoopTransferFunction(unittest.TestCase):
    """
    Contract tests for the LQRLoopTransferFunction helper.
    """

    def test_evaluate_finite(self):
        A = np.array([[0.0]])
        B = np.array([[1.0]])
        K = np.array([[1.0]])
        tf = LQRLoopTransferFunction(A, B, K)
        val = tf.evaluate(1.0)
        self.assertTrue(np.isfinite(val))

    def test_evaluate_singularity(self):
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        K = np.array([[1.0]])
        tf = LQRLoopTransferFunction(A, B, K)
        val = tf.evaluate(1.0)
        self.assertEqual(val, np.inf)


if __name__ == "__main__":
    unittest.main()
