import unittest

import numpy as np

from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter


class TestEstimators(unittest.TestCase):
    """
    Unit Tests for Kalman Filter classes.
    """

    def test_kf_init_shapes(self):
        A = np.eye(2)
        B = np.zeros((2, 1))
        C = np.array([[1, 0]])  # Measure state 0
        Q = np.eye(2) * 0.1
        R = np.eye(1) * 0.1
        x0 = [0, 0]

        kf = KalmanFilter(A, B, C, Q, R, x0)
        self.assertEqual(kf.x_hat.shape, (2, 1))
        self.assertEqual(kf.P.shape, (2, 2))

    def test_kf_update_convergence(self):
        """Simple scalar estimator should converge to measurement average."""
        A = np.array([[1]])
        B = np.array([[0]])
        C = np.array([[1]])
        Q = np.array([[0.001]])
        R = np.array([[0.1]])
        x0 = [0]

        kf = KalmanFilter(A, B, C, Q, R, x0)

        # Noisy measurements of value 10
        np.random.seed(42)
        measurements = np.random.normal(10, 0.5, 50)

        for z in measurements:
            kf.update(u=0, y_meas=z)

        # Should be close to 10
        self.assertTrue(9.0 < kf.x_hat[0, 0] < 11.0)

    def test_ekf_init_jacobian(self):
        def f(x, u):
            return x

        def h(x):
            return x

        ekf = ExtendedKalmanFilter(f, h, np.eye(1), np.eye(1), [0])

        # Test compute_jacobian explicit call. Must pass 'u' because lambda accepts it.
        J = ekf.compute_jacobian(lambda x, u: x**2, np.array([3.0]), u=0)
        self.assertAlmostEqual(J[0, 0], 6.0)  # d(x^2)/dx at 3 = 2*3 = 6

    def test_ekf_predict_variance_growth(self):
        """Without updates, P should grow due to Q."""
        f = lambda x, u: x
        h = lambda x: x
        Q = np.array([[0.1]])
        R = np.array([[0.1]])
        x0 = [0]

        ekf = ExtendedKalmanFilter(f, h, Q, R, x0)
        initial_P = ekf.P[0, 0]

        ekf.predict(u=0, dt=1.0)

        # P_new = F*P*F' + Q = 1*P*1 + 0.1
        self.assertGreater(ekf.P[0, 0], initial_P)

    def test_ekf_full_loop(self):
        """Run predict and update loop."""
        # x_dot = 0. Measured directly.
        f = lambda x, u: np.zeros_like(x)
        h = lambda x: x

        ekf = ExtendedKalmanFilter(f, h, np.eye(1) * 0.1, np.eye(1) * 0.1, [5.0])

        # Measurement says 0. State says 5. Should decrease.
        ekf.predict(u=0, dt=0.1)
        ekf.update(y_meas=np.array([0.0]))

        self.assertLess(ekf.x_hat[0, 0], 5.0)


if __name__ == "__main__":
    unittest.main()
