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
        C = np.array([[1, 0]])
        Q = np.eye(2) * 0.1
        R = np.eye(1) * 0.1
        x0 = [0, 0]

        kf = KalmanFilter(A, B, C, Q, R, x0)
        self.assertEqual(kf.x_hat.shape, (2, 1))

    def test_kf_update_convergence(self):
        A = np.array([[1]])
        B = np.array([[0]])
        C = np.array([[1]])
        Q = np.array([[0.001]])
        R = np.array([[0.1]])
        x0 = [0]

        kf = KalmanFilter(A, B, C, Q, R, x0)

        np.random.seed(42)
        measurements = np.random.normal(10, 0.5, 50)

        for z in measurements:
            kf.update(u=0, y_meas=z)

        self.assertTrue(9.0 < kf.x_hat[0, 0] < 11.0)

    # --- EKF Vectorization Tests ---

    def test_ekf_jacobian_vectorization_success(self):
        """
        Tests the optimized vectorized path for Jacobian computation.
        We use a simple function x -> x^2 which supports broadcasting natively.
        """

        def f_vectorizable(x, u=None):
            return x**2

        h = lambda x: x
        ekf = ExtendedKalmanFilter(f_vectorizable, h, np.eye(1), np.eye(1), [0])

        # Evaluate at x=3. J should be 2*3 = 6.
        # This calls compute_jacobian, which should use the vectorized block
        J = ekf.compute_jacobian(f_vectorizable, np.array([3.0]))
        self.assertAlmostEqual(J[0, 0], 6.0)

    def test_ekf_jacobian_vectorization_fallback(self):
        """
        Tests the fallback logic.
        We use a function that fails on matrix inputs to force the loop.
        """

        def f_non_vectorizable(x, u=None):
            # Explicit check that fails if x is a matrix (perturbation batch)
            if np.ndim(x) > 1 and x.shape[1] > 1:
                raise ValueError("I do not support matrices!")
            return x**2

        h = lambda x: x
        ekf = ExtendedKalmanFilter(f_non_vectorizable, h, np.eye(1), np.eye(1), [0])

        # Evaluate at x=3. J should be 6.
        J = ekf.compute_jacobian(f_non_vectorizable, np.array([3.0]))
        self.assertAlmostEqual(J[0, 0], 6.0)

    def test_ekf_predict_variance_growth(self):
        f = lambda x, u: x
        h = lambda x: x
        Q = np.array([[0.1]])
        R = np.array([[0.1]])
        x0 = [0]

        ekf = ExtendedKalmanFilter(f, h, Q, R, x0)
        initial_P = ekf.P[0, 0]

        ekf.predict(u=0, dt=1.0)
        self.assertGreater(ekf.P[0, 0], initial_P)


if __name__ == "__main__":
    unittest.main()
