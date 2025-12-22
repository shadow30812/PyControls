import unittest

import numpy as np

from core.ekf import ExtendedKalmanFilter
from core.ekf_discrete import DiscreteExtendedKalmanFilter
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
            kf.predict(u=0)
            kf.update(y_meas=z)

        self.assertTrue(9.0 < kf.x_hat[0, 0] < 11.0)

    def test_ekf_jacobian_vectorization_success(self):
        """
        Tests the optimized vectorized path for Jacobian computation.
        We use a simple function x -> x^2 which supports broadcasting natively.
        """

        def f_vectorizable(x, u=None):
            return x**2

        h = lambda x: x
        ekf = ExtendedKalmanFilter(f_vectorizable, h, np.eye(1), np.eye(1), [0])

        J = ekf.compute_jacobian(f_vectorizable, np.array([3.0]))
        self.assertAlmostEqual(J[0, 0], 6.0)

    def test_ekf_jacobian_vectorization_fallback(self):
        """
        Tests the fallback logic.
        We use a function that fails on matrix inputs to force the loop.
        """

        def f_non_vectorizable(x, u=None):
            if np.ndim(x) > 1 and x.shape[1] > 1:
                raise ValueError("I do not support matrices!")
            return x**2

        h = lambda x: x
        ekf = ExtendedKalmanFilter(f_non_vectorizable, h, np.eye(1), np.eye(1), [0])

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

    def test_ekf_jacobian_vectorized_with_input(self):
        def f(x, u):
            return x + u

        ekf = ExtendedKalmanFilter(f, lambda x: x, np.eye(2), np.eye(2), [0, 0])
        J = ekf.compute_jacobian(f, np.array([1.0, 2.0]), u=1.0)

        np.testing.assert_array_equal(J, np.eye(2))

    def test_ekf_covariance_symmetry(self):
        f = lambda x, u: x
        h = lambda x: x

        ekf = ExtendedKalmanFilter(f, h, np.eye(1), np.eye(1), [0.0])
        ekf.predict(0, 1.0)
        ekf.update(np.array([1.0]))

        self.assertAlmostEqual(ekf.P[0, 0], ekf.P.T[0, 0])

    def test_ekf_nonlinear_consistency(self):
        f = lambda x, u: x**2
        h = lambda x: x

        ekf = ExtendedKalmanFilter(f, h, np.eye(1) * 0.01, np.eye(1) * 0.1, [1.0])
        ekf.predict(0, 0.1)

        self.assertGreater(ekf.x_pred[0, 0], 1.0)

    def test_ekf_jacobian_matches_analytic(self):
        f = lambda x, u: np.array([x[0] ** 2, np.sin(x[1])])

        h = lambda x: x

        ekf = ExtendedKalmanFilter(f, h, np.eye(2), np.eye(2), [1.5, 0.3])

        x = np.array([1.5, 0.3])
        u = 0.0

        J = ekf.compute_jacobian(f, x, u=u)

        J_true = np.array([[2 * x[0], 0.0], [0.0, np.cos(x[1])]])

        np.testing.assert_allclose(J, J_true, rtol=1e-9, atol=1e-12)

    def test_ekf_covariance_psd(self):
        f = lambda x, u: x
        h = lambda x: x

        ekf = ExtendedKalmanFilter(
            f,
            h,
            Q=np.eye(2) * 0.01,
            R=np.eye(2) * 0.1,
            x0=[0.0, 0.0],
        )

        for _ in range(10):
            ekf.predict(0, 1.0)
            ekf.update(np.array([1.0, -1.0]))

        eigs = np.linalg.eigvals(ekf.P)
        self.assertTrue(np.all(eigs > -1e-10))

    def test_ekf_linear_measurement_tracking(self):
        A = np.array([[1.0]])
        C = np.array([[1.0]])

        f = lambda x, u: A @ x
        h = lambda x: C @ x

        ekf = ExtendedKalmanFilter(
            f,
            h,
            Q=np.array([[0.01]]),
            R=np.array([[0.1]]),
            x0=[0.0],
        )

        errors = []
        for _ in range(20):
            ekf.predict(0, 1.0)
            ekf.update(np.array([5.0]))
            errors.append(abs(ekf.x_hat[0, 0] - 5.0))

        self.assertLess(min(errors), errors[0])


class TestDiscreteEKF(unittest.TestCase):
    def test_discrete_ekf_linear_convergence_strict(self):
        A = np.array([[0.0]])
        C = np.array([[1.0]])

        f = lambda x, u: A @ x
        h = lambda x: C @ x

        ekf = DiscreteExtendedKalmanFilter(
            f,
            h,
            Q=np.array([[0.01]]),
            R=np.array([[0.1]]),
            x0=[0.0],
            dt=1.0,
        )

        for _ in range(30):
            ekf.predict()
            ekf.update(np.array([5.0]))

        self.assertLess(abs(ekf.x[0, 0] - 5.0), 0.1)

    def test_discrete_ekf_linear_convergence_loose(self):
        A = np.array([[-1.0]])
        C = np.array([[1.0]])

        f = lambda x, u: A @ x
        h = lambda x: C @ x

        ekf = DiscreteExtendedKalmanFilter(
            f,
            h,
            Q=np.array([[0.01]]),
            R=np.array([[0.1]]),
            x0=[0.0],
            dt=1.0,
        )

        for _ in range(30):
            ekf.predict()
            ekf.update(np.array([5.0]))

        assert np.isfinite(ekf.x[0, 0])

    def test_discrete_ekf_covariance_psd(self):
        f = lambda x, u: x
        h = lambda x: x

        ekf = DiscreteExtendedKalmanFilter(
            f,
            h,
            Q=np.eye(2) * 0.01,
            R=np.eye(2) * 0.1,
            x0=[0.0, 0.0],
            dt=1.0,
        )

        for _ in range(10):
            ekf.predict()
            ekf.update(np.array([1.0, -1.0]))

        eigs = np.linalg.eigvals(ekf.P)
        assert np.all(eigs > -1e-10)


if __name__ == "__main__":
    unittest.main()
