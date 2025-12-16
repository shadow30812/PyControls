import unittest

import numpy as np

from core.mpc import ModelPredictiveControl
from core.ukf import UnscentedKalmanFilter


class TestAdvancedControl(unittest.TestCase):
    """
    Unit Tests for advanced control and estimation strategies (UKF & MPC).
    Updated to support the new ADMM (Linear) and iLQR (Nonlinear) solvers.
    """

    def test_ukf_sigma_points_generation(self):
        """Verify sigma points are generated symmetrically around the mean."""
        x0 = np.array([1.0, 2.0])
        P0 = np.eye(2)

        f = lambda x, u, dt: x
        h = lambda x: x
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.1

        ukf = UnscentedKalmanFilter(f, h, Q, R, x0, P0)

        sigmas = ukf._generate_sigma_points(x0, P0)

        self.assertEqual(sigmas.shape, (5, 2))
        np.testing.assert_array_equal(sigmas[0], x0)

        mean_rest = np.mean(sigmas[1:], axis=0)
        np.testing.assert_array_almost_equal(mean_rest, x0)

    def test_ukf_convergence_linear(self):
        """Verify UKF converges on a simple linear problem."""
        f = lambda x, u, dt: x
        h = lambda x: x

        x0_guess = np.array([0.0])
        P0 = np.eye(1) * 1.0
        Q = np.eye(1) * 0.01
        R = np.eye(1) * 0.1

        ukf = UnscentedKalmanFilter(f, h, Q, R, x0_guess, P0)

        for _ in range(20):
            z = np.array([5.0])
            ukf.predict(0, 0.1)
            ukf.update(z)

        self.assertAlmostEqual(ukf.x[0], 5.0, places=1)

    def test_ukf_nonlinear_transform(self):
        """Verify UKF handles a non-linear transform better than linearization."""
        f = lambda x, u, dt: x**2
        h = lambda x: x

        x0 = np.array([2.0])
        P0 = np.eye(1) * 0.1

        ukf = UnscentedKalmanFilter(f, h, np.eye(1) * 0.01, np.eye(1) * 0.01, x0, P0)
        ukf.predict(0, 0.1)

        self.assertGreater(ukf.x[0], 4.05)

    def test_mpc_linear_admm_selection(self):
        """Test that providing A, B matrices triggers the ADMM solver."""
        A = np.eye(2)
        B = np.eye(2)
        mpc = ModelPredictiveControl(A=A, B=B, horizon=5)

        self.assertEqual(mpc.mode, "linear")
        self.assertTrue(hasattr(mpc, "H_inv"), "ADMM pre-computation missing")

    def test_mpc_nonlinear_ilqr_selection(self):
        """Test that providing model_func triggers the iLQR solver."""
        f = lambda x, u, dt: x + u
        mpc = ModelPredictiveControl(model_func=f, x0=np.array([0.0]), horizon=5)

        self.assertEqual(mpc.mode, "nonlinear")
        self.assertIsNone(mpc.A)

    def test_mpc_admm_optimization(self):
        """Test ADMM solver accuracy on a simple integrator."""
        dt = 1.0
        A = np.array([[1.0]])
        B = np.array([[dt]])

        mpc = ModelPredictiveControl(
            A=A,
            B=B,
            horizon=5,
            dt=dt,
            Q=[[1.0]],
            R=[[0.1]],
            u_min=-10,
            u_max=10,
        )

        x0 = np.array([0.0])
        x_ref = np.array([10.0])

        u_opt = mpc.optimize(x0, x_ref, iterations=20)

        self.assertGreater(u_opt[0], 2.0)

    def test_mpc_ilqr_optimization(self):
        """Test iLQR solver accuracy on the same integrator."""
        f = lambda x, u, dt: x + u * dt

        mpc = ModelPredictiveControl(
            model_func=f,
            x0=np.array([0.0]),
            horizon=5,
            dt=1.0,
            Q=[[1.0]],
            R=[[0.1]],
            u_min=-10,
            u_max=10,
        )

        x0 = np.array([0.0])
        x_ref = np.array([10.0])

        u_opt = mpc.optimize(x0, x_ref, iterations=10)

        self.assertGreater(u_opt[0], 2.0)

    def test_mpc_constraints(self):
        """Test that constraints are respected by the solvers."""
        A = np.array([[1.0]])
        B = np.array([[1.0]])

        limit = 2.0
        mpc = ModelPredictiveControl(A=A, B=B, horizon=5, u_min=-limit, u_max=limit)

        x0 = np.array([0.0])
        x_ref = np.array([100.0])

        u_opt = mpc.optimize(x0, x_ref)

        self.assertLessEqual(u_opt[0], limit + 1e-4)


if __name__ == "__main__":
    unittest.main()
