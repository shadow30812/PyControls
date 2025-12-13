import unittest

import numpy as np

from core.mpc import ModelPredictiveControl
from core.ukf import UnscentedKalmanFilter


class TestAdvancedControl(unittest.TestCase):
    """
    Unit Tests for advanced control and estimation strategies (UKF & MPC).
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

    def test_mpc_horizon_prediction(self):
        """Test the internal prediction model simulates correctly."""
        f = lambda x, u, dt: x + u * dt
        x0 = np.array([0.0])

        mpc = ModelPredictiveControl(f, x0, horizon=5, dt=1.0)

        u_seq = np.ones((5, 1))
        traj = mpc._predict_trajectory(x0, u_seq)

        self.assertEqual(len(traj), 6)
        self.assertAlmostEqual(traj[-1, 0], 5.0)

    def test_mpc_cost_function(self):
        """Test cost calculation logic."""
        f = lambda x, u, dt: x + u
        x0 = np.array([0.0])

        mpc = ModelPredictiveControl(
            f, x0, horizon=2, dt=1.0, Q=np.eye(1), R=np.zeros((1, 1))
        )

        x_ref = np.array([10.0])
        u_seq = np.array([[0.0], [0.0]])

        cost = mpc._cost_function(x0, u_seq, x_ref)
        self.assertAlmostEqual(cost, 200.0)

    def test_mpc_optimization_unconstrained(self):
        """Test if optimizer finds solution for simple tracking."""
        f = lambda x, u, dt: x + u
        x0 = np.array([0.0])
        x_ref = np.array([5.0])

        mpc = ModelPredictiveControl(
            f, x0, horizon=1, dt=1.0, u_min=-100, u_max=100, Q=[[1.0]], R=[[0.0]]
        )

        u_opt = mpc.optimize(x0, x_ref, learning_rate=0.5, iterations=100)

        self.assertAlmostEqual(u_opt[0], 5.0, places=2)

    def test_mpc_optimization_constrained(self):
        """Test if optimizer respects u_max constraints."""
        f = lambda x, u, dt: x + u
        x0 = np.array([0.0])
        x_ref = np.array([10.0])

        mpc = ModelPredictiveControl(f, x0, horizon=1, dt=1.0, u_min=-2, u_max=2)

        u_opt = mpc.optimize(x0, x_ref, learning_rate=0.5, iterations=50)

        self.assertAlmostEqual(u_opt[0], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
