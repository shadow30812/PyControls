import unittest

import numpy as np

from core.control_utils import Check, PIDController, dlqr, solve_discrete_riccati


class TestDiscreteRiccati(unittest.TestCase):
    """
    Exhaustive tests for the discrete Riccati solver.
    Tests only guaranteed mathematical properties.
    """

    def setUp(self):
        self.A = np.array([[1.0]])
        self.B = np.array([[1.0]])
        self.Q = np.array([[1.0]])
        self.R = np.array([[1.0]])

    def test_solution_shape(self):
        P = solve_discrete_riccati(self.A, self.B, self.Q, self.R)
        self.assertEqual(P.shape, (1, 1))

    def test_solution_symmetric(self):
        P = solve_discrete_riccati(self.A, self.B, self.Q, self.R)
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_solution_finite(self):
        P = solve_discrete_riccati(self.A, self.B, self.Q, self.R)
        self.assertTrue(np.all(np.isfinite(P)))

    def test_solution_positive_semidefinite(self):
        P = solve_discrete_riccati(self.A, self.B, self.Q, self.R)
        eigs = np.linalg.eigvals(P)
        self.assertTrue(np.all(eigs.real >= -1e-10))

    def test_zero_cost_returns_zero(self):
        Q = np.zeros((1, 1))
        P = solve_discrete_riccati(self.A, self.B, Q, self.R)
        np.testing.assert_array_almost_equal(P, np.zeros_like(P))

    def test_deterministic(self):
        P1 = solve_discrete_riccati(self.A, self.B, self.Q, self.R)
        P2 = solve_discrete_riccati(self.A, self.B, self.Q, self.R)
        np.testing.assert_array_equal(P1, P2)


class TestDiscreteRiccatiRegression(unittest.TestCase):
    """
    Regression tests for discrete-time Riccati solver and LQR.
    These tests lock in numerical correctness and stability guarantees.
    """

    def test_riccati_solution_properties(self):
        """
        Regression test:
        - P must be symmetric
        - P must be positive semi-definite
        - Closed-loop system must be stable
        """

        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.eye(2)
        R = np.array([[1.0]])

        P = solve_discrete_riccati(A, B, Q, R)

        np.testing.assert_array_almost_equal(P, P.T, decimal=10)

        eigvals = np.linalg.eigvals(P)
        self.assertTrue(np.all(eigvals >= -1e-10))

        K = dlqr(A, B, Q, R)
        A_cl = A - B @ K

        cl_eigs = np.linalg.eigvals(A_cl)

        self.assertTrue(np.all(np.abs(cl_eigs) < 1.0))


class TestDLQR(unittest.TestCase):
    """
    Tests for discrete-time LQR gain computation.
    """

    def setUp(self):
        self.A = np.array([[1.0]])
        self.B = np.array([[1.0]])
        self.Q = np.array([[1.0]])
        self.R = np.array([[1.0]])

    def test_gain_shape(self):
        K = dlqr(self.A, self.B, self.Q, self.R)
        self.assertEqual(K.shape, (1, 1))

    def test_gain_finite(self):
        K = dlqr(self.A, self.B, self.Q, self.R)
        self.assertTrue(np.all(np.isfinite(K)))

    def test_closed_loop_stability(self):
        """
        For scalar stable LQR, closed-loop pole magnitude < 1 is guaranteed.
        """
        K = dlqr(self.A, self.B, self.Q, self.R)
        A_cl = self.A - self.B @ K
        eigs = np.linalg.eigvals(A_cl)
        self.assertTrue(np.all(np.abs(eigs) < 1.0))

    def test_deterministic(self):
        K1 = dlqr(self.A, self.B, self.Q, self.R)
        K2 = dlqr(self.A, self.B, self.Q, self.R)
        np.testing.assert_array_equal(K1, K2)


class TestControllabilityObservability(unittest.TestCase):
    def test_controllable_system(self):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])

        self.assertTrue(Check().is_controllable(A, B))

    def test_uncontrollable_system(self):
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[1], [0]])

        self.assertFalse(Check().is_controllable(A, B))

    def test_observable_system(self):
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[1, 0]])

        self.assertTrue(Check().is_observable(A, B))

    def test_unobservable_system(self):
        A = np.array([[1, 0], [0, 2]])
        B = np.array([[0, 1]])

        self.assertFalse(Check().is_observable(A, B))

    def test_controllability_matrix_shape(self):
        A = np.eye(3)
        B = np.ones((3, 1))

        Cm = Check().controllability_matrix(A, B)
        self.assertEqual(Cm.shape, (3, 3))

    def test_observability_matrix_shape(self):
        A = np.eye(3)
        B = np.ones((1, 3))

        Om = Check().observability_matrix(A, B)
        self.assertEqual(Om.shape, (3, 3))

    def test_basic_identity(self):
        I = np.eye(4)
        self.assertEqual(Check()._matrix_rank(I), 4)

    def test_rank_deficient(self):
        M = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        self.assertEqual(Check()._matrix_rank(M), 2)

    def test_atol_dominance(self):
        M = np.eye(3) * 1e-8

        rank = Check()._matrix_rank(M, atol=1e-7, rtol=0)
        self.assertEqual(rank, 0)

        rank = Check()._matrix_rank(M, atol=1e-9, rtol=0)
        self.assertEqual(rank, 3)

    def test_rtol_dominance(self):
        S = np.diag([1000, 1000, 0.1])

        rank = Check()._matrix_rank(S, atol=1e-15, rtol=1e-3)
        self.assertEqual(rank, 2)

    def test_numpy_consistency(self):
        rng = np.random.default_rng(42)
        M = rng.random((10, 10))

        np_rank = np.linalg.matrix_rank(M)
        my_rank = Check()._matrix_rank(M)

        self.assertEqual(my_rank, np_rank)


class TestPIDController(unittest.TestCase):
    """
    Exhaustive tests for PIDController guarantees.
    Tests only behaviors explicitly guaranteed by implementation.
    """

    def test_zero_gains_zero_output(self):
        pid = PIDController(0.0, 0.0, 0.0)
        u = pid.update(measurement=1.0, setpoint=1.0, dt=0.1)
        self.assertEqual(u, 0.0)

    def test_proportional_only(self):
        pid = PIDController(Kp=2.0, Ki=0.0, Kd=0.0)
        u = pid.update(measurement=1.0, setpoint=2.0, dt=0.1)
        self.assertAlmostEqual(u, 2.0)

    def test_integral_accumulation(self):
        pid = PIDController(Kp=0.0, Ki=1.0, Kd=0.0)
        pid.update(0.0, 1.0, 0.1)
        u = pid.update(0.0, 1.0, 0.1)
        self.assertAlmostEqual(u, 0.2)

    def test_derivative_on_measurement_response(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0, derivative_on_measurement=True)

        u1 = pid.update(measurement=0.0, setpoint=0.0, dt=0.1)
        u2 = pid.update(measurement=1.0, setpoint=1.0, dt=0.1)

        self.assertNotEqual(u2, u1)

    def test_derivative_on_error_response(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0, derivative_on_measurement=False)

        u1 = pid.update(measurement=0.0, setpoint=1.0, dt=0.1)
        u2 = pid.update(measurement=0.5, setpoint=1.0, dt=0.1)

        self.assertNotEqual(u2, u1)

    def test_derivative_on_measurement_sign(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0, derivative_on_measurement=True)
        pid.reset()

        u1 = pid.update(measurement=0.0, setpoint=0.0, dt=0.1)
        u2 = pid.update(measurement=1.0, setpoint=0.0, dt=0.1)

        self.assertLess(u2, u1)

    def test_derivative_on_error_sign(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0, derivative_on_measurement=False)
        pid.reset()

        u1 = pid.update(measurement=0.0, setpoint=0.0, dt=0.1)
        u2 = pid.update(measurement=1.0, setpoint=0.0, dt=0.1)

        self.assertLess(u2, u1)

    def test_derivative_filter_smoothing(self):
        pid = PIDController(Kp=0.0, Ki=0.0, Kd=1.0, tau=0.5)

        u1 = pid.update(0.0, 0.0, 0.1)
        u2 = pid.update(10.0, 10.0, 0.1)

        self.assertTrue(abs(u2) < abs((10.0 / 0.1)))

    def test_output_limits(self):
        pid = PIDController(Kp=10.0, Ki=0.0, Kd=0.0, output_limits=(-1.0, 1.0))
        u = pid.update(measurement=0.0, setpoint=1.0, dt=0.1)
        self.assertEqual(u, 1.0)

    def test_output_limits_after_reset(self):
        pid = PIDController(
            Kp=10.0,
            Ki=0.0,
            Kd=0.0,
            output_limits=(-1.0, 1.0),
        )
        pid.reset()

        u = pid.update(measurement=0.0, setpoint=10.0, dt=0.1)
        self.assertLessEqual(u, 1.0)
        self.assertGreaterEqual(u, -1.0)

    def test_reset_clears_internal_state(self):
        pid = PIDController(Kp=1.0, Ki=1.0, Kd=1.0)
        pid.update(0.0, 1.0, 0.1)
        pid.reset()
        self.assertEqual(pid.integral_error, 0.0)
        self.assertEqual(pid.prev_value, 0.0)
        self.assertEqual(pid.prev_derivative, 0.0)

    def test_zero_dt_returns_zero(self):
        pid = PIDController(1.0, 1.0, 1.0)
        u = pid.update(1.0, 1.0, 0.0)
        self.assertEqual(u, 0.0)

    def test_zero_dt_returns_zero_after_reset(self):
        pid = PIDController(1.0, 1.0, 1.0)
        pid.reset()

        u = pid.update(measurement=1.0, setpoint=2.0, dt=0.0)
        self.assertEqual(u, 0.0)
