import unittest

import numpy as np

from core.solver import ExactSolver, NonlinearSolver, _mat_mul, manual_matrix_exp


class TestSolvers(unittest.TestCase):
    """
    Unit Tests for Exact and Nonlinear Solvers.
    Checks JIT compilation and Vectorization logic.
    """

    def test_exact_solver_init(self):
        A = [[0]]
        B = [[1]]
        C = [[1]]
        D = [[0]]
        dt = 0.1
        s = ExactSolver(A, B, C, D, dt)

        self.assertEqual(s.Phi[0, 0], 1.0)
        self.assertAlmostEqual(s.Gamma[0, 0], 0.1)

    def test_exact_solver_step_scalar(self):
        s = ExactSolver([[0]], [[1]], [[1]], [[0]], 0.1)
        y = s.step(10.0)
        self.assertAlmostEqual(s.x[0, 0], 1.0)
        self.assertAlmostEqual(y, 1.0)

    def test_exact_solver_step_vector(self):
        """Test MIMO system steps."""
        A = np.eye(2)
        B = np.eye(2)
        C = np.eye(2)
        D = np.zeros((2, 2))
        s = ExactSolver(A, B, C, D, 1.0)

        u = np.array([1, 1])
        s.step(u)
        self.assertEqual(s.x.shape, (2, 1))

    def test_exact_solver_reset(self):
        s = ExactSolver([[0]], [[1]], [[1]], [[0]], 0.1)
        s.step(1.0)
        self.assertNotEqual(s.x[0, 0], 0.0)
        s.reset()
        self.assertEqual(s.x[0, 0], 0.0)

    def test_nonlinear_solver_decay(self):
        """Test x_dot = -x."""
        f = lambda t, x, u: -x
        solver = NonlinearSolver(f, dt_min=0.01, dt_max=0.1, tol=1e-5)
        t, x = solver.solve_adaptive(1.0, np.array([1.0]))

        expected = np.exp(-1.0)
        self.assertAlmostEqual(x[-1, 0], expected, places=4)

    def test_nonlinear_solver_vectorized_stages(self):
        """
        Implicitly tests the vectorized inner loop of RK45.
        We use a system size > 1 to ensure matrix ops work.
        """

        def decay_3d(t, x, u):
            rates = np.array([-1, -2, -3])
            if x.ndim == 2:
                x = x.flatten()
            return rates * x

        solver = NonlinearSolver(decay_3d, tol=1e-6)
        x0 = np.array([1.0, 1.0, 1.0])
        t_end = 1.0

        t, x_hist = solver.solve_adaptive(t_end, x0)

        final_x = x_hist[-1]

        self.assertAlmostEqual(final_x[0], np.exp(-1), places=4)
        self.assertAlmostEqual(final_x[1], np.exp(-2), places=4)
        self.assertAlmostEqual(final_x[2], np.exp(-3), places=4)

    def test_nonlinear_solver_input_func(self):
        """Test x_dot = u(t)."""
        f = lambda t, x, u: np.array([u])
        u_func = lambda t: t

        solver = NonlinearSolver(f)
        t, x = solver.solve_adaptive(2.0, np.array([0.0]), u_func=u_func)

        self.assertAlmostEqual(x[-1, 0], 2.0, places=4)

    def test_nonlinear_solver_oscillator(self):
        """Test harmonic oscillator x'' = -x."""

        def f(t, x, u):
            return np.array([x[1], -x[0]])

        solver = NonlinearSolver(f)
        t, x = solver.solve_adaptive(np.pi / 2, np.array([0.0, 1.0]))

        self.assertAlmostEqual(x[-1, 0], 1.0, places=4)
        self.assertAlmostEqual(x[-1, 1], 0.0, places=4)

    def test_nonlinear_solver_stiff_handling(self):
        """Ensure solver reduces dt for stiff segments."""

        def stiff(t, x, u):
            if 0.4 < t < 0.6:
                return np.array([100.0])
            return np.array([1.0])

        solver = NonlinearSolver(stiff, dt_max=0.1)
        t, x = solver.solve_adaptive(1.0, np.array([0.0]))

        dts = np.diff(t)
        self.assertTrue(np.any(dts < 0.09))

    def test_manual_matrix_exp_numba_gate(self):
        A = np.array([[0.1, 0.2], [0.0, -0.1]])
        res = manual_matrix_exp(A)
        self.assertEqual(res.shape, (2, 2))


class TestSolverExtendedContracts(unittest.TestCase):
    """
    Exhaustive contract tests for solver guarantees not covered elsewhere.
    These tests lock in behavior that is deterministic and guaranteed
    by the solver APIs (not numerical coincidences).
    """

    def test_mat_mul_with_views(self):
        A = np.arange(16.0).reshape(4, 4)
        B = np.eye(4)
        A_view = A[::2, ::2]
        B_view = B[::2, ::2]

        expected = A_view @ B_view
        result = _mat_mul(A_view, B_view)

        np.testing.assert_allclose(result, expected)

    def test_mat_mul_zero_columns(self):
        A = np.zeros((3, 0))
        B = np.zeros((0, 2))
        result = _mat_mul(A, B)
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.all(result == 0.0))

    def test_matrix_exp_identity(self):
        I = np.eye(3)
        E = manual_matrix_exp(I * 0.0)
        np.testing.assert_allclose(E, np.eye(3))

    def test_matrix_exp_zero(self):
        Z = np.zeros((4, 4))
        E = manual_matrix_exp(Z)
        np.testing.assert_allclose(E, np.eye(4))

    def test_matrix_exp_small_norm(self):
        A = np.array([[1e-6, 0.0], [0.0, -1e-6]])
        E = manual_matrix_exp(A)
        expected = np.eye(2) + A
        np.testing.assert_allclose(E, expected, rtol=1e-10, atol=1e-12)

    def test_matrix_exp_order_override_convergence(self):
        A = np.array([[0.2, 0.1], [0.0, -0.1]])

        E1 = manual_matrix_exp(A, order=1)
        E2 = manual_matrix_exp(A, order=2)
        E6 = manual_matrix_exp(A, order=6)

        diff_12 = np.linalg.norm(E2 - E1)
        diff_26 = np.linalg.norm(E6 - E2)

        self.assertGreater(diff_12, diff_26)

    def test_exact_solver_mimo_output_shape(self):
        A = np.zeros((2, 2))
        B = np.eye(2)
        C = np.eye(2)
        D = np.eye(2)
        solver = ExactSolver(A, B, C, D, dt=0.5)

        y = solver.step(np.array([1.0, 2.0]))
        self.assertEqual(y.shape, (2,))

    def test_exact_solver_d_matrix_effect(self):
        A = [[0]]
        B = [[0]]
        C = [[1]]
        D = [[2]]
        solver = ExactSolver(A, B, C, D, dt=1.0)

        y = solver.step(3.0)
        self.assertEqual(y, 6.0)

    def test_exact_solver_multiple_steps_deterministic(self):
        solver = ExactSolver([[0]], [[1]], [[1]], [[0]], 0.1)
        y1 = solver.step(1.0)
        y2 = solver.step(1.0)
        self.assertAlmostEqual(y2, 2 * y1)

    def test_exact_solver_zero_dynamics(self):
        solver = ExactSolver([[0]], [[0]], [[1]], [[0]], 1.0)
        y = solver.step(10.0)
        self.assertEqual(y, 0.0)

    def test_nonlinear_solver_dt_min_enforced(self):
        def stiff(t, x, u):
            return np.array([1e6])

        solver = NonlinearSolver(stiff, dt_min=1e-3, dt_max=1.0, tol=1e-12)
        t, _ = solver.solve_adaptive(0.01, np.array([0.0]))
        dts = np.diff(t)
        self.assertTrue(np.all(dts >= 1e-3))

    def test_nonlinear_solver_dt_max_reasonable(self):
        def slow(t, x, u):
            return np.array([0.0])

        dt_max = 0.05
        solver = NonlinearSolver(slow, dt_min=1e-4, dt_max=dt_max)

        t, _ = solver.solve_adaptive(0.5, np.array([0.0]))
        dts = np.diff(t)

        self.assertLessEqual(np.max(dts), dt_max * 1.01)

    def test_nonlinear_solver_zero_error_branch(self):
        def constant(t, x, u):
            return np.array([1.0])

        solver = NonlinearSolver(constant)
        t, _ = solver.solve_adaptive(0.2, np.array([0.0]))
        self.assertGreater(np.max(np.diff(t)), solver.dt_min)

    def test_nonlinear_solver_column_vector_input(self):
        def decay(t, x, u):
            return -x

        solver = NonlinearSolver(decay)
        t, x = solver.solve_adaptive(1.0, np.array([[1.0]]))
        self.assertEqual(x.shape[1], 1)

    def test_nonlinear_solver_time_monotonicity(self):
        def f(t, x, u):
            return -x

        solver = NonlinearSolver(f)
        t, _ = solver.solve_adaptive(1.0, np.array([1.0]))
        self.assertTrue(np.all(np.diff(t) > 0))


if __name__ == "__main__":
    unittest.main()
