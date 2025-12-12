import unittest

import numpy as np

from core.solver import ExactSolver, NonlinearSolver


class TestSolvers(unittest.TestCase):
    """
    Unit Tests for Exact and Nonlinear Solvers.
    """

    # --- ExactSolver Tests ---

    def test_exact_solver_init(self):
        A = [[0]]
        B = [[1]]
        C = [[1]]
        D = [[0]]
        dt = 0.1
        s = ExactSolver(A, B, C, D, dt)

        # e^(0*0.1) = 1
        self.assertEqual(s.Phi[0, 0], 1.0)
        # integral(e^0 * 1) dt = t |0->0.1 = 0.1
        self.assertAlmostEqual(s.Gamma[0, 0], 0.1)

    def test_exact_solver_step_scalar(self):
        s = ExactSolver([[0]], [[1]], [[1]], [[0]], 0.1)
        # x_k+1 = x_k + 0.1 * u_k
        y = s.step(10.0)
        self.assertAlmostEqual(s.x[0, 0], 1.0)  # 0 + 0.1*10
        self.assertAlmostEqual(y, 1.0)

    def test_exact_solver_step_vector(self):
        """Test MIMO system steps."""
        # 2 states, 2 inputs
        A = np.eye(2)
        B = np.eye(2)
        C = np.eye(2)
        D = np.zeros((2, 2))
        s = ExactSolver(A, B, C, D, 1.0)

        # Phi = e^I = e*I
        # Gamma = int(e^t) = e-1

        u = np.array([1, 1])
        s.step(u)

        self.assertEqual(s.x.shape, (2, 1))

    def test_exact_solver_reset(self):
        s = ExactSolver([[0]], [[1]], [[1]], [[0]], 0.1)
        s.step(1.0)
        self.assertNotEqual(s.x[0, 0], 0.0)
        s.reset()
        self.assertEqual(s.x[0, 0], 0.0)

    # --- NonlinearSolver Tests ---

    def test_nonlinear_solver_decay(self):
        """Test x_dot = -x."""
        f = lambda t, x, u: -x
        solver = NonlinearSolver(f, dt_min=0.01, dt_max=0.1, tol=1e-5)
        t, x = solver.solve_adaptive(1.0, np.array([1.0]))

        expected = np.exp(-1.0)
        self.assertAlmostEqual(x[-1, 0], expected, places=4)

    def test_nonlinear_solver_time_dependent(self):
        """Test x_dot = t. x(t) = 0.5*t^2 + x0."""
        f = lambda t, x, u: np.array([t])
        solver = NonlinearSolver(f, tol=1e-5)
        t_end = 2.0
        t, x = solver.solve_adaptive(t_end, np.array([0.0]))

        expected = 0.5 * t_end**2
        self.assertAlmostEqual(x[-1, 0], expected, places=4)

    def test_nonlinear_solver_input_func(self):
        """Test x_dot = u(t). u(t)=t. Same as above."""
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
        # Start at x=0, v=1 -> sin(t)
        t, x = solver.solve_adaptive(np.pi / 2, np.array([0.0, 1.0]))

        # At pi/2, x=1, v=0
        self.assertAlmostEqual(x[-1, 0], 1.0, places=4)
        self.assertAlmostEqual(x[-1, 1], 0.0, places=4)

    def test_nonlinear_solver_stiff_handling(self):
        """Ensure solver reduces dt for stiff segments."""

        # Function that changes rapidly at t=0.5
        def stiff(t, x, u):
            if 0.4 < t < 0.6:
                return np.array([100.0])
            return np.array([1.0])

        solver = NonlinearSolver(stiff, dt_max=0.1)
        t, x = solver.solve_adaptive(1.0, np.array([0.0]))

        # Check if any step size is smaller than dt_max (indicating adaptation)
        dts = np.diff(t)
        self.assertTrue(np.any(dts < 0.09))  # Should drop well below 0.1


if __name__ == "__main__":
    unittest.main()
