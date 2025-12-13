import unittest

import numpy as np

try:
    from scipy.linalg import expm

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from core.exceptions import ConvergenceError
from core.math_utils import (
    Differentiation,
    Root,
    implicit_mul,
    make_func,
    make_system_func,
    preprocess_power,
)
from core.solver import manual_matrix_exp


class TestMathUtils(unittest.TestCase):
    """
    Comprehensive Unit Tests for core mathematical utilities.
    """

    # --- Matrix Exponential Tests ---

    def test_matrix_exponential_identity(self):
        """e^0 = I"""
        A = np.zeros((3, 3))
        expected = np.eye(3)
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_diagonal(self):
        """e^diag(a,b) = diag(e^a, e^b)"""
        A = np.diag([1.0, 2.0, -1.0])
        expected = np.diag([np.e, np.e**2, np.e**-1])
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_nilpotent(self):
        """e^A where A^k=0. e^A = I + A + ..."""
        A = np.array([[0, 1], [0, 0]], dtype=float)
        expected = np.eye(2) + A
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_rotation(self):
        """e^A for skew-symmetric matrix (Rotation)."""
        theta = np.pi / 2
        A = np.array([[0, -theta], [theta, 0]])
        expected = np.array([[0, -1], [1, 0]])
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_compare_vs_scipy_random(self):
        """Compare vs Scipy for random stable matrices."""
        if not SCIPY_AVAILABLE:
            self.skipTest("SciPy not installed, skipping comparison test")

        np.random.seed(42)
        for _ in range(3):
            A = np.random.randn(5, 5) - 2 * np.eye(5)
            scipy_result = expm(A)
            my_result = manual_matrix_exp(A, order=20)
            np.testing.assert_allclose(my_result, scipy_result, rtol=1e-5, atol=1e-6)

    # --- String Parsing Tests ---

    def test_implicit_multiplication_basics(self):
        self.assertEqual(implicit_mul("3x"), "3*x")
        self.assertEqual(implicit_mul("2sin(x)"), "2*sin(x)")
        self.assertEqual(implicit_mul("x2"), "x*2")
        self.assertEqual(implicit_mul("(x+1)(y-2)"), "(x+1)*(y-2)")

    def test_implicit_multiplication_edge_cases(self):
        self.assertEqual(implicit_mul("x(t)"), "x(t)")
        self.assertEqual(implicit_mul("sin(x)cos(x)"), "sin(x)*cos(x)")
        self.assertEqual(implicit_mul("3(x)"), "3*(x)")
        self.assertEqual(implicit_mul(""), "")

    def test_preprocess_power(self):
        self.assertEqual(preprocess_power("x^2"), "x**2")
        self.assertEqual(preprocess_power("x^(2+y)"), "x**(2+y)")
        self.assertEqual(preprocess_power("e^(-t)"), "e**(-t)")

    def test_make_func_valid(self):
        f = make_func("t^2 + log(t)", "t")
        self.assertAlmostEqual(f(np.e), np.e**2 + 1.0)
        self.assertEqual(f(1j), (1j) ** 2 + 0 + 1.5707963267948966j)

    def test_make_func_invalid(self):
        """Should return 0.0 on evaluation error."""
        f = make_func("1/0")
        self.assertEqual(f(0), 0.0)
        f = make_func("unknown_function(x)")
        self.assertEqual(f(0), 0.0)

    def test_make_system_func_numpy(self):
        f = make_system_func("[sin(t), cos(t)]")
        res = f(0, np.array([0, 0]), 0)
        np.testing.assert_array_equal(res, np.array([0.0, 1.0]))

    def test_make_system_func_state_access(self):
        f = make_system_func("[x[1], -x[0]]")
        x = np.array([10.0, 5.0])
        res = f(0, x, 0)
        np.testing.assert_array_equal(res, np.array([5.0, -10.0]))

    # --- Differentiation Tests ---

    def test_diff_real(self):
        diff = Differentiation()
        func = lambda x: x**3
        val = diff.real_diff(func, 2.0)
        self.assertAlmostEqual(val, 12.0, places=8)

    def test_diff_fallback(self):
        """Test fallback to finite difference when complex step fails."""
        diff = Differentiation()

        def strict_float_func(x):
            if isinstance(x, complex):
                raise TypeError("No complex numbers allowed")
            return x**2

        val = diff.real_diff(strict_float_func, 2.0)
        self.assertAlmostEqual(val, 4.0, places=4)

    # --- Root Finding Tests ---

    def test_root_brent_standard(self):
        r = Root()
        f = lambda x: x**2 - 4
        res = r.find_root(f, -1, 3)
        self.assertAlmostEqual(res, 2.0)

    def test_root_brent_bad_bracket(self):
        """Should raise ValueError if no sign change."""
        r = Root()
        f = lambda x: x**2 + 1
        with self.assertRaises(ValueError):
            r.brent_root(f, 0, 5)

    def test_root_newton_convergence(self):
        r = Root()
        f = lambda x: (x - 3) ** 2
        res = r.find_root(f, 10.0)
        self.assertAlmostEqual(res, 3.0, places=5)

    def test_root_newton_maxiter_exception(self):
        """Ensure it raises ConvergenceError if maxiter reached."""
        r = Root()
        f = lambda x: x**2 + 1
        with self.assertRaises(ConvergenceError):
            r.newton_root(f, 0.0, maxiter=10)


if __name__ == "__main__":
    unittest.main()
