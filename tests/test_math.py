import unittest

import numpy as np
from scipy.linalg import expm

from core.solver import manual_matrix_exp


class TestMathUtils(unittest.TestCase):
    """
    Unit Tests for core mathematical utilities and solvers.
    """

    def test_matrix_exponential_identity(self):
        """Verifies e^0 = I."""
        A = np.zeros((3, 3))
        expected = np.eye(3)
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_diagonal(self):
        """Verifies e^A where A is diagonal (Result is diag(e^a_ii))."""
        A = np.diag([1.0, 2.0, 0.5])
        expected = np.diag([np.e, np.e**2, np.e**0.5])
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_rotation(self):
        """Verifies e^A for a skew-symmetric matrix (90-degree rotation)."""
        theta = np.pi / 2
        A = np.array([[0, -theta], [theta, 0]])

        expected = np.array([[0, -1], [1, 0]])

        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_compare_vs_scipy(self):
        """
        Compares the custom Taylor Series solver against Scipy's Pade approximation.
        Validates the solver for a random stable matrix.
        """
        np.random.seed(42)
        A = np.random.randn(4, 4) - 2 * np.eye(4)

        scipy_result = expm(A)
        my_result = manual_matrix_exp(A, order=15)

        np.testing.assert_allclose(my_result, scipy_result, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
