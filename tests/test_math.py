import unittest

import numpy as np
from scipy.linalg import expm

# Import your custom implementation
from core.solver import manual_matrix_exp


class TestMathUtils(unittest.TestCase):
    def test_matrix_exponential_identity(self):
        """Test that e^0 = I"""
        A = np.zeros((3, 3))
        expected = np.eye(3)
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_diagonal(self):
        """Test e^A where A is diagonal (e^diag(a,b) = diag(e^a, e^b))"""
        A = np.diag([1.0, 2.0, 0.5])
        expected = np.diag([np.e, np.e**2, np.e**0.5])
        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_matrix_exponential_rotation(self):
        """Test e^A for a skew-symmetric matrix (Rotation)"""
        theta = np.pi / 2
        A = np.array([[0, -theta], [theta, 0]])

        # Expected result is a rotation matrix of 90 degrees
        expected = np.array([[0, -1], [1, 0]])

        result = manual_matrix_exp(A)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_compare_vs_scipy(self):
        """
        The Ultimate Test:
        Compare your 'First Principles' Taylor Series solver
        against the industry standard scipy.linalg.expm (Pade Approximation).
        """
        # Create a random stable matrix
        np.random.seed(42)
        A = np.random.randn(4, 4) - 2 * np.eye(
            4
        )  # Shift eigenvalues to be negative-ish

        scipy_result = expm(A)
        my_result = manual_matrix_exp(
            A, order=15
        )  # Boost order for high precision check

        # We expect high accuracy (approx 1e-6 or better)
        np.testing.assert_allclose(my_result, scipy_result, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
