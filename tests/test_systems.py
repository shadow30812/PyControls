import unittest

import numpy as np

from systems.dc_motor import DCMotor


class TestDCMotor(unittest.TestCase):
    """
    Unit Tests for the DC Motor system implementation.
    """

    def setUp(self):
        """Initializes a standard motor instance for testing."""
        self.motor = DCMotor(J=0.01, b=0.1, K=0.01, R=1, L=0.5)
        self.ss = self.motor.get_state_space()

    def test_dimensions(self):
        """Checks if State-Space matrices (A, B, C, D) have valid MIMO dimensions."""
        self.assertEqual(self.ss.A.shape, (2, 2))
        self.assertEqual(self.ss.B.shape, (2, 2))
        self.assertEqual(self.ss.C.shape, (2, 2))
        self.assertEqual(self.ss.D.shape, (2, 2))

    def test_stability(self):
        """
        Checks if the system is stable.
        A passive motor should have eigenvalues with negative real parts.
        """
        eigenvalues = np.linalg.eigvals(self.ss.A)
        max_real_part = np.max(eigenvalues.real)

        self.assertLess(max_real_part, 0, "System A-matrix has unstable poles!")

    def test_physics_direction(self):
        """
        Checks steady-state physics.
        Applying positive voltage should result in positive speed and current.
        """
        A = self.ss.A
        B = self.ss.B

        u = np.array([[10], [0]])

        x_ss = -np.linalg.inv(A) @ B @ u

        speed = x_ss[0, 0]
        current = x_ss[1, 0]

        self.assertGreater(speed, 0, "Positive voltage should make motor spin forward")
        self.assertGreater(current, 0, "Positive voltage should draw positive current")


if __name__ == "__main__":
    unittest.main()
