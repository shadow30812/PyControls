import unittest

import numpy as np

from systems.dc_motor import DCMotor


class TestDCMotor(unittest.TestCase):
    def setUp(self):
        """Runs before every test. Creates a standard motor."""
        self.motor = DCMotor(J=0.01, b=0.1, K=0.01, R=1, L=0.5)
        self.ss = self.motor.get_state_space()

    def test_dimensions(self):
        """Check if A, B, C, D matrices have correct MIMO shapes."""
        # We expect 2 States (Speed, Current), 2 Inputs (V, Torque), 2 Outputs
        self.assertEqual(self.ss.A.shape, (2, 2))
        self.assertEqual(self.ss.B.shape, (2, 2))
        self.assertEqual(self.ss.C.shape, (2, 2))
        self.assertEqual(self.ss.D.shape, (2, 2))

    def test_stability(self):
        """
        Check if the motor is stable.
        Eigenvalues of 'A' must have negative real parts.
        """
        eigenvalues = np.linalg.eigvals(self.ss.A)
        max_real_part = np.max(eigenvalues.real)

        # If max_real_part is >= 0, the system is unstable (blows up)
        self.assertLess(max_real_part, 0, "System A-matrix has unstable poles!")

    def test_physics_direction(self):
        """
        Steady State Check:
        If we apply +Volts, Speed should be positive.
        x_ss = -inv(A) * B * u
        """
        A = self.ss.A
        B = self.ss.B

        # Apply 10 Volts, 0 Load
        u = np.array([[10], [0]])

        # Calculate Steady State: x = -A^-1 * B * u
        x_ss = -np.linalg.inv(A) @ B @ u

        speed = x_ss[0, 0]
        current = x_ss[1, 0]

        self.assertGreater(speed, 0, "Positive voltage should make motor spin forward")
        self.assertGreater(current, 0, "Positive voltage should draw positive current")


if __name__ == "__main__":
    unittest.main()
