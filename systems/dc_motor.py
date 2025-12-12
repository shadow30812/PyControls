import numpy as np

from core.state_space import StateSpace
from core.transfer_function import TransferFunction


class DCMotor:
    def __init__(self, J=0.01, b=0.1, K=0.01, R=1, L=0.5):
        self.params = {"J": J, "b": b, "K": K, "R": R, "L": L}

    def get_closed_loop_tf(self, Kp, Ki, Kd):
        """Returns T(s) = Ref -> Speed"""
        J, b, K, R, L = self.params.values()

        # Plant G(s)
        p_num = [K]
        p_den = [J * L, (J * R + b * L), (b * R + K**2)]

        # Controller C(s)
        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        # T(s) construction
        num = np.convolve(c_num, p_num)
        term1 = np.convolve(c_den, p_den)
        term2 = np.convolve(c_num, p_num)

        # Pad to add
        diff = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den = term1 + term2
        return TransferFunction(num, den)

    def get_disturbance_tf(self, Kp, Ki, Kd):
        """Returns T_d(s) = Load Torque -> Speed"""
        J, b, K, R, L = self.params.values()

        # Plant Denominator
        p_den = [J * L, (J * R + b * L), (b * R + K**2)]
        # Plant Numerator
        p_num = [K]

        # Controller
        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        # Load G(s) = -(Ls+R) / Delta
        g_load_num = [-L, -R]

        # T_d(s) = (g_load_num * c_den) / (p_den * c_den + p_num * c_num)
        num = np.convolve(g_load_num, c_den)

        term1 = np.convolve(p_den, c_den)
        term2 = np.convolve(p_num, c_num)

        diff = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den = term1 + term2
        return TransferFunction(num, den)

    def get_state_space(self):
        """
        Returns the MIMO State-Space representation of the open-loop motor.

        States:  [Speed (rad/s), Current (A)]
        Inputs:  [Voltage (V), Load Torque (Nm)]
        Outputs: [Speed (rad/s), Current (A)]
        """
        J, b, K, R, L = self.params.values()

        # x_dot = Ax + Bu
        # Rows: dw/dt, di/dt
        # Cols: w, i
        A = [[-b / J, K / J], [-K / L, -R / L]]

        # Cols: Voltage, Load_Torque
        B = [[0, -1 / J], [1 / L, 0]]

        # y = Cx + Du
        # We want to observe both states directly
        C = np.eye(2)
        D = np.zeros((2, 2))

        return StateSpace(A, B, C, D)

    def get_augmented_state_space(self):
        """
        Returns Augmented System for the Kalman Filter.
        States: [Speed, Current, Disturbance_Bias]
        """
        J, b, K, R, L = self.params.values()

        # 1. Standard A Matrix (2x2)
        A_std = np.array([[-b / J, K / J], [-K / L, -R / L]])

        # 2. Effect of Disturbance on standard states (from B matrix col 2)
        # Disturbance affects Speed (-1/J), doesn't affect Current directly (0)
        B_dist_effect = np.array([[-1 / J], [0]])

        # 3. Augment A:
        # [ A_std   B_dist_effect ]
        # [ 0 0     0             ]
        top = np.hstack((A_std, B_dist_effect))
        bottom = np.array([[0, 0, 0]])  # d_dot = 0
        A_aug = np.vstack((top, bottom))

        # 4. Augment B (Inputs: Voltage only):
        # Load Torque is now internal state, so Input is just Voltage.
        B_aug = np.array([[0], [1 / L], [0]])

        # 5. Augment C (Outputs: Speed, Current):
        # We don't measure the disturbance directly.
        C_aug = np.array(
            [
                [1, 0, 0],  # Measure Speed
                [0, 1, 0],  # Measure Current
            ]
        )

        D_aug = np.zeros((2, 1))
        return StateSpace(A_aug, B_aug, C_aug, D_aug)

    def get_parameter_estimation_func(self):
        """
        Returns a function f(x, u) -> x_dot compatible with the EKF.
        State x = [Speed, Current, Inertia_J, Friction_b]
        """
        # We need K, R, L to be "known" constants, but J and b are in the state
        _, _, K, R, L = (
            self.params.values()
        )  # Ignore J, b from params (we estimate them)

        def motor_dynamics_4_state(x, u):
            # Unpack State (supports complex numbers)
            omega = x[0, 0]
            i = x[1, 0]
            J_est = np.exp(x[2, 0])
            b_est = np.exp(x[3, 0])

            voltage = u[0, 0]

            # Physics Equations
            # 1. dw/dt = (K*i - b*w) / J
            dw_dt = (K * i - b_est * omega) / J_est

            # 2. di/dt = (V - R*i - K*w) / L
            di_dt = (voltage - R * i - K * omega) / L

            # dJ/dt = 0 and db/dt = 0 (Constant parameters)
            dJ_dt = 0.0
            db_dt = 0.0

            return np.array([[dw_dt], [di_dt], [dJ_dt], [db_dt]])

        return motor_dynamics_4_state
