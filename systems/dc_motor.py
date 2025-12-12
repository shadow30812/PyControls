import numpy as np

from core.state_space import StateSpace
from core.transfer_function import TransferFunction


class DCMotor:
    """
    Physical model of a DC Motor.

    Parameters:
    - J: Rotor Inertia
    - b: Viscous Friction
    - K: Back-EMF / Torque Constant
    - R: Armature Resistance
    - L: Armature Inductance
    """

    def __init__(self, J=0.01, b=0.1, K=0.01, R=1, L=0.5):
        self.params = {"J": J, "b": b, "K": K, "R": R, "L": L}

    def get_closed_loop_tf(self, Kp, Ki, Kd):
        """
        Derives the Closed-Loop Transfer Function T(s) = Output/Reference.
        Ref -> Voltage -> Motor -> Speed.
        """
        J, b, K, R, L = self.params.values()

        p_num = [K]
        p_den = [J * L, (J * R + b * L), (b * R + K**2)]

        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        num = np.convolve(c_num, p_num)
        term1 = np.convolve(c_den, p_den)
        term2 = np.convolve(c_num, p_num)

        diff = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den = term1 + term2
        return TransferFunction(num, den)

    def get_disturbance_tf(self, Kp, Ki, Kd):
        """
        Derives the Disturbance Transfer Function Td(s) = Output/Disturbance.
        Load Torque -> Motor -> Speed (with Controller fighting it).
        """
        J, b, K, R, L = self.params.values()

        p_den = [J * L, (J * R + b * L), (b * R + K**2)]
        p_num = [K]

        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        g_load_num = [-L, -R]

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
        Constructs the Open-Loop State-Space model.

        States:  [Speed (rad/s), Current (A)]
        Inputs:  [Voltage (V), Load Torque (Nm)]
        Outputs: [Speed (rad/s), Current (A)]
        """
        J, b, K, R, L = self.params.values()

        A = [[-b / J, K / J], [-K / L, -R / L]]

        B = [[0, -1 / J], [1 / L, 0]]

        C = np.eye(2)
        D = np.zeros((2, 2))

        return StateSpace(A, B, C, D)

    def get_augmented_state_space(self):
        """
        Constructs an Augmented State-Space model for the Kalman Filter.
        Assumes the disturbance is a constant bias state (random walk).

        States: [Speed, Current, Disturbance_Bias]
        """
        J, b, K, R, L = self.params.values()

        A_std = np.array([[-b / J, K / J], [-K / L, -R / L]])

        B_dist_effect = np.array([[-1 / J], [0]])

        top = np.hstack((A_std, B_dist_effect))
        bottom = np.array([[0, 0, 0]])
        A_aug = np.vstack((top, bottom))

        B_aug = np.array([[0], [1 / L], [0]])

        C_aug = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ]
        )

        D_aug = np.zeros((2, 1))
        return StateSpace(A_aug, B_aug, C_aug, D_aug)

    def get_parameter_estimation_func(self):
        """
        Returns the system dynamics function f(x, u) for parameter estimation.
        Designed for the Extended Kalman Filter (EKF).

        State Vector: [Speed, Current, log(J), log(b)]
        """
        _, _, K, R, L = self.params.values()

        def motor_dynamics_4_state(x, u):
            omega = x[0, 0]
            i = x[1, 0]
            J_est = np.exp(x[2, 0])
            b_est = np.exp(x[3, 0])

            voltage = u[0, 0]

            dw_dt = (K * i - b_est * omega) / J_est

            di_dt = (voltage - R * i - K * omega) / L

            dJ_dt = 0.0
            db_dt = 0.0

            return np.array([[dw_dt], [di_dt], [dJ_dt], [db_dt]])

        return motor_dynamics_4_state
