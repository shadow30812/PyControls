import numpy as np

from config import DC_MOTOR_DEFAULTS, UKF_MOTOR_PARAMS
from core.solver import manual_matrix_exp
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

    def __init__(
        self,
        J=DC_MOTOR_DEFAULTS["J"],
        b=DC_MOTOR_DEFAULTS["b"],
        K=DC_MOTOR_DEFAULTS["K"],
        R=DC_MOTOR_DEFAULTS["R"],
        L=DC_MOTOR_DEFAULTS["L"],
    ):
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

        Supports VECTORIZATION: x can be (4, 1) or (4, N).
        State Vector: [Speed, Current, log(J), log(b)]
        """
        _, _, K, R, L = self.params.values()

        def motor_dynamics_4_state(x, u):
            omega = x[0]
            i = x[1]
            J_est = np.exp(x[2])
            b_est = np.exp(x[3])

            if hasattr(u, "ndim") and u.ndim == 2:
                voltage = u[0, 0]
            elif hasattr(u, "__len__"):
                voltage = u[0]
            else:
                voltage = u

            dw_dt = (K * i - b_est * omega) / J_est
            di_dt = (voltage - R * i - K * omega) / L

            dJ_dt = np.zeros_like(omega)
            db_dt = np.zeros_like(omega)

            result = np.array([dw_dt, di_dt, dJ_dt, db_dt])
            if result.ndim == 1:
                return result.reshape(-1, 1)
            return result

        return motor_dynamics_4_state

    def get_nonlinear_dynamics(self):
        """
        Returns (f, h) for UKF with a "Stiction" (Stick-Slip) friction model.

        State: [Speed (rad/s), Current (A)]
        Measurement: [Speed, Current]
        """
        J = self.params["J"]
        K = self.params["K"]
        R = self.params["R"]
        L = self.params["L"]

        # Non-linear Friction parameters
        T_coulomb = UKF_MOTOR_PARAMS["coulomb_friction"]
        b_viscous = UKF_MOTOR_PARAMS["viscous_friction"]

        def motor_stiction_dynamics(x, u, dt):
            omega = x[0]
            current = x[1]

            # Input handling
            if hasattr(u, "__len__"):
                voltage = u[0]
            else:
                voltage = u

            # --- Non-Linear Stiction Model ---
            # Torque produced by motor
            T_motor = K * current

            # Friction Torque: Viscous + Coulomb (Sign(omega))
            # We use a slight smoothing tanh to help stability, or raw sign
            T_friction = b_viscous * omega + T_coulomb * np.sign(omega)

            # Stiction Logic: If moving slowly and torque < static friction, we stop.
            if abs(omega) < 0.1 and abs(T_motor) < T_coulomb:
                domega = -omega / dt  # Kill speed instantly (in one step)
            else:
                domega = (T_motor - T_friction) / J

            di = (voltage - R * current - K * omega) / L

            # Euler Step
            omega_next = omega + domega * dt
            current_next = current + di * dt

            return np.array([omega_next, current_next])

        def measurement_model(x):
            return x  # We measure both states

        return motor_stiction_dynamics, measurement_model

    def get_mpc_model(self, dt):
        """
        Returns Linear Discrete Matrices (A_d, B_d) for MPC.
        Uses Exact Zero-Order Hold (ZOH) discretization to match physics.
        """
        # 1. Get Continuous Matrices
        ss = self.get_state_space()
        A = np.array(ss.A)
        B = np.array(ss.B)

        n_states = A.shape[0]
        n_inputs = B.shape[1]

        # 2. Exact Discretization (ZOH)
        # Construct [A B; 0 0] matrix
        M = np.zeros((n_states + n_inputs, n_states + n_inputs))
        M[:n_states, :n_states] = A
        M[:n_states, n_states:] = B

        # Matrix Exponential
        M_exp = manual_matrix_exp(M * dt)

        # Extract discrete matrices
        A_d = M_exp[:n_states, :n_states]
        B_d = M_exp[:n_states, n_states:]

        # DCMotor B is 2x2 [[0, -1/J], [1/L, 0]] (Voltage, Disturbance).
        # MPC controls Input 0 (Voltage).
        B_d_voltage = B_d[:, 0].reshape(-1, 1)

        return A_d, B_d_voltage
