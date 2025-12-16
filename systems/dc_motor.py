import numpy as np

from config import DC_MOTOR_DEFAULTS, UKF_MOTOR_PARAMS
from core.solver import manual_matrix_exp
from core.state_space import StateSpace
from core.transfer_function import TransferFunction


class DCMotor:
    """
    Physical model of a DC Motor (Direct Current Motor).

    This class encapsulates the electromechanical dynamics of the motor and provides
    methods for analysis and control design, including Transfer Functions,
    State-Space models, and Non-linear physics for simulation.

    Parameters:
        J (float): Rotor Moment of Inertia (kg*m^2).
        b (float): Viscous Friction coefficient (N*m*s).
        K (float): Back-EMF constant (V/(rad/s)) and Torque constant (N*m/A).
        R (float): Armature Resistance (Ohms).
        L (float): Armature Inductance (Henries).
    """

    def __init__(
        self,
        J=DC_MOTOR_DEFAULTS["J"],
        b=DC_MOTOR_DEFAULTS["b"],
        K=DC_MOTOR_DEFAULTS["K"],
        R=DC_MOTOR_DEFAULTS["R"],
        L=DC_MOTOR_DEFAULTS["L"],
    ):
        """
        Initializes the DC Motor model with the specified physical parameters.
        """
        self.params = {"J": J, "b": b, "K": K, "R": R, "L": L}

    def get_open_loop_tf(self, Kp, Ki, Kd):
        """
        Derives the Open-Loop Transfer Function L(s) = C(s) * P(s).

        This represents the loop gain used for stability analysis (Gain/Phase Margins).

        Plant P(s) (Voltage -> Speed):
            P(s) = K / [ (Js + b)(Ls + R) + K^2 ]

        Controller C(s) (PID):
            C(s) = (Kd*s^2 + Kp*s + Ki) / s

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.

        Returns:
            TransferFunction: The loop transfer function L(s).
        """
        J, b, K, R, L = self.params.values()

        p_num = [K]
        p_den = [J * L, (J * R + b * L), (b * R + K**2)]

        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        num = np.convolve(c_num, p_num)
        den = np.convolve(c_den, p_den)

        return TransferFunction(num, den)

    def get_closed_loop_tf(self, Kp, Ki, Kd):
        """
        Derives the Closed-Loop Transfer Function T(s) for a PID-controlled system.

        The relationship represents:
            Output (Speed) / Reference (Target Speed)

        The derivation combines the Plant Transfer Function P(s) and the Controller C(s):
            T(s) = (P(s)C(s)) / (1 + P(s)C(s))

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.

        Returns:
            TransferFunction: The resulting closed-loop transfer function object.
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
        Derives the Disturbance Transfer Function Td(s) for load torque rejection.

        The relationship represents:
            Output (Speed) / Disturbance (Load Torque)

        This models how the motor speed reacts to an external load when the PID
        controller is actively trying to reject it.

        Args:
            Kp (float): Proportional gain.
            Ki (float): Integral gain.
            Kd (float): Derivative gain.

        Returns:
            TransferFunction: The disturbance transfer function object.
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
        Constructs the Open-Loop State-Space model of the motor.

        State Equations:
            d(omega)/dt = (-b/J)*omega + (K/J)*i - (1/J)*T_load
            d(i)/dt     = (-K/L)*omega - (R/L)*i + (1/L)*V

        State Vector:   [Speed (rad/s), Current (A)]
        Input Vector:   [Voltage (V), Load Torque (Nm)]
        Output Vector:  [Speed (rad/s), Current (A)]

        Returns:
            StateSpace: The linear state-space model.
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

        This model treats the external disturbance (Load Torque) as an additional
        state variable evolving as a random walk (derivative is zero + noise).
        This allows the Kalman Filter to estimate the load torque in real-time.

        Augmented State Vector:
            [Speed (rad/s), Current (A), Disturbance_Bias (Nm)]

        Returns:
            StateSpace: The augmented linear state-space model.
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
        Generates the system dynamics function f(x, u) tailored for the
        Extended Kalman Filter (EKF) to perform joint state and parameter estimation.

        This implementation supports vectorization for particle filters or batched EKF.

        Augmented State Vector:
            [Speed, Current, log(Inertia_J), log(Friction_b)]

        Returns:
            function: A callable f(x, u) computing state derivatives.
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
        Returns the non-linear dynamics and measurement functions tailored for the
        Unscented Kalman Filter (UKF).

        This model includes a "Stiction" (Stick-Slip) friction model, which is
        highly non-linear and difficult for standard EKFs to handle.

        Stiction Logic:
            - If speed is near zero and motor torque < Coulomb friction limit,
              the motor stops dead (acceleration forces speed to zero).
            - Otherwise, standard dynamics apply with both Viscous and Coulomb friction.

        Returns:
            tuple: (f_dynamics, h_measurement) functions.
        """
        J = self.params["J"]
        K = self.params["K"]
        R = self.params["R"]
        L = self.params["L"]

        T_coulomb = UKF_MOTOR_PARAMS["coulomb_friction"]
        b_viscous = UKF_MOTOR_PARAMS["viscous_friction"]

        def motor_stiction_dynamics(x, u, dt):
            omega = x[0]
            current = x[1]

            if hasattr(u, "__len__"):
                voltage = u[0]
            else:
                voltage = u

            T_motor = K * current

            T_friction = b_viscous * omega + T_coulomb * np.sign(omega)

            if abs(omega) < 0.1 and abs(T_motor) < T_coulomb:
                domega = -omega / dt
            else:
                domega = (T_motor - T_friction) / J

            di = (voltage - R * current - K * omega) / L

            omega_next = omega + domega * dt
            current_next = current + di * dt

            return np.array([omega_next, current_next])

        def measurement_model(x):
            return x

        return motor_stiction_dynamics, measurement_model

    def get_mpc_model(self, dt):
        """
        Returns the Discrete-Time Linear Matrices (A_d, B_d) for Model Predictive Control.

        This method performs an Exact Discretization using the Zero-Order Hold (ZOH)
        assumption, which is accurate for digital control systems where the input
        is constant between time steps.

        Computation:
            1. Construct continuous block matrix M = [A B; 0 0].
            2. Compute Matrix Exponential of M * dt.
            3. Extract A_d and B_d from the result.

        Args:
            dt (float): The control time step.

        Returns:
            tuple: (A_d, B_d_voltage) numpy arrays.
        """
        ss = self.get_state_space()
        A = np.array(ss.A)
        B = np.array(ss.B)

        n_states = A.shape[0]
        n_inputs = B.shape[1]

        M = np.zeros((n_states + n_inputs, n_states + n_inputs))
        M[:n_states, :n_states] = A
        M[:n_states, n_states:] = B

        M_exp = manual_matrix_exp(M * dt)

        A_d = M_exp[:n_states, :n_states]
        B_d = M_exp[:n_states, n_states:]

        B_d_voltage = B_d[:, 0].reshape(-1, 1)

        return A_d, B_d_voltage
