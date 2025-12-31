from typing import Any, Callable, Dict, Final, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from core.solver import manual_matrix_exp
from core.state_space import StateSpace
from core.transfer_function import TransferFunction
from helpers.config import DC_MOTOR_DEFAULTS, UKF_MOTOR_PARAMS

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit(cache=True)
def _dc_motor_linear_matrices(
    J: float, b: float, K: float, R: float, L: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    A: NDArray[np.float64] = np.array(
        [[-b / J, K / J], [-K / L, -R / L]],
        dtype=np.float64,
    )

    B: NDArray[np.float64] = np.array(
        [[0.0, -1.0 / J], [1.0 / L, 0.0]],
        dtype=np.float64,
    )
    return A, B


@njit(cache=True)
def _motor_param_estimation_step(
    x: NDArray[Any], voltage: float, K: float, R: float, L: float
) -> Tuple[Union[float, complex], Union[float, complex]]:
    omega: Union[float, complex] = x[0]
    i: Union[float, complex] = x[1]

    J_est: Union[float, complex] = np.exp(x[2])
    b_est: Union[float, complex] = np.exp(x[3])

    dw_dt: Union[float, complex] = (K * i - b_est * omega) / J_est
    di_dt: Union[float, complex] = (voltage - R * i - K * omega) / L

    return dw_dt, di_dt


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
        J: float = DC_MOTOR_DEFAULTS["J"],
        b: float = DC_MOTOR_DEFAULTS["b"],
        K: float = DC_MOTOR_DEFAULTS["K"],
        R: float = DC_MOTOR_DEFAULTS["R"],
        L: float = DC_MOTOR_DEFAULTS["L"],
    ) -> None:
        """
        Initializes the DC Motor model with the specified physical parameters.
        """
        self.params: Dict[str, float] = {"J": J, "b": b, "K": K, "R": R, "L": L}

    def get_open_loop_tf(self, Kp: float, Ki: float, Kd: float) -> TransferFunction:
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

        p_num: List[float] = [K]
        p_den: List[float] = [J * L, (J * R + b * L), (b * R + K**2)]

        c_num: List[float] = [Kd, Kp, Ki]
        c_den: List[float] = [1, 0]

        num: NDArray[np.floating] = np.convolve(c_num, p_num)
        den: NDArray[np.floating] = np.convolve(c_den, p_den)

        return TransferFunction(num, den)

    def get_closed_loop_tf(self, Kp: float, Ki: float, Kd: float) -> TransferFunction:
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

        p_num: List[float] = [K]
        p_den: List[float] = [J * L, (J * R + b * L), (b * R + K**2)]

        c_num: List[float] = [Kd, Kp, Ki]
        c_den: List[float] = [1, 0]

        num: NDArray[np.floating] = np.convolve(c_num, p_num)
        term1: NDArray[np.floating] = np.convolve(c_den, p_den)
        term2: NDArray[np.floating] = np.convolve(c_num, p_num)

        diff: int = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den: NDArray[np.floating] = term1 + term2
        return TransferFunction(num, den)

    def get_disturbance_tf(self, Kp: float, Ki: float, Kd: float) -> TransferFunction:
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

        p_den: List[float] = [J * L, (J * R + b * L), (b * R + K**2)]
        p_num: List[float] = [K]

        c_num: List[float] = [Kd, Kp, Ki]
        c_den: List[float] = [1, 0]

        g_load_num: List[float] = [-L, -R]

        num: NDArray[np.floating] = np.convolve(g_load_num, c_den)

        term1: NDArray[np.floating] = np.convolve(p_den, c_den)
        term2: NDArray[np.floating] = np.convolve(p_num, c_num)

        diff: int = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den: NDArray[np.floating] = term1 + term2
        return TransferFunction(num, den)

    def get_state_space(self) -> StateSpace:
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

        mats: Tuple[NDArray[np.float64], NDArray[np.float64]] = (
            _dc_motor_linear_matrices(J, b, K, R, L)
        )
        A, B = mats
        C: NDArray[np.float64] = np.eye(2)
        D: NDArray[np.float64] = np.zeros((2, 2))

        return StateSpace(A, B, C, D)

    def get_augmented_state_space(self) -> StateSpace:
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

        A_std, _ = _dc_motor_linear_matrices(J, b, K, R, L)
        B_dist: NDArray[np.float64] = np.array([[-1.0 / J], [0.0]])

        A_aug: NDArray[np.float64] = np.vstack(
            (np.hstack((A_std, B_dist)), [[0.0, 0.0, 0.0]])
        )
        B_aug: NDArray[np.float64] = np.array([[0.0], [1.0 / L], [0.0]])

        C_aug: NDArray[np.float64] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        D_aug: NDArray[np.float64] = np.zeros((2, 1))
        return StateSpace(A_aug, B_aug, C_aug, D_aug)

    def get_parameter_estimation_func(
        self,
    ) -> Callable[[NDArray[Any], Union[float, NDArray[np.floating]]], NDArray[Any]]:
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

        def motor_dynamics_4_state(
            x: NDArray[Any], u: Union[float, NDArray[np.floating]]
        ) -> NDArray[Any]:
            voltage = 0.0
            if isinstance(u, np.ndarray):
                if hasattr(u, "ndim") and u.ndim == 2:
                    voltage: float = u[0, 0]
                elif hasattr(u, "__len__"):
                    voltage = u[0]
            else:
                voltage = u

            dw_dt, di_dt = _motor_param_estimation_step(x, voltage, K, R, L)

            zeros: NDArray[Any] = np.zeros_like(dw_dt)
            result: NDArray[Any] = np.array([dw_dt, di_dt, zeros, zeros])

            if result.ndim == 1:
                return result.reshape(-1, 1)
            return result

        return motor_dynamics_4_state

    def get_nonlinear_dynamics(
        self,
    ) -> Tuple[
        Callable[
            [NDArray[np.float64], Union[float, NDArray[Any]], float],
            NDArray[np.float64],
        ],
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ]:
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
        J: float = self.params["J"]
        K: float = self.params["K"]
        R: float = self.params["R"]
        L: float = self.params["L"]

        T_coulomb: Final[float] = UKF_MOTOR_PARAMS["coulomb_friction"]
        b_viscous: Final[float] = UKF_MOTOR_PARAMS["viscous_friction"]

        def motor_stiction_dynamics(
            x: NDArray[np.float64], u: Union[float, NDArray[Any]], dt: float
        ) -> NDArray[np.float64]:
            omega: float = x[0]
            current: float = x[1]

            voltage = 0.0
            if isinstance(u, np.ndarray) and hasattr(u, "__len__"):
                voltage: float = u[0]
            elif isinstance(u, float):
                voltage = u

            T_motor: float = K * current

            T_friction: float = b_viscous * omega + T_coulomb * np.sign(omega)

            if abs(omega) < 0.1 and abs(T_motor) < T_coulomb:
                domega: float = -omega / dt
            else:
                domega = (T_motor - T_friction) / J

            di_dt: float = (voltage - R * current - K * omega) / L

            omega_next: float = omega + domega * dt
            current_next: float = current + di_dt * dt

            return np.array([omega_next, current_next])

        def measurement_model(x) -> NDArray[np.float64]:
            return x

        return motor_stiction_dynamics, measurement_model

    def get_mpc_model(
        self, dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        ss: StateSpace = self.get_state_space()
        A: NDArray[np.float64] = np.asarray(ss.A)
        B: NDArray[np.float64] = np.asarray(ss.B)

        n_states: int = A.shape[0]
        n_inputs: int = B.shape[1]

        M: NDArray[np.float64] = np.zeros((n_states + n_inputs, n_states + n_inputs))
        M[:n_states, :n_states] = A
        M[:n_states, n_states:] = B

        M_exp: NDArray[np.float64] = manual_matrix_exp(M * dt)

        A_d: NDArray[np.float64] = M_exp[:n_states, :n_states]
        B_d: NDArray[np.float64] = M_exp[:n_states, n_states:]

        B_d_voltage: NDArray[np.float64] = B_d[:, 0].reshape(-1, 1)

        return A_d, B_d_voltage
