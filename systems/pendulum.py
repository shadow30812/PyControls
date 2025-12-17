import numpy as np

from config import PENDULUM_LQR_PARAMS, PENDULUM_PARAMS
from core.control_utils import dlqr
from core.state_space import StateSpace
from modules.physics_engine import pendulum_dynamics, rk4_fixed_step

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
def _linear_pendulum_matrices(M, m, l, b, g):
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 2] = -m * g / M
    A[2, 3] = 1.0
    A[3, 2] = (M + m) * g / (M * l)
    A[3, 3] = -(M + m) * b / (M * m * l * l)

    B = np.zeros((4, 1))
    B[1, 0] = 1.0 / M
    B[3, 0] = -1.0 / (M * l)

    return A, B


@njit(cache=True)
def _pendulum_param_step(x, force, M, b, g):
    x_dot = x[1]
    theta = x[2]
    omega = x[3]

    m_est = np.exp(x[4])
    l_est = np.exp(x[5])

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m_est * (1.0 - cos_t * cos_t)

    theta_ddot = (
        (M + m_est) * g * sin_t
        - cos_t * (force + m_est * l_est * omega * omega * sin_t)
        - (M + m_est) * b * omega / (m_est * l_est)
    ) / (l_est * denom)

    x_ddot = (
        force + m_est * l_est * omega * omega * sin_t - m_est * g * sin_t * cos_t
    ) / denom

    return x_dot, x_ddot, omega, theta_ddot


@njit(cache=True)
def _pendulum_ukf_step(theta, omega, g, l, b, m, dt):
    theta_next = theta + omega * dt
    omega_next = omega + (-(g / l) * np.sin(theta) - (b / (m * l * l)) * omega) * dt
    return theta_next, omega_next


@njit(cache=True)
def _param_est_wrapper(x, force, M, b, g):
    x_flat = x.ravel()
    dx0, dx1, dx2, dx3 = _pendulum_param_step(x_flat, force, M, b, g)

    res = np.zeros(6)
    res[0] = dx0
    res[1] = dx1
    res[2] = dx2
    res[3] = dx3

    return res.reshape(-1, 1)


@njit(cache=True)
def _param_est_batch(x, force, M, b, g):
    n_batch = x.shape[1]
    res = np.zeros((6, n_batch))

    for i in range(n_batch):
        x_col = x[:, i]
        f = force[i]

        dx0, dx1, dx2, dx3 = _pendulum_param_step(x_col, f, M, b, g)

        res[0, i] = dx0
        res[1, i] = dx1
        res[2, i] = dx2
        res[3, i] = dx3

    return res


class InvertedPendulum:
    """
    Represents the linearized dynamics of an inverted pendulum on a cart about the
    upright equilibrium position.

    This class provides methods to generate linear state-space models, augmented models
    for disturbance estimation, and non-linear dynamics for advanced observers like
    EKF/UKF and controllers like LQR/MPC.

    Attributes:
        params (dict): System parameters (Mass, length, gravity, damping, etc.).
        A (np.ndarray): The linearized system state matrix.
        B (np.ndarray): The linearized system input matrix.
    """

    name = "Pendulum"

    def __init__(self, params=None, **kwargs):
        """
        Initializes the pendulum system with default or custom parameters.

        Args:
            params (dict, optional): A dictionary of physical parameters.
                                     Defaults to PENDULUM_PARAMS.
            **kwargs: Arbitrary keyword arguments to override specific parameters.
        """
        if params is None:
            params = PENDULUM_PARAMS.copy()

        if kwargs:
            params.update(kwargs)

        self.params = params
        self.A, self.B = self._linear_matrices()

    def _linear_matrices(self):
        """
        Computes the linearized State-Space matrices (A, B) around the upright vertical.

        The linearization assumes small angles (theta approx 0).

        The State Matrix A captures:
            - dx/dt = v
            - dtheta/dt = omega
            - Linearized coupling between cart acceleration and pendulum angle (-mg/M).
            - The gravitational instability term ((M+m)g / Ml).
            - Damping effects.

        The Input Matrix B captures:
            - Effect of force on the cart acceleration (1/M).
            - Reaction force effect on the pendulum angular acceleration (-1/Ml).

        Returns:
            tuple: (A, B) numpy arrays.
        """
        p = self.params
        return _linear_pendulum_matrices(p["M"], p["m"], p["l"], p["b"], p["g"])

    def get_parameter_estimation_func(self):
        """
        Generates the system dynamics function f(x, u) tailored for the
        Extended Kalman Filter (EKF) to perform joint state and parameter estimation.

        This specific implementation estimates the Pole Mass (m) and Pole Length (l)
        alongside the physical states.

        The Augmented State Vector (6x1) is defined as:
            x = [Position (x), Velocity (v), Angle (theta), Ang. Velocity (omega), log(m), log(l)]

        Note: Parameters are stored in logarithmic form to ensure positivity
        when exponentiated during the dynamics calculation.

        Returns:
            function: A callable f(x, u) that returns the derivatives of the 6-element state vector.
        """
        M = self.params["M"]
        b = self.params["b"]
        g = self.params["g"]

        def pendulum_dynamics(x, u):
            if x.ndim == 2 and x.shape[1] > 1:
                u_arr = np.atleast_1d(u).ravel()
                if u_arr.size == 1:
                    u_arr = np.full(x.shape[1], u_arr[0])
                elif u_arr.size != x.shape[1]:
                    raise ValueError(
                        f"Input u shape {u.shape} does not match state batch size {x.shape[1]}"
                    )

                return _param_est_batch(x, u_arr, M, b, g)

            if hasattr(u, "ndim") and u.ndim == 2:
                force = u[0, 0]
            elif hasattr(u, "__len__"):
                force = u[0]
            else:
                force = u
            return _param_est_wrapper(x, force, M, b, g)

        return pendulum_dynamics

    def get_state_space(self):
        """
        Returns the standard Linear Time-Invariant (LTI) StateSpace object.

        States:  [x, x_dot, theta, omega]
        Outputs: Full state output (C = Identity), usually measuring all 4 states for simulation,
                 though physical measurement might vary.

        Returns:
            StateSpace: The linear system model.
        """
        C = np.eye(4)
        D = np.zeros((4, 1))
        return StateSpace(self.A, self.B, C, D)

    def get_augmented_state_space(self):
        """
        Constructs an Augmented State-Space model specifically for Disturbance Estimation.

        This adds a 5th state representing a 'Disturbance Bias' (e.g., constant external force/torque).
        The Kalman Filter uses this model to estimate steady-state errors.

        Augmented State Vector:
            [x, x_dot, theta, omega, Disturbance_Bias]

        The B matrix is augmented assuming the control input 'u' is known and does not
        affect the disturbance state directly.

        Returns:
            StateSpace: The augmented linear system model.
        """
        A_std, B_std = self._linear_matrices()
        B_dist = np.zeros((4, 1))
        B_dist[3, 0] = (self.params["M"] + self.params["m"]) / (
            self.params["M"] * self.params["m"] * self.params["l"] ** 2
        )

        A_aug = np.vstack((np.hstack((A_std, B_dist)), np.zeros((1, 5))))
        B_aug = np.vstack((B_std, [[0]]))

        C_aug = np.zeros((4, 5))
        C_aug[:, :4] = np.eye(4)

        D_aug = np.zeros((4, 1))
        return StateSpace(A_aug, B_aug, C_aug, D_aug)

    def dlqr_gain(self, dt=0.01):
        """
        Computes the discrete-time LQR feedback gain matrix K.

        The weighting matrices Q and R are configured to prioritize balancing the pole
        (theta) and preventing drift (position), while penalizing control effort.

        Args:
            dt (float, optional): The discretization time step. Defaults to 0.01.

        Returns:
            np.ndarray: The optimal gain matrix K.
        """
        Q = np.diag(PENDULUM_LQR_PARAMS["Q_diag"])
        R = np.array([[PENDULUM_LQR_PARAMS["R_val"]]])

        A_d = np.eye(4) + self.A * dt
        B_d = self.B * dt

        return dlqr(A_d, B_d, Q, R)

    def get_open_loop_tf(self, K):
        """
        Returns the Open-Loop Transfer Function L(s) for the LQR controller.

        Args:
            K (np.ndarray): The LQR Gain matrix (1x4).

        Returns:
            LQRLoopTransferFunction: Object representing L(s) = K(sI-A)^-1 B.
        """
        return LQRLoopTransferFunction(self.A, self.B, K)

    def measurement(self, x):
        """
        The measurement function h(x) for the Extended Kalman Filter.

        Simulates a real-world scenario where we only measure positions directly,
        not velocities.

        Args:
            x (np.ndarray): Full state vector [x, v, theta, omega].

        Returns:
            np.ndarray: Measured vector [x, theta].
        """
        return np.array([x[0], x[2]])

    def measurement_jacobian(self, x):
        """
        Computes the Jacobian matrix H of the measurement function h(x).
        Since h(x) is linear mapping states 0 and 2, H is constant.

        Returns:
            np.ndarray: Jacobian matrix (2x4).
        """
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        return H

    def dynamics(self, x, u, dt):
        """
        Computes the next state using the non-linear physics engine and RK4 integration.
        This serves as the 'Discrete-time nonlinear dynamics' step for estimators.

        Args:
            x (np.ndarray): Current state.
            u (float): Control input.
            dt (float): Time step.

        Returns:
            np.ndarray: Next state.
        """
        return rk4_fixed_step(
            pendulum_dynamics,
            x,
            u,
            dt,
            self.params,
            disturbance=0.0,
        )

    def dynamics_continuous(self, x, u):
        """
        Wrapper for the continuous-time nonlinear pendulum dynamics.
        Used by continuous-time solvers or linearization routines.

        Args:
            x (np.ndarray): State vector.
            u (float): Control input.

        Returns:
            np.ndarray: State derivatives dx/dt.
        """
        return pendulum_dynamics(0, x, u, self.params, 0.0)

    def get_nonlinear_dynamics(self):
        """
        Returns the dynamics and measurement functions tailored for the
        Unscented Kalman Filter (UKF).

        This simplified model focuses on the rotational dynamics for the free-swing
        demo.

        State:
            [Theta, Theta_dot]
        Measurement:
            [Theta]

        Returns:
            tuple: (f_dynamics, h_measurement) functions.
        """
        g = self.params["g"]
        l = self.params["l"]
        b = self.params["b"]
        m = self.params["m"]

        def pendulum_ukf_dynamics(x, u, dt):
            theta, omega = x
            th, om = _pendulum_ukf_step(theta, omega, g, l, b, m, dt)
            return np.array([th, om])

        def measurement_model(x):
            return np.array([x[0]])

        return pendulum_ukf_dynamics, measurement_model

    def get_mpc_model(self, dt):
        """
        Returns the discrete-time non-linear dynamics function f(x, u, dt) specifically
        designed for Model Predictive Control (MPC) solvers like iLQR.

        To ensure stability of gradients in the iLQR solver, this model uses
        simple Euler integration for the prediction step.

        Args:
            dt (float): The prediction time step.

        Returns:
            function: A callable f(x, u, dt) -> x_next.
        """
        M = self.params["M"]
        b = self.params["b"]
        g = self.params["g"]
        m = self.params["m"]
        l = self.params["l"]
        log_m, log_l = np.log(m), np.log(l)

        @njit(cache=True)
        def mpc_dynamics_fast(x, u, dt):
            u_arr = np.atleast_1d(u)
            force = u_arr.flat[0]
            x_flat = x.ravel()

            x_aug = np.zeros(6)
            x_aug[:4] = x_flat
            x_aug[4] = log_m
            x_aug[5] = log_l

            _, x_ddot, _, theta_ddot = _pendulum_param_step(x_aug, force, M, b, g)

            x_next = np.empty_like(x)
            x_next_flat = x_next.ravel()

            x_next_flat[0] = x_flat[0] + x_flat[1] * dt
            x_next_flat[1] = x_flat[1] + x_ddot * dt
            x_next_flat[2] = x_flat[2] + x_flat[3] * dt
            x_next_flat[3] = x_flat[3] + theta_ddot * dt

            return x_next

        @njit(cache=True)
        def mpc_dynamics_complex(x, u, dt):
            u_arr = np.atleast_1d(u)
            force = u_arr.flat[0]
            x_flat = x.ravel()

            x_aug = np.zeros(6, dtype=np.complex128)
            x_aug[:4] = x_flat
            x_aug[4] = log_m
            x_aug[5] = log_l

            _, x_ddot, _, theta_ddot = _pendulum_param_step(x_aug, force, M, b, g)

            x_next = np.empty_like(x)
            x_next_flat = x_next.ravel()

            x_next_flat[0] = x_flat[0] + x_flat[1] * dt
            x_next_flat[1] = x_flat[1] + x_ddot * dt
            x_next_flat[2] = x_flat[2] + x_flat[3] * dt
            x_next_flat[3] = x_flat[3] + theta_ddot * dt

            return x_next

        def mpc_dynamics_dispatcher(x, u, dt):
            if np.iscomplexobj(x) or np.iscomplexobj(u):
                return mpc_dynamics_complex(x, u, dt)
            return mpc_dynamics_fast(x, u, dt)

        return mpc_dynamics_dispatcher


class LQRLoopTransferFunction:
    """
    A virtual Transfer Function representing the Open Loop Gain L(s) of an LQR system.

    For a system dx/dt = Ax + Bu with feedback u = -Kx, the loop gain at the
    plant input is:
        L(s) = K * (sI - A)^-1 * B

    This class mimics the interface of core.transfer_function.TransferFunction
    so it can be used with analysis tools.
    """

    def __init__(self, A, B, K):
        self.A = A
        self.B = B
        self.K = K
        self.n_states = A.shape[0]
        self.I = np.eye(self.n_states)

    def evaluate(self, s):
        """
        Evaluates L(s) = K(sI - A)^-1 B at a complex frequency s.
        Uses linear solver (sI - A)x = B to avoid explicit inversion.
        """
        try:
            term = np.linalg.solve(s * self.I - self.A, self.B)
            val = (self.K @ term)[0, 0]
            return val

        except np.linalg.LinAlgError:
            return np.inf
