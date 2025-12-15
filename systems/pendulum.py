import numpy as np

from config import PENDULUM_LQR_PARAMS, PENDULUM_PARAMS
from core.control_utils import dlqr
from core.state_space import StateSpace
from modules.physics_engine import pendulum_dynamics, rk4_fixed_step


class InvertedPendulum:
    """
    Linearized inverted pendulum about upright equilibrium.
    """

    name = "Pendulum"

    def __init__(self, params=None, **kwargs):
        if params is None:
            params = PENDULUM_PARAMS.copy()

        # Update params with any keyword arguments provided
        if kwargs:
            params.update(kwargs)

        self.params = params
        self.A, self.B = self._linear_matrices()

    def _linear_matrices(self):
        M = self.params["M"]
        m = self.params["m"]
        l = self.params["l"]
        b = self.params["b"]
        g = self.params["g"]

        A = np.zeros((4, 4))
        A[0, 1] = 1.0  # dx/dt = v
        A[1, 2] = -m * g / M  # Linearized coupling (small angle approx)
        A[2, 3] = 1.0  # dtheta/dt = omega
        A[3, 2] = (M + m) * g / (M * l)  # Instability term (gravity)
        A[3, 3] = -(M + m) * b / (M * m * l**2)

        B = np.zeros((4, 1))
        B[1, 0] = 1.0 / M  # Force on Cart
        B[3, 0] = -1.0 / (M * l)  # Reaction force on Pendulum

        return A, B

    def get_parameter_estimation_func(self):
        """
        Returns the system dynamics function f(x, u) for parameter estimation.
        Designed for the Extended Kalman Filter (EKF).

        Estimates: Pole Mass (m) and Pole Length (l)

        State Vector (6x1):
            [Position (x), Velocity (v), Angle (theta), Ang. Velocity (omega), log(m), log(l)]
        """
        # We hold M, b, g constant from the initial config
        M_const = self.params["M"]
        b_const = self.params["b"]
        g_const = self.params["g"]

        def pendulum_dynamics_6_state(x, u):
            # Unpack States
            # x_pos = x[0]  (Unused in acceleration calc, but tracked)
            x_vel = x[1]
            theta = x[2]
            theta_dot = x[3]
            m_est = np.exp(x[4])
            l_est = np.exp(x[5])

            # Unpack Input
            if hasattr(u, "ndim") and u.ndim == 2:
                force = u[0, 0]
            elif hasattr(u, "__len__"):
                force = u[0]
            else:
                force = u

            # Equations of Motion (Non-linear)
            # Derived from physics_engine.py logic but adapted for variable m/l
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            denom = M_const + m_est * (1 - cos_t**2)

            # 1. Angular Acceleration (theta_ddot)
            term_grav = (M_const + m_est) * g_const * sin_t
            term_coupled = -cos_t * (force + m_est * l_est * theta_dot**2 * sin_t)
            term_fric = -(M_const + m_est) * b_const * theta_dot / (m_est * l_est)

            theta_ddot = (term_grav + term_coupled + term_fric) / (l_est * denom)

            # 2. Linear Acceleration (x_ddot)
            term3 = force + m_est * l_est * theta_dot**2 * sin_t
            term4 = -m_est * g_const * sin_t * cos_t
            x_ddot = (term3 + term4) / denom

            # Derivatives of parameters are zero (constants)
            dm_dt = np.zeros_like(x_vel)
            dl_dt = np.zeros_like(x_vel)

            # [dx/dt, dv/dt, dtheta/dt, domega/dt, dm/dt, dl/dt]
            result = np.array([x_vel, x_ddot, theta_dot, theta_ddot, dm_dt, dl_dt])

            if result.ndim == 1:
                return result.reshape(-1, 1)
            return result

        return pendulum_dynamics_6_state

    def get_state_space(self):
        """
        Returns the linear StateSpace model (A, B, C, D).
        Outputs: [Angle (theta), Angular Velocity (theta_dot)]
        """
        C = np.eye(4)
        D = np.zeros((4, 1))
        return StateSpace(self.A, self.B, C, D)

    def get_augmented_state_space(self):
        """
        Constructs an Augmented State-Space model for Disturbance Estimation.
        States: [Theta, Theta_dot, Disturbance_Bias]
        """
        A_std, B_std = self._linear_matrices()
        B_dist_effect = np.zeros((4, 1))
        B_dist_effect[3, 0] = (self.params["M"] + self.params["m"]) / (
            self.params["M"] * self.params["m"] * self.params["l"] ** 2
        )

        top = np.hstack((A_std, B_dist_effect))
        bottom = np.zeros((1, 5))
        A_aug = np.vstack((top, bottom))

        # We assume the control input 'u' is known
        B_aug = np.vstack((B_std, [[0]]))

        # Measurement Matrix C_aug (5 states)
        # We output the 4 physical states
        C_aug = np.zeros((4, 5))
        C_aug[:, :4] = np.eye(4)

        D_aug = np.zeros((4, 1))

        return StateSpace(A_aug, B_aug, C_aug, D_aug)

    def dlqr_gain(self, dt=0.01):
        # Q penalizes [x, x_dot, theta, theta_dot]
        # Heavier penalty on Theta (balancing) and Position (drift)
        Q = np.diag(PENDULUM_LQR_PARAMS["Q_diag"])
        R = np.array([[PENDULUM_LQR_PARAMS["R_val"]]])

        A_d = np.eye(4) + self.A * dt
        B_d = self.B * dt

        return dlqr(A_d, B_d, Q, R)

    def measurement(self, x):
        """
        Measurement model for EKF.
        Usually we measure positions [x, theta].
        """
        # x is [x_pos, x_vel, theta, theta_vel]
        return np.array([x[0], x[2]])

    def measurement_jacobian(self, x):
        """
        Jacobian of measurement function H.
        """
        H = np.zeros((2, 4))
        H[0, 0] = 1.0  # Measure x
        H[1, 2] = 1.0  # Measure theta
        return H

    def dynamics(self, x, u, dt):
        """
        Discrete-time nonlinear dynamics wrapper.
        Uses existing pendulum_dynamics + RK4.
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
        Continuous-time nonlinear pendulum dynamics.
        """
        return pendulum_dynamics(0, x, u, self.params, 0.0)

    def get_nonlinear_dynamics(self):
        """
        Returns (f_dynamics, h_measurement) for Unscented Kalman Filter (UKF).

        State: [Theta, Theta_dot] (Simplified 2-state model for UKF demo)
        Input: u (Force on cart - set to 0 for free swing demo)
        Measurement: [Theta]
        """
        g = self.params["g"]
        l = self.params["l"]
        b = self.params["b"]
        m = self.params["m"]

        def pendulum_ukf_dynamics(x, u, dt):
            """
            Simple non-linear pendulum equation:
            theta_ddot = -(g/l)sin(theta) - (b/ml^2)theta_dot
            """
            theta = x[0]
            omega = x[1]

            # Simple Euler integration for prediction step
            theta_next = theta + omega * dt

            # Dynamics
            alpha = -(g / l) * np.sin(theta) - (b / (m * l**2)) * omega
            omega_next = omega + alpha * dt

            return np.array([theta_next, omega_next])

        def measurement_model(x):
            # We measure the angle theta
            return np.array([x[0]])

        return pendulum_ukf_dynamics, measurement_model

    def get_mpc_model(self, dt):
        """
        Returns the dynamics function f(x, u, dt) for Nonlinear MPC (iLQR).
        """
        # Capture params in closure
        params = self.params.copy()

        def mpc_dynamics(x, u, dt):
            # iLQR expects x, u arrays.
            # Reuse the rk4_fixed_step logic or simple integration.
            # Using simple Euler for the PREDICTION model makes iLQR
            # derivatives (gradients) much more stable/smooth.

            theta = x[2]
            theta_dot = x[3]
            x_dot = x[1]

            force = u[0]

            M = params["M"]
            m = params["m"]
            l = params["l"]
            b = params["b"]
            g = params["g"]

            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            denom = M + m * (1 - cos_t**2)

            # Acceleration calc (Same as physics engine)
            term_grav = (M + m) * g * sin_t
            term_coupled = -cos_t * (force + m * l * theta_dot**2 * sin_t)
            term_fric = -(M + m) * b * theta_dot / (m * l)
            theta_ddot = (term_grav + term_coupled + term_fric) / (l * denom)

            term3 = force + m * l * theta_dot**2 * sin_t
            term4 = -m * g * sin_t * cos_t
            x_ddot = (term3 + term4) / denom

            # Euler Integration
            x_next = x.copy()
            x_next[0] += x_dot * dt
            x_next[1] += x_ddot * dt
            x_next[2] += theta_dot * dt
            x_next[3] += theta_ddot * dt

            return x_next

        return mpc_dynamics
