import numpy as np


def solve_discrete_riccati(A, B, Q, R, tol=1e-8, max_iter=1000):
    """
    Solves the discrete-time Algebraic Riccati Equation (DARE) via iterative convergence.

    The equation solved is:
        P = A^T P A - (A^T P B) (R + B^T P B)^-1 (B^T P A) + Q

    This method iterates until the maximum difference between P_next and P
    is less than the specified tolerance.

    Args:
        A (np.ndarray): The state transition matrix.
        B (np.ndarray): The input matrix.
        Q (np.ndarray): The state cost matrix (must be positive semi-definite).
        R (np.ndarray): The input cost matrix (must be positive definite).
        tol (float, optional): The convergence tolerance. Defaults to 1e-8.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

    Returns:
        np.ndarray: The solution matrix P.
    """
    P = Q.copy()

    for _ in range(max_iter):
        P_next = (
            A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        )

        if np.max(np.abs(P_next - P)) < tol:
            break

        P = P_next

    return P


def dlqr(A, B, Q, R):
    """
    Computes the optimal Linear Quadratic Regulator (LQR) gain for a discrete-time system.

    This function first solves the Discrete Algebraic Riccati Equation (DARE) to find
    the cost-to-go matrix P, and then computes the optimal feedback gain K such that
    u[k] = -K * x[k].

    Args:
        A (np.ndarray): The state transition matrix.
        B (np.ndarray): The input matrix.
        Q (np.ndarray): The state weighting matrix.
        R (np.ndarray): The input weighting matrix.

    Returns:
        np.ndarray: The optimal gain matrix K.
    """
    P = solve_discrete_riccati(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


class PIDController:
    """
    A robust PID controller implementation with support for Derivative-on-Measurement
    and Low-Pass Filtering for the derivative term.

    Features:
    - Derivative-on-Measurement: Calculates the D-term using the change in measurement
      rather than error to prevent "derivative kick" on setpoint changes.
    - Low-Pass Filter: Smooths the derivative term to reduce noise amplification.
    - Anti-Windup: Handled via output saturation limits.

    Args:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        derivative_on_measurement (bool, optional): If True, computes derivative on
            measurement changes (dy/dt). If False, uses error changes (de/dt).
            Defaults to True.
        output_limits (tuple, optional): A tuple (min, max) for output saturation.
            Use None for no limit (e.g., (None, 10.0)). Defaults to (None, None).
        tau (float, optional): Time constant for the derivative low-pass filter.
            Controls the smoothing factor alpha = dt / (tau + dt).
            Typical values range from 0.01 to 0.1s. Defaults to 0.02.
    """

    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        derivative_on_measurement=True,
        output_limits=(None, None),
        tau=0.02,
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.derivative_on_measurement = derivative_on_measurement
        self.min_out, self.max_out = output_limits
        self.tau = tau

        self.reset()

    def reset(self):
        """
        Resets the controller's internal state.

        Clears the accumulated integral error and resets the previous values used
        for derivative calculations to zero.
        """
        self.integral_error = 0.0
        self.prev_value = 0.0
        self.prev_derivative = 0.0

    def update(self, measurement, setpoint, dt):
        """
        Calculates the control output based on the current measurement and setpoint.

        This method performs the following steps:
        1. Updates the integral term (accumulating error * dt).
        2. Calculates the raw derivative based on either the measurement or error slope.
        3. Applies a low-pass filter to the raw derivative using the time constant `tau`.
        4. Computes the raw PID output.
        5. Clamps the output to the configured limits (Anti-Windup).

        Args:
            measurement (float): The current system process variable.
            setpoint (float): The desired target value.
            dt (float): The time step duration in seconds.

        Returns:
            float: The computed control output u.
        """
        if dt <= 0.0:
            return 0.0

        error = setpoint - measurement
        self.integral_error += error * dt

        if self.derivative_on_measurement:
            raw_derivative = (measurement - self.prev_value) / dt
            self.prev_value = measurement
            sign = -1.0
        else:
            raw_derivative = (error - self.prev_value) / dt
            self.prev_value = error
            sign = 1.0

        alpha = dt / (self.tau + dt)
        derivative = alpha * raw_derivative + (1.0 - alpha) * self.prev_derivative
        self.prev_derivative = derivative

        u = (
            (self.Kp * error)
            + (self.Ki * self.integral_error)
            + (sign * self.Kd * derivative)
        )

        if self.min_out is not None:
            u = max(self.min_out, u)
        if self.max_out is not None:
            u = min(self.max_out, u)

        return u
