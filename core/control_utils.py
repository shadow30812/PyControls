import numpy as np


def solve_discrete_riccati(A, B, Q, R, tol=1e-8, max_iter=1000):
    """
    Solves the discrete-time algebraic Riccati equation via iteration.
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
    P = solve_discrete_riccati(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


class PIDController:
    """
    A robust PID controller with Derivative-on-Measurement and Low-Pass Filtering.

    Args:
        tau (float): Time constant for the derivative low-pass filter.
                     Typical value: 0.01 to 0.1. Set to 0.0 to disable.
    """

    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        derivative_on_measurement=True,
        output_limits=(None, None),
        tau=0.02,  # Default filter time constant (20ms)
    ):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.derivative_on_measurement = derivative_on_measurement
        self.min_out, self.max_out = output_limits
        self.tau = tau

        self.reset()

    def reset(self):
        self.integral_error = 0.0
        self.prev_value = 0.0  # Stores prev_measurement (if mode=True) or prev_error
        self.prev_derivative = 0.0

    def update(self, measurement, setpoint, dt):
        """
        Calculates control output.
        """
        if dt <= 0.0:
            return 0.0

        error = setpoint - measurement
        self.integral_error += error * dt

        # --- Derivative Calculation with Low-Pass Filter ---
        if self.derivative_on_measurement:
            # Derivative of y
            raw_derivative = (measurement - self.prev_value) / dt
            self.prev_value = measurement
            sign = -1.0  # D-term is subtracted
        else:
            # Derivative of e
            raw_derivative = (error - self.prev_value) / dt
            self.prev_value = error
            sign = 1.0  # D-term is added

        # Filter: alpha determines how much of the new raw value we accept
        # alpha = dt / (tau + dt) -> If tau=0, alpha=1 (no filter)
        alpha = dt / (self.tau + dt)
        derivative = alpha * raw_derivative + (1.0 - alpha) * self.prev_derivative
        self.prev_derivative = derivative

        # --- PID Output ---
        u = (
            (self.Kp * error)
            + (self.Ki * self.integral_error)
            + (sign * self.Kd * derivative)
        )

        # Apply Saturation Limits
        if self.min_out is not None:
            u = max(self.min_out, u)
        if self.max_out is not None:
            u = min(self.max_out, u)

        return u
