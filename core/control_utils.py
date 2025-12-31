from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


def solve_discrete_riccati(
    A: Union[ArrayLike, NDArray[np.float64]],
    B: Union[ArrayLike, NDArray[np.float64]],
    Q: Union[ArrayLike, NDArray[np.float64]],
    R: Union[ArrayLike, NDArray[np.float64]],
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> NDArray[np.float64]:
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
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P: NDArray[np.float64] = Q.copy()

    for _ in range(max_iter):
        BT_P: NDArray[np.float64] = B.T @ P
        S: NDArray[np.float64] = R + BT_P @ B

        K: NDArray[np.floating] = np.linalg.solve(S, BT_P @ A)
        P_next: NDArray[np.float64] = A.T @ P @ A - A.T @ P @ B @ K + Q

        P_next = 0.5 * (P_next + P_next.T)

        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break

        P = P_next

    return P


def dlqr(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> NDArray[np.floating]:
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
    P: NDArray[np.float64] = solve_discrete_riccati(A, B, Q, R)
    BT_P: NDArray[np.float64] = B.T @ P
    return np.linalg.solve(R + BT_P @ B, BT_P @ A)


class Check:
    def _matrix_rank(
        self,
        M: Union[ArrayLike, NDArray[np.float64]],
        atol: float = 1e-15,
        rtol: Optional[float] = None,
    ) -> int:
        """
        Computes the rank of a matrix using SVD with hybrid tolerance.

        Rank is the number of singular values greater than the tolerance threshold.
        Threshold = max(atol, rtol * sigma_max)
        """
        M = np.asarray(M, dtype=float)
        if M.size == 0:
            return 0

        s: NDArray[np.floating] = np.linalg.svd(M, compute_uv=False)

        if atol is None:
            atol = max(M.shape)

        if rtol is None:
            rtol = max(M.shape) * np.finfo(M.dtype).eps

        threshold: float = max(atol, rtol * s[0])
        return int(np.sum(s > threshold))

    def controllability_matrix(
        self,
        A: Union[NDArray[np.float64], ArrayLike],
        B: Union[NDArray[np.float64], ArrayLike],
    ) -> NDArray[np.float64]:
        """
        Constructs the controllability matrix [B, AB, A^2B, ..., A^{n-1}B].
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        n: int = A.shape[0]
        mats: List[NDArray[np.float64]] = [B]

        Ak: NDArray[np.float64] = np.eye(n)
        for _ in range(1, n):
            Ak = Ak @ A
            mats.append(Ak @ B)

        return np.concatenate(mats, axis=1)

    def observability_matrix(
        self,
        A: Union[NDArray[np.float64], ArrayLike],
        C: Union[NDArray[np.float64], ArrayLike],
    ) -> NDArray[np.float64]:
        """
        Constructs the observability matrix [C; CA; CA^2; ...; CA^{n-1}].
        """
        A = np.asarray(A, dtype=float)
        C = np.asarray(C, dtype=float)

        n: int = A.shape[0]
        mats: List[NDArray[np.float64]] = [C]

        Ak: NDArray[np.float64] = np.eye(n)
        for _ in range(1, n):
            Ak = Ak @ A
            mats.append(C @ Ak)

        return np.concatenate(mats, axis=0)

    def is_controllable(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        atol: float = 1e-15,
        rtol: Optional[float] = None,
    ) -> bool:
        """
        Checks controllability via rank of controllability matrix.
        """
        Ctrb: NDArray[np.float64] = self.controllability_matrix(A, B)
        n: int = A.shape[0]
        rank: int = self._matrix_rank(Ctrb, atol=atol, rtol=rtol)
        return rank == n

    def is_observable(
        self,
        A: NDArray[np.float64],
        C: NDArray[np.float64],
        atol: float = 1e-15,
        rtol: Optional[float] = None,
    ) -> bool:
        """
        Checks observability via rank of observability matrix.
        """
        Obsv: NDArray[np.float64] = self.observability_matrix(A, C)
        n: int = A.shape[0]
        rank: int = self._matrix_rank(Obsv, atol=atol, rtol=rtol)
        return rank == n


class PIDController:
    """
    A robust PID controller implementation with support for Derivative-on-Measurement
    and Low-Pass Filtering for the derivative term, and anti-Windup for the integral.

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
        Kp: Union[int, float],
        Ki: Union[int, float],
        Kd: Union[int, float],
        derivative_on_measurement: bool = True,
        output_limits: Tuple[Optional[float], Optional[float]] = (None, None),
        integral_limits: Tuple[Optional[float], Optional[float]] = (None, None),
        tau: Union[int, float] = 0.02,
    ) -> None:
        self.Kp: float = float(Kp)
        self.Ki: float = float(Ki)
        self.Kd: float = float(Kd)

        self.derivative_on_measurement: bool = derivative_on_measurement
        self.min_out, self.max_out = output_limits
        self.min_int, self.max_int = integral_limits

        self.tau: float = float(tau)
        self.reset()

    def reset(self) -> None:
        """
        Resets the controller's internal state.

        Clears the accumulated integral error and resets the previous values used
        for derivative calculations to zero.
        """
        self.integral_error: float = 0.0
        self.prev_value: float = 0.0
        self.prev_derivative: float = 0.0

    def update(self, measurement: float, setpoint: float, dt: float) -> float:
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

        error: float = setpoint - measurement
        self.integral_error += error * dt

        new_integral: float = self.integral_error
        if self.min_int is not None:
            new_integral = max(self.min_int, new_integral)
        if self.max_int is not None:
            new_integral = min(self.max_int, new_integral)
        self.integral_error = new_integral

        if self.derivative_on_measurement:
            raw_derivative: float = (measurement - self.prev_value) / dt
            self.prev_value = measurement
            sign: float = -1.0
        else:
            raw_derivative = (error - self.prev_value) / dt
            self.prev_value = error
            sign = 1.0

        alpha: float = dt / (self.tau + dt)
        derivative: float = (
            alpha * raw_derivative + (1.0 - alpha) * self.prev_derivative
        )
        self.prev_derivative = derivative

        u: float = (
            self.Kp * error
            + self.Ki * self.integral_error
            + sign * self.Kd * derivative
        )

        if self.min_out is not None:
            u = max(self.min_out, u)
        if self.max_out is not None:
            u = min(self.max_out, u)

        return u
