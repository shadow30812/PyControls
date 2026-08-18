from _collections_abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.base import BaseController

try:
    from numba import njit

    NUMBA_AVAILABLE: bool = True
except ImportError:
    NUMBA_AVAILABLE: bool = False

    def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


@njit(cache=True, fastmath=True)
def _sda_core(
    A: NDArray[np.float64],
    G: NDArray[np.float64],
    H: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Core numerical kernel for the Structure-Preserving Doubling Algorithm (SDA).

    Solves the Discrete Algebraic Riccati Equation (DARE) via quadratic
    doubling iterations on the symplectic matrix pencil.

    Args:
        A (NDArray[np.float64]): State transition matrix (n x n).
        G (NDArray[np.float64]): Input coupling matrix G = B * R^-1 * B^T (n x n).
        H (NDArray[np.float64]): Initial cost matrix H_0 = Q (n x n).
        tol (float, optional): Infinity-norm convergence threshold. Defaults to 1e-10.
        max_iter (int, optional): Maximum doubling iterations. Defaults to 50.

    Returns:
        NDArray[np.float64]: Positive semi-definite solution matrix P (n x n).
    """
    n: int = A.shape[0]
    I_n: NDArray[np.float64] = np.eye(n, dtype=np.float64)

    A_k: NDArray[np.float64] = A.copy()
    G_k: NDArray[np.float64] = G.copy()
    H_k: NDArray[np.float64] = H.copy()

    for _ in range(max_iter):
        W: NDArray[np.float64] = I_n + G_k @ H_k

        # Solve linear systems W * X = A_k and W * Y = G_k
        inv_W_A: NDArray[np.float64] = np.linalg.solve(W, A_k)
        inv_W_G: NDArray[np.float64] = np.linalg.solve(W, G_k)

        A_next: NDArray[np.float64] = A_k @ inv_W_A
        G_next: NDArray[np.float64] = G_k + A_k @ inv_W_G @ A_k.T
        H_next: NDArray[np.float64] = H_k + A_k.T @ H_k @ inv_W_A

        # Enforce numerical symmetry
        H_next = 0.5 * (H_next + H_next.T)
        G_next = 0.5 * (G_next + G_next.T)

        # Infinity-norm convergence check: max|H_next - H_k|
        max_diff: float = 0.0
        for r in range(n):
            for c in range(n):
                d: float = abs(H_next[r, c] - H_k[r, c])
                max_diff = max(max_diff, d)

        if max_diff < tol:
            return H_next

        A_k = A_next
        G_k = G_next
        H_k = H_next

    return H_k


def solve_discrete_riccati(
    A: Union[ArrayLike, NDArray[np.float64]],
    B: Union[ArrayLike, NDArray[np.float64]],
    Q: Union[ArrayLike, NDArray[np.float64]],
    R: Union[ArrayLike, NDArray[np.float64]],
    tol: float = 1e-10,
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """
    Solves the Discrete-time Algebraic Riccati Equation (DARE) using the
    Structure-Preserving Doubling Algorithm (SDA).

    The algebraic equation solved is:
        P = A^T P A - (A^T P B) (R + B^T P B)^-1 (B^T P A) + Q

    Unlike classical fixed-point Picard iteration which exhibits linear convergence,
    SDA achieves quadratic convergence by doubling the effective horizon at every step
    (2^k steps at iteration k), resolving the solution to machine precision in 6–15 iterations.

    Args:
        A (array-like): Continuous/discrete state transition matrix (n x n).
        B (array-like): Input matrix (n x m).
        Q (array-like): State cost weighting matrix (positive semi-definite, n x n).
        R (array-like): Input cost weighting matrix (positive definite, m x m or scalar).
        tol (float, optional): Maximum absolute difference tolerance for convergence.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum doubling iterations (50 iterations corresponds
            to an effective horizon of 2^50 steps). Defaults to 50.

    Returns:
        NDArray[np.float64]: The unique stabilizing positive semi-definite solution matrix P (n x n).

    Raises:
        ValueError: If input dimensions are incompatible.
    """
    A_mat: NDArray[np.float64] = np.asarray(A, dtype=np.float64)
    B_mat: NDArray[np.float64] = np.asarray(B, dtype=np.float64)
    Q_mat: NDArray[np.float64] = np.asarray(Q, dtype=np.float64)
    R_mat: NDArray[np.float64] = np.asarray(R, dtype=np.float64)

    # Compute G_0 = B * R^-1 * B^T
    if R_mat.ndim == 0 or R_mat.shape == (1, 1) or R_mat.size == 1:
        G_mat: NDArray[np.float64] = (B_mat @ B_mat.T) / float(R_mat.squeeze())
    else:
        G_mat = B_mat @ np.linalg.solve(R_mat, B_mat.T)

    return _sda_core(A_mat, G_mat, Q_mat, tol=tol, max_iter=max_iter)


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


class PIDController(BaseController):
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

    def compute(
        self,
        state: NDArray[np.float64],
        reference: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """
        Unified controller interface. Extracts scalar measurement/setpoint
        from state/reference vectors and returns control as 1-D array.
        """
        measurement = float(state.flat[0])
        setpoint = float(reference.flat[0])
        u = self.update(measurement, setpoint, dt)
        return np.array([u])

    def save_state(self) -> Dict[str, Any]:
        return {
            "class": "PIDController",
            "integral_error": self.integral_error,
            "prev_value": self.prev_value,
            "prev_derivative": self.prev_derivative,
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        self.integral_error = state_dict.get("integral_error", 0.0)
        self.prev_value = state_dict.get("prev_value", 0.0)
        self.prev_derivative = state_dict.get("prev_derivative", 0.0)
