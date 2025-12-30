from typing import Any, Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helpers.config import SOLVER_PARAMS

try:
    from numba import njit

    NUMBA_AVAILABLE: bool = True
except ImportError:
    NUMBA_AVAILABLE: bool = False

    def njit(*args, **kwargs) -> Callable[..., Any]:
        def decorator(func) -> Any:
            return func

        return decorator


@njit(cache=True, fastmath=True)
def _mat_mul(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Manual matrix multiplication to avoid Numba requiring SciPy/BLAS.
    Performs C = A @ B.
    """
    rows_A: int = A.shape[0]
    cols_A: int = A.shape[1]
    cols_B: int = B.shape[1]
    C: NDArray[np.float64] = np.zeros((rows_A, cols_B), dtype=np.float64)

    for i in range(rows_A):
        for j in range(cols_B):
            acc: float = 0.0
            for k in range(cols_A):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


@njit(cache=True)
def manual_matrix_exp(
    A: NDArray[np.float64], order: int = SOLVER_PARAMS["matrix_exp_order"]
) -> NDArray[np.float64]:
    """
    Computes the matrix exponential e^A using Scaling and Squaring with Taylor Series.
    Formula: e^A = (e^(A/2^s))^(2^s)

    Optimized with Numba JIT compilation. Uses manual matrix multiplication
    to maintain independence from scipy/blas.
    """
    shape: Tuple[int, int] = A.shape
    rows, cols = shape

    if rows == 1:
        return np.array([[np.exp(A[0, 0])]])

    norm_A: float = 0.0
    for i in range(rows):
        row_sum: float = 0.0
        for j in range(cols):
            row_sum += np.abs(A[i, j])
        if row_sum > norm_A:
            norm_A = row_sum

    s: int = 0
    while norm_A > 0.5:
        norm_A *= 0.5
        s += 1

    inv_scale: float = 1.0 / (2.0**s)
    A_scaled: NDArray[np.float64] = A * inv_scale

    E: NDArray[np.float64] = np.eye(rows)
    term: NDArray[np.float64] = np.eye(rows)

    for k in range(1, order + 1):
        term = _mat_mul(term, A_scaled) / k
        E += term

    for _ in range(s):
        E = _mat_mul(E, E)

    return E


class ExactSolver:
    """
    Exact Discrete-Time Solver for Linear Time-Invariant (LTI) Systems.
    Uses the Zero-Order Hold (ZOH) method to discretize continuous matrices.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        dt: float,
    ) -> None:
        self.A: NDArray[np.float64] = np.atleast_2d(A)
        self.B: NDArray[np.float64] = np.atleast_2d(B)
        self.C: NDArray[np.float64] = np.atleast_2d(C)
        self.D: NDArray[np.float64] = np.atleast_2d(D)

        self.x: NDArray[np.float64] = np.zeros((self.A.shape[0], 1))

        n_states: int = self.A.shape[0]
        n_inputs: int = self.B.shape[1]

        top: NDArray[np.float64] = np.hstack((self.A, self.B))
        bottom: NDArray[np.float64] = np.zeros((n_inputs, n_states + n_inputs))
        M: NDArray[np.float64] = np.vstack((top, bottom))

        M_exp: NDArray[np.float64] = manual_matrix_exp(M * dt)
        self.Phi: NDArray[np.float64] = M_exp[:n_states, :n_states]
        self.Gamma: NDArray[np.float64] = M_exp[:n_states, n_states:]

    def step(
        self, u_input: Union[float, ArrayLike]
    ) -> Union[float, NDArray[np.float64]]:
        """
        Advances the simulation by one discrete time step.
        """
        u: NDArray[np.float64] = np.asarray(u_input, dtype=float)

        if u.ndim == 0:
            u = u.reshape(1, 1)
        elif u.ndim == 1:
            u = u.reshape(-1, 1)

        self.x = self.Phi @ self.x + self.Gamma @ u
        y: NDArray[np.float64] = self.C @ self.x + self.D @ u

        if y.size == 1:
            return y.item()
        return y.flatten()

    def reset(self) -> None:
        self.x[:] = 0.0


@njit(cache=True)
def _rk_error_norm(
    x5: NDArray[np.float64], x4: NDArray[np.float64]
) -> Union[np.float64, float]:
    err: Union[np.float64, float] = 0.0
    for i in range(x5.size):
        diff: np.float64 = abs(x5.flat[i] - x4.flat[i])
        if diff > err:
            err = diff
    return err


class NonlinearSolver:
    """
    Adaptive Step-Size Solver for Non-Linear Systems.
    Implements the Dormand-Prince (RK5(4)7M) method.
    """

    def __init__(
        self,
        dynamics_func,
        dt_min: float = SOLVER_PARAMS["adaptive_dt_min"],
        dt_max: float = SOLVER_PARAMS["adaptive_dt_max"],
        tol: float = SOLVER_PARAMS["adaptive_tol"],
    ) -> None:
        self.f = dynamics_func
        self.dt_min: float = dt_min
        self.dt_max: float = dt_max
        self.tol: float = tol

        self.c: NDArray[np.float64] = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

        self.A_tableau: NDArray[np.float64] = np.zeros((7, 7))
        self.A_tableau[1, :1] = 1 / 5
        self.A_tableau[2, :2] = [3 / 40, 9 / 40]
        self.A_tableau[3, :3] = [44 / 45, -56 / 15, 32 / 9]
        self.A_tableau[4, :4] = [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]
        self.A_tableau[5, :5] = [
            9017 / 3168,
            -355 / 33,
            46732 / 5247,
            49 / 176,
            -5103 / 18656,
        ]
        self.A_tableau[6, :6] = [
            35 / 384,
            0,
            500 / 1113,
            125 / 192,
            -2187 / 6784,
            11 / 84,
        ]

        self.b: NDArray[np.float64] = np.array(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        )
        self.b_hat: NDArray[np.float64] = np.array(
            [
                5179 / 57600,
                0,
                7571 / 16695,
                393 / 640,
                -92097 / 339200,
                187 / 2100,
                1 / 40,
            ]
        )

    def solve_adaptive(
        self, t_end: float, x0, u_func=None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solves the IVP from t=0 to t_end using vectorized operations.
        """
        t: float = 0.0
        x: NDArray[np.float64] = np.asarray(x0, dtype=float).flatten()
        dt: float = SOLVER_PARAMS["adaptive_initial_dt"]

        t_hist = [t]
        x_hist = [x.copy()]

        k: NDArray[np.float64] = np.zeros((7, x.shape[0]))

        while t < t_end:
            if t + dt > t_end:
                dt = t_end - t

            u_val = u_func(t) if u_func else 0.0

            try:
                k[0] = self.f(t, x[:, None], u_val).flatten()
            except Exception:
                k[0] = self.f(t, x, u_val).flatten()

            for i in range(1, 7):
                A_tab: NDArray[np.float64] = self.A_tableau
                dx_sum: NDArray[np.float64] = A_tab[i, :i] @ k[:i]
                t_inner: float = t + self.c[i] * dt
                u_inner = u_func(t_inner) if u_func else 0.0
                k[i] = self.f(t_inner, x + dt * dx_sum, u_inner).flatten()

            x5: NDArray[np.float64] = x + dt * (self.b @ k)
            x4: NDArray[np.float64] = x + dt * (self.b_hat @ k)

            error: float = _rk_error_norm(x5, x4)

            if error < self.tol or dt <= self.dt_min:
                t += dt
                x = x5
                t_hist.append(t)
                x_hist.append(x.copy())

            if error == 0.0:
                dt *= 2.0
            else:
                dt *= (
                    SOLVER_PARAMS["safety_factor_1"]
                    * (self.tol / error) ** SOLVER_PARAMS["safety_factor_2"]
                )

            dt = min(max(dt, self.dt_min), self.dt_max)
        return np.array(t_hist), np.array(x_hist)
