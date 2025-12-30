from typing import Any, Callable, Final, List, Optional, Tuple, TypeAlias, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helpers.config import MPC_SOLVER_PARAMS

NumericArray: TypeAlias = Union[NDArray[np.float64], NDArray[np.complex128]]


class ModelPredictiveControl:
    """
    Multi-Purpose Model Predictive Controller (MPC).

    Automatically selects the best algorithm based on the provided system model:
    1. Linear System (A, B provided) -> ADMM (Alternating Direction Method of Multipliers)
       - extremely fast, robust constraint handling, globally optimal.
    2. Nonlinear System (model_func provided) -> iLQR (Iterative Linear Quadratic Regulator)
       - standard for robotics, handles complex dynamics, locally optimal.
    """

    def __init__(
        self,
        model_func: Optional[
            Callable[
                [NDArray[np.float64], NDArray[np.float64], float],
                NDArray[np.float64],
            ]
        ] = None,
        x0: Optional[NDArray] = None,
        horizon: int = 10,
        dt: float = 0.1,
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        u_min: Union[float, NDArray[np.float64]] = -10,
        u_max: Union[float, NDArray[np.float64]] = 10,
        A: Optional[NDArray] = None,
        B: Optional[NDArray] = None,
    ) -> None:
        """
        Args:
            model_func: Function f(x, u, dt) -> x_next (Required for Nonlinear/iLQR)
            x0: Initial State (used for dimension inference)
            horizon: Prediction horizon steps (N)
            dt: Time step
            Q: State Cost Matrix (n x n)
            R: Input Cost Matrix (m x m)
            u_min, u_max: Control constraints
            A, B: Discrete Linear System Matrices (Optional - triggers ADMM if provided)
        """
        self.dt: Final[float] = dt
        self.N: Final[int] = horizon
        self.u_min: Final[Union[float, NDArray[np.float64]]] = u_min
        self.u_max: Final[Union[float, NDArray[np.float64]]] = u_max

        if x0 is not None:
            self.x_dim: int = len(x0)
        elif A is not None:
            self.x_dim = A.shape[0]
        else:
            raise ValueError("Must provide either x0 or A matrix to infer dimensions.")

        if B is not None:
            self.u_dim: int = B.shape[1]
        else:
            self.u_dim = 1

        self.Q: Final[NDArray[np.float64]] = (
            np.eye(self.x_dim) if Q is None else np.array(Q, dtype=float)
        )
        self.R: Final[NDArray[np.float64]] = (
            np.eye(self.u_dim) if R is None else np.array(R, dtype=float)
        )

        if A is not None and B is not None:
            print("\nMPC: Linear matrices detected. Using ADMM solver.\n")
            self.mode: str = "linear"
            self.A: Optional[NDArray[np.float64]] = np.array(A, dtype=float)
            self.B: Optional[NDArray[np.float64]] = np.array(B, dtype=float)
            self._setup_admm()
        else:
            print(
                "\nMPC: No linear matrices. Using iLQR solver for nonlinear dynamics.\n"
            )
            self.mode = "nonlinear"
            self.f: Optional[
                Callable[
                    [NumericArray, NumericArray, Union[float, complex]],
                    NumericArray,
                ]
            ] = None

            if model_func is not None:
                self.f = cast(
                    Callable[
                        [NumericArray, NumericArray, Union[float, complex]],
                        NumericArray,
                    ],
                    model_func,
                )
            else:
                self.f = None

            self.A = None
            self.B = None
            if self.f is None:
                raise ValueError("For Nonlinear MPC, 'model_func' must be provided.")

        self.u_seq: NDArray[np.float64] = np.zeros((self.N, self.u_dim))
        self._x_seq: NDArray[np.float64] = np.zeros((self.N + 1, self.x_dim))
        self._k: NDArray[np.float64] = np.zeros((self.N, self.u_dim))
        self._K: NDArray[np.float64] = np.zeros((self.N, self.u_dim, self.x_dim))

        self._A_seq: NDArray[np.float64] = np.zeros((self.N, self.x_dim, self.x_dim))
        self._B_seq: NDArray[np.float64] = np.zeros((self.N, self.x_dim, self.u_dim))

    def _setup_admm(self) -> None:
        """
        Pre-computes the Condensed QP matrices for ADMM.
        This makes the real-time loop extremely fast (O(1) matrix multiplies).
        """
        if self.A is None or self.B is None:
            return

        Q_bar: NDArray[np.floating] = np.kron(np.eye(self.N), self.Q)
        R_bar: NDArray[np.floating] = np.kron(np.eye(self.N), self.R)

        S_u: NDArray[np.float64] = np.zeros((self.N * self.x_dim, self.N * self.u_dim))

        A_powers: List[NDArray[np.float64]] = [np.eye(self.x_dim)]
        for _ in range(self.N):
            A_powers.append(A_powers[-1] @ self.A)

        for r in range(self.N):
            row_start: int = r * self.x_dim
            for c in range(r + 1):
                col_start: int = c * self.u_dim
                mat: NDArray[np.float64] = A_powers[r - c] @ self.B
                S_u[
                    row_start : row_start + self.x_dim,
                    col_start : col_start + self.u_dim,
                ] = mat

        self.S_u: NDArray[np.float64] = S_u
        self.H: NDArray[np.float64] = self.S_u.T @ Q_bar @ self.S_u + R_bar
        self.rho: float = MPC_SOLVER_PARAMS["rho"]
        self.H_inv: NDArray[np.floating] = np.linalg.inv(
            self.H + self.rho * np.eye(self.H.shape[0])
        )
        self.Q_bar: NDArray[np.floating] = Q_bar
        self._A_powers: List[NDArray[np.float64]] = A_powers

    def _solve_admm(
        self,
        x_current: NDArray[np.float64],
        x_ref: NDArray[np.float64],
        iterations: int = 50,
    ) -> NDArray[np.float64]:
        """
        Solves Min 1/2 U'HU + q'U s.t. u_min < U < u_max
        using Alternating Direction Method of Multipliers.
        """
        free_response: NDArray[np.float64] = np.zeros(self.N * self.x_dim)
        for i in range(self.N):
            free_response[i * self.x_dim : (i + 1) * self.x_dim] = (
                self._A_powers[i + 1] @ x_current
            )

        ref_vec: NDArray[np.float64] = np.tile(x_ref, self.N)
        q: NDArray[np.float64] = self.S_u.T @ self.Q_bar @ (free_response - ref_vec)

        x_val: NDArray[np.floating] = self.u_seq.reshape(-1)
        z_val: NDArray[np.float64] = x_val.copy()
        u_val: NDArray[np.float64] = np.zeros_like(x_val)

        for _ in range(iterations):
            rhs: NDArray[np.float64] = self.rho * (z_val - u_val) - q
            x_val = np.linalg.solve(self.H + self.rho * np.eye(self.H.shape[0]), rhs)
            z_val = np.clip(x_val + u_val, self.u_min, self.u_max)
            u_val += x_val - z_val

        return z_val.reshape(self.N, self.u_dim)

    def _get_derivatives(
        self, x: NDArray[np.float64], u: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Complex step derivatives for iLQR (Robust for any user function)."""
        eps: Final[float] = 1e-20
        nx: int = self.x_dim
        nu: int = self.u_dim

        A: NDArray[np.float64] = np.zeros((nx, nx))
        B: NDArray[np.float64] = np.zeros((nx, nu))

        if not self.f:
            return (A, B)

        x_c: NDArray[np.complex128] = x.astype(np.complex128)
        u_c: NDArray[np.complex128] = u.astype(np.complex128)

        for i in range(nx):
            dx: NDArray[np.complex128] = np.zeros(nx, dtype=np.complex128)
            dx[i] = 1j * eps
            fx: NumericArray = self.f(x_c + dx, u_c, self.dt)
            A[:, i] = fx.imag / eps

        for j in range(nu):
            du: NDArray[np.complex128] = np.zeros(nu, dtype=np.complex128)
            du[j] = 1j * eps
            fx = self.f(x_c, u_c + du, self.dt)
            B[:, j] = fx.imag / eps

        return A.real, B.real

    def _solve_ilqr(
        self,
        x_current: NDArray[np.float64],
        x_ref: NDArray[np.float64],
        iterations: int = 10,
    ) -> NDArray[np.float64]:
        """
        Iterative Linear Quadratic Regulator (iLQR).
        Solves nonlinear optimal control by iteratively linearizing the dynamics
        and solving a time-varying LQR problem backward.
        """
        if not self.f:
            return np.zeros(self.x_dim)

        if iterations <= 0:
            return self.u_seq

        tol: float = MPC_SOLVER_PARAMS["ilqr_tol"]
        if hasattr(self, "_prev_k_norm") and self._prev_k_norm < 1e-3:
            iterations = min(iterations, 3)
        iterations_used: int = 0

        self._x_seq[0] = x_current
        for i in range(self.N):
            self._x_seq[i + 1] = self.f(self._x_seq[i], self.u_seq[i], self.dt)

        for it in range(iterations):
            iterations_used += 1
            prev_k: NDArray[np.float64] = self._k.copy()

            V_x: NDArray[np.float64] = self.Q @ (self._x_seq[-1] - x_ref)
            V_xx: NDArray[np.float64] = self.Q

            for i in range(self.N - 1, -1, -1):
                x_i: NDArray[np.float64] = self._x_seq[i]
                u_i: NDArray[np.float64] = self.u_seq[i]

                if hasattr(self, "model") and hasattr(self.model, "linearize"):
                    if it < 2:
                        kk: Tuple[NDArray[np.float64], NDArray[np.float64]] = (
                            self.model.linearize(x_i, u_i, self.dt)
                        )
                        A_k, B_k = kk
                        self._A_seq[i] = A_k
                        self._B_seq[i] = B_k
                    else:
                        A_k: NDArray[np.float64] = self._A_seq[i]
                        B_k: NDArray[np.float64] = self._B_seq[i]
                else:
                    A_k, B_k = self._get_derivatives(x_i, u_i)

                l_x: NDArray[np.float64] = self.Q @ (x_i - x_ref)
                l_u: NDArray[np.float64] = self.R @ u_i

                Q_x: NDArray[np.float64] = l_x + A_k.T @ V_x
                Q_u: NDArray[np.float64] = l_u + B_k.T @ V_x

                Q_xx: NDArray[np.float64] = self.Q + A_k.T @ V_xx @ A_k
                Q_uu: NDArray[np.float64] = self.R + B_k.T @ V_xx @ B_k
                Q_ux: NDArray[np.float64] = A_k.T @ V_xx @ B_k

                Q_uu_reg: NDArray[np.float64] = (
                    Q_uu + np.eye(self.u_dim) * MPC_SOLVER_PARAMS["ilqr_reg"]
                )

                self._k[i] = -np.linalg.solve(Q_uu_reg, Q_u)
                self._K[i] = -np.linalg.solve(Q_uu_reg, Q_ux.T)

                V_x = Q_x + self._K[i].T @ Q_uu @ self._k[i]
                V_xx = Q_xx + self._K[i].T @ Q_uu @ self._K[i]

            max_k_delta: float = np.max(np.abs(self._k - prev_k))
            self._prev_k_norm: np.floating[Any] = np.mean(np.abs(self._k))

            best_cost: float = np.inf
            best_u_seq: Optional[NDArray[np.float64]] = None
            best_x_seq: Optional[NDArray[np.float64]] = None

            for alpha in MPC_SOLVER_PARAMS["ilqr_alphas"]:
                curr = x_current
                x_seq: NDArray[np.float64] = np.zeros_like(self._x_seq)
                u_seq: NDArray[np.float64] = np.zeros_like(self.u_seq)
                x_seq[0] = curr
                cost: float = 0.0

                for i in range(self.N):
                    u: NDArray[np.float64] = (
                        self.u_seq[i]
                        + alpha * self._k[i]
                        + self._K[i] @ (curr - self._x_seq[i])
                    )
                    u = np.clip(u, self.u_min, self.u_max)
                    u_seq[i] = u

                    cost += float((curr - x_ref).T @ self.Q @ (curr - x_ref))
                    cost += float(np.real(u.T @ self.R @ u))

                    curr: NDArray[np.float64] = np.asarray(
                        self.f(curr, u, self.dt), dtype=float
                    )
                    x_seq[i + 1] = curr

                cost += float((curr - x_ref).T @ self.Q @ (curr - x_ref))

                if cost < best_cost:
                    best_cost = cost
                    best_u_seq = u_seq
                    best_x_seq = x_seq

            if best_u_seq is None or best_x_seq is None:
                raise RuntimeError("iLQR failed to find a valid rollout")

            self.u_seq[:] = best_u_seq
            self._x_seq[:] = best_x_seq

            if max_k_delta < tol:
                break

        return self.u_seq

    def optimize(
        self,
        x_current: Union[ArrayLike, NDArray[np.float64]],
        x_ref: Union[ArrayLike, NDArray[np.float64]],
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        """
        Computes the optimal control input.
        ILQR is typically more expensive per iteration
        than ADMM.
        """
        x_current = np.array(x_current, dtype=float)
        x_ref = np.array(x_ref, dtype=float)

        if self.mode == "linear":
            iters: int = kwargs.get(
                "iterations",
                MPC_SOLVER_PARAMS["default_linear_iters"],
            )
            self.u_seq = self._solve_admm(x_current, x_ref, iterations=iters)
        else:
            iters = kwargs.get(
                "iterations",
                MPC_SOLVER_PARAMS["default_nonlinear_iters"],
            )
            self.u_seq = self._solve_ilqr(x_current, x_ref, iterations=iters)

        u_optimal: NDArray[np.float64] = self.u_seq[0].copy()

        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1] = 0.0

        return u_optimal
