import numpy as np

from helpers.config import MPC_SOLVER_PARAMS


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
        model_func=None,
        x0=None,
        horizon=10,
        dt=0.1,
        Q=None,
        R=None,
        u_min=-10,
        u_max=10,
        A=None,
        B=None,
    ):
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
        self.dt = dt
        self.N = horizon
        self.u_min = u_min
        self.u_max = u_max

        if x0 is not None:
            self.x_dim = len(x0)
        elif A is not None:
            self.x_dim = A.shape[0]
        else:
            raise ValueError("Must provide either x0 or A matrix to infer dimensions.")

        if B is not None:
            self.u_dim = B.shape[1]
        else:
            self.u_dim = 1

        self.Q = np.eye(self.x_dim) if Q is None else np.array(Q, dtype=float)
        self.R = np.eye(self.u_dim) if R is None else np.array(R, dtype=float)

        if A is not None and B is not None:
            print("\nMPC: Linear matrices detected. Using ADMM solver.\n")
            self.mode = "linear"
            self.A = np.array(A, dtype=float)
            self.B = np.array(B, dtype=float)
            self._setup_admm()
        else:
            print(
                "\nMPC: No linear matrices. Using iLQR solver for nonlinear dynamics.\n"
            )
            self.mode = "nonlinear"
            self.f = model_func
            self.A = None
            self.B = None
            if self.f is None:
                raise ValueError("For Nonlinear MPC, 'model_func' must be provided.")

        self.u_seq = np.zeros((self.N, self.u_dim))
        self._x_seq = np.zeros((self.N + 1, self.x_dim))
        self._k = np.zeros((self.N, self.u_dim))
        self._K = np.zeros((self.N, self.u_dim, self.x_dim))

    def _setup_admm(self):
        """
        Pre-computes the Condensed QP matrices for ADMM.
        This makes the real-time loop extremely fast (O(1) matrix multiplies).
        """
        Q_bar = np.kron(np.eye(self.N), self.Q)
        R_bar = np.kron(np.eye(self.N), self.R)

        S_u = np.zeros((self.N * self.x_dim, self.N * self.u_dim))

        A_powers = [np.eye(self.x_dim)]
        for _ in range(self.N):
            A_powers.append(A_powers[-1] @ self.A)

        for r in range(self.N):
            row_start = r * self.x_dim
            for c in range(r + 1):
                col_start = c * self.u_dim
                mat = A_powers[r - c] @ self.B
                S_u[
                    row_start : row_start + self.x_dim,
                    col_start : col_start + self.u_dim,
                ] = mat

        self.S_u = S_u
        self.H = self.S_u.T @ Q_bar @ self.S_u + R_bar
        self.rho = MPC_SOLVER_PARAMS["rho"]
        self.H_inv = np.linalg.inv(self.H + self.rho * np.eye(self.H.shape[0]))
        self.Q_bar = Q_bar
        self._A_powers = A_powers

    def _solve_admm(self, x_current, x_ref, iterations=50):
        """
        Solves Min 1/2 U'HU + q'U s.t. u_min < U < u_max
        using Alternating Direction Method of Multipliers.
        """
        free_response = np.zeros(self.N * self.x_dim)
        for i in range(self.N):
            free_response[i * self.x_dim : (i + 1) * self.x_dim] = (
                self._A_powers[i + 1] @ x_current
            )

        ref_vec = np.tile(x_ref, self.N)
        q = self.S_u.T @ self.Q_bar @ (free_response - ref_vec)

        x_val = self.u_seq.reshape(-1)
        z_val = x_val.copy()
        u_val = np.zeros_like(x_val)

        for _ in range(iterations):
            rhs = self.rho * (z_val - u_val) - q
            x_val = self.H_inv @ rhs
            z_val = np.clip(x_val + u_val, self.u_min, self.u_max)
            u_val += x_val - z_val

        return z_val.reshape(self.N, self.u_dim)

    def _get_derivatives(self, x, u):
        """Complex step derivatives for iLQR (Robust for any user function)."""
        eps = 1e-20
        nx = self.x_dim
        nu = self.u_dim

        A = np.zeros((nx, nx))
        for i in range(nx):
            dx = np.zeros(nx, dtype=complex)
            dx[i] = 1j * eps
            A[:, i] = self.f((x.astype(complex) + dx), u, self.dt).imag / eps

        B = np.zeros((nx, nu))
        for i in range(nu):
            du = np.zeros(nu, dtype=complex)
            du[i] = 1j * eps
            B[:, i] = (
                self.f(x.astype(complex), (u.astype(complex) + du), self.dt).imag / eps
            )

        return A, B

    def _solve_ilqr(self, x_current, x_ref, iterations=10):
        """
        Iterative Linear Quadratic Regulator (iLQR).
        Solves nonlinear optimal control by iteratively linearizing the dynamics
        and solving a time-varying LQR problem backward.
        """
        self._x_seq[0] = x_current
        for i in range(self.N):
            self._x_seq[i + 1] = self.f(self._x_seq[i], self.u_seq[i], self.dt)

        for _ in range(iterations):
            V_x = self.Q @ (self._x_seq[-1] - x_ref)
            V_xx = self.Q

            for i in range(self.N - 1, -1, -1):
                x_i = self._x_seq[i]
                u_i = self.u_seq[i]

                A_k, B_k = self._get_derivatives(x_i, u_i)

                l_x = self.Q @ (x_i - x_ref)
                l_u = self.R @ u_i

                Q_x = l_x + A_k.T @ V_x
                Q_u = l_u + B_k.T @ V_x
                Q_xx = self.Q + A_k.T @ V_xx @ A_k
                Q_uu = self.R + B_k.T @ V_xx @ B_k
                Q_ux = A_k.T @ V_xx @ B_k

                Q_uu_reg = Q_uu + np.eye(self.u_dim) * MPC_SOLVER_PARAMS["ilqr_reg"]
                Q_uu_inv = np.linalg.inv(Q_uu_reg)

                self._k[i] = -Q_uu_inv @ Q_u
                self._K[i] = -Q_uu_inv @ Q_ux.T

                V_x = Q_x + self._K[i].T @ Q_uu @ self._k[i]
                V_xx = Q_xx + self._K[i].T @ Q_uu @ self._K[i]

            alphas = MPC_SOLVER_PARAMS["ilqr_alphas"]

            best_cost = np.inf
            best_u_seq = None
            best_x_seq = None

            for alpha in alphas:
                curr = x_current
                x_seq = np.zeros_like(self._x_seq)
                u_seq = np.zeros_like(self.u_seq)
                x_seq[0] = curr
                cost = 0.0

                for i in range(self.N):
                    u = (
                        self.u_seq[i]
                        + alpha * self._k[i]
                        + self._K[i] @ (curr - self._x_seq[i])
                    )
                    u = np.clip(u, self.u_min, self.u_max)
                    u_seq[i] = u
                    cost += (curr - x_ref).T @ self.Q @ (
                        curr - x_ref
                    ) + u.T @ self.R @ u
                    curr = self.f(curr, u, self.dt)
                    x_seq[i + 1] = curr

                cost += (curr - x_ref).T @ self.Q @ (curr - x_ref)

                if cost < best_cost:
                    best_cost = cost
                    best_u_seq = u_seq
                    best_x_seq = x_seq

            self.u_seq[:] = best_u_seq
            self._x_seq[:] = best_x_seq

        return self.u_seq

    def optimize(self, x_current, x_ref, **kwargs):
        """
        Computes the optimal control input.
        ILQR is typically more expensive per iteration
        than ADMM.
        """
        x_current = np.array(x_current, dtype=float)
        x_ref = np.array(x_ref, dtype=float)

        if self.mode == "linear":
            iters = kwargs.get(
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

        u_optimal = self.u_seq[0].copy()

        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1] = 0.0

        return u_optimal
