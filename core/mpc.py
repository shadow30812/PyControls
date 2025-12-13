import numpy as np


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

        self.Q = np.eye(self.x_dim) if Q is None else np.array(Q)
        self.R = np.eye(self.u_dim) if R is None else np.array(R)

        if A is not None and B is not None:
            print("MPC: Linear matrices detected. Using ADMM solver.")
            self.mode = "linear"
            self.A = np.array(A, dtype=float)
            self.B = np.array(B, dtype=float)
            self._setup_admm()
        else:
            print("MPC: No linear matrices. Using iLQR solver for nonlinear dynamics.")
            self.mode = "nonlinear"
            self.f = model_func
            self.A = None
            self.B = None
            if self.f is None:
                raise ValueError("For Nonlinear MPC, 'model_func' must be provided.")

        self.u_seq = np.zeros((self.N, self.u_dim))

    def _setup_admm(self):
        """
        Pre-computes the Condensed QP matrices for ADMM.
        This makes the real-time loop extremely fast (O(1) matrix multiplies).
        """
        Q_bar = np.kron(np.eye(self.N), self.Q)
        R_bar = np.kron(np.eye(self.N), self.R)

        S_u = np.zeros((self.N * self.x_dim, self.N * self.u_dim))

        for r in range(self.N):
            row_start = r * self.x_dim
            for c in range(r + 1):
                col_start = c * self.u_dim
                power = r - c
                mat = np.linalg.matrix_power(self.A, power) @ self.B
                S_u[
                    row_start : row_start + self.x_dim,
                    col_start : col_start + self.u_dim,
                ] = mat

        self.S_u = S_u

        self.H = self.S_u.T @ Q_bar @ self.S_u + R_bar

        self.rho = 1.0
        self.H_inv = np.linalg.inv(self.H + self.rho * np.eye(self.H.shape[0]))

        self.Q_bar = Q_bar

    def _solve_admm(self, x_current, x_ref, iterations=50):
        """
        Solves Min 1/2 U'HU + q'U s.t. u_min < U < u_max
        using Alternating Direction Method of Multipliers.
        """
        free_response = []
        curr = x_current.copy()
        for _ in range(self.N):
            curr = self.A @ curr
            free_response.append(curr)
        free_response = np.array(free_response).flatten()

        ref_vec = np.tile(x_ref, self.N)

        error_free = free_response - ref_vec

        q = self.S_u.T @ self.Q_bar @ error_free

        x_val = self.u_seq.flatten()
        z_val = x_val.copy()
        u_val = np.zeros_like(x_val)

        for _ in range(iterations):
            rhs = self.rho * (z_val - u_val) - q
            x_val = self.H_inv @ rhs
            z_val = np.clip(x_val + u_val, self.u_min, self.u_max)
            u_val = u_val + x_val - z_val

        return z_val.reshape(self.N, self.u_dim)

    def _get_derivatives(self, x, u):
        """Finite difference derivatives for iLQR (Robust for any user function)."""
        eps = 1e-5
        nx = self.x_dim
        nu = self.u_dim

        A = np.zeros((nx, nx))
        f0 = self.f(x, u, self.dt)
        for i in range(nx):
            x_p = x.copy()
            x_p[i] += eps
            A[:, i] = (self.f(x_p, u, self.dt) - f0) / eps

        B = np.zeros((nx, nu))
        for i in range(nu):
            u_p = u.copy()
            u_p[i] += eps
            B[:, i] = (self.f(x, u_p, self.dt) - f0) / eps

        return A, B

    def _solve_ilqr(self, x_current, x_ref, iterations=10):
        """
        Iterative Linear Quadratic Regulator (iLQR).
        Solves nonlinear optimal control by iteratively linearizing the dynamics
        and solving a time-varying LQR problem backward.
        """
        u_seq = self.u_seq.copy()
        x_seq = [x_current]

        curr = x_current.copy()
        for i in range(self.N):
            curr = self.f(curr, u_seq[i], self.dt)
            x_seq.append(curr)

        for _ in range(iterations):
            k_gains = []
            K_gains = []

            delta_x = x_seq[-1] - x_ref
            V_x = self.Q @ delta_x
            V_xx = self.Q

            for i in range(self.N - 1, -1, -1):
                x_i = x_seq[i]
                u_i = u_seq[i]

                A_k, B_k = self._get_derivatives(x_i, u_i)

                l_x = self.Q @ (x_i - x_ref)
                l_u = self.R @ u_i
                l_xx = self.Q
                l_uu = self.R

                Q_x = l_x + A_k.T @ V_x
                Q_u = l_u + B_k.T @ V_x
                Q_xx = l_xx + A_k.T @ V_xx @ A_k
                Q_uu = l_uu + B_k.T @ V_xx @ B_k
                Q_ux = A_k.T @ V_xx @ B_k

                Q_uu_reg = Q_uu + np.eye(self.u_dim) * 1e-6
                Q_uu_inv = np.linalg.inv(Q_uu_reg)

                k = -Q_uu_inv @ Q_u
                K = -Q_uu_inv @ Q_ux.T

                k_gains.append(k)
                K_gains.append(K)

                V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux @ k
                V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux.T + Q_ux @ K

            k_gains = k_gains[::-1]
            K_gains = K_gains[::-1]

            x_new_seq = [x_current]
            u_new_seq = np.zeros_like(u_seq)
            curr = x_current.copy()

            for i in range(self.N):
                delta_x = curr - x_seq[i]
                u_ctrl = u_seq[i] + k_gains[i] + K_gains[i] @ delta_
                u_ctrl = np.clip(u_ctrl, self.u_min, self.u_max)

                u_new_seq[i] = u_ctrl
                curr = self.f(curr, u_ctrl, self.dt)
                x_new_seq.append(curr)

            u_seq = u_new_seq
            x_seq = x_new_seq

        return u_seq

    def optimize(self, x_current, x_ref, **kwargs):
        """
        Computes the optimal control input.
        ILQR is typically more expensive per iteration
        than ADMM.
        """
        x_current = np.array(x_current, dtype=float)
        x_ref = np.array(x_ref, dtype=float)

        if self.mode == "linear":
            iters = kwargs.get("iterations", 50)
            self.u_seq = self._solve_admm(x_current, x_ref, iterations=iters)
        else:
            iters = kwargs.get("iterations", 10)
            self.u_seq = self._solve_ilqr(x_current, x_ref, iterations=iters)

        u_optimal = self.u_seq[0]

        self.u_seq = np.roll(self.u_seq, -1, axis=0)
        self.u_seq[-1] = 0.0

        return u_optimal
