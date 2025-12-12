import numpy as np


def manual_matrix_exp(A, order=20):
    """
    Computes the matrix exponential e^A using Scaling and Squaring with Taylor Series.
    Formula: e^A = (e^(A/2^s))^(2^s)

    This implementation does not rely on scipy.linalg.expm, keeping the library
    dependency-free for core math.
    """
    norm_A = np.max(np.sum(np.abs(A), axis=1))
    s = 0
    while norm_A > 0.5:
        norm_A /= 2.0
        s += 1

    A_scaled = A / (2**s)

    E = np.eye(A.shape[0])
    term = np.eye(A.shape[0])

    for k in range(1, order + 1):
        term = term @ A_scaled / k
        E = E + term

    for _ in range(s):
        E = E @ E

    return E


class ExactSolver:
    """
    Exact Discrete-Time Solver for Linear Time-Invariant (LTI) Systems.
    Uses the Zero-Order Hold (ZOH) method to discretize continuous matrices.
    """

    def __init__(self, A, B, C, D, dt):
        """
        Computes the discrete transition matrices (Phi and Gamma) upon initialization.

        Continuous: dx/dt = Ax + Bu
        Discrete:   x[k+1] = Phi * x[k] + Gamma * u[k]

        Phi = e^(A*dt)
        Gamma = Integral(e^(A*tau) * B) from 0 to dt
        """
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.C = np.atleast_2d(C)
        self.D = np.atleast_2d(D)

        self.x = np.zeros((self.A.shape[0], 1))

        n_states = self.A.shape[0]
        n_inputs = self.B.shape[1]

        top = np.hstack((self.A, self.B))
        bottom = np.zeros((n_inputs, n_states + n_inputs))
        M = np.vstack((top, bottom))

        M_exp = manual_matrix_exp(M * dt)

        self.Phi = M_exp[:n_states, :n_states]
        self.Gamma = M_exp[:n_states, n_states:]

    def step(self, u_input):
        """
        Advances the simulation by one discrete time step.
        """
        u = np.array(u_input, dtype=float)
        if u.ndim == 0:
            u = u.reshape(1, 1)
        elif u.ndim == 1:
            u = u.reshape(-1, 1)

        self.x = self.Phi @ self.x + self.Gamma @ u
        y = self.C @ self.x + self.D @ u

        if y.size == 1:
            return y.item()
        return y.flatten()

    def reset(self):
        """Resets the internal state to zero."""
        self.x = np.zeros_like(self.x)


class NonlinearSolver:
    """
    Adaptive Step-Size Solver for Non-Linear Systems.
    Implements the Dormand-Prince (RK5(4)) method, often known as RK45.
    Adjusts the integration step size (dt) automatically based on the local error estimate.
    """

    def __init__(self, dynamics_func, dt_min=1e-5, dt_max=0.5, tol=1e-6):
        self.f = dynamics_func
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol

        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
        self.a = [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ]
        self.b = np.array(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        )
        self.b_hat = np.array(
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

    def solve_adaptive(self, t_end, x0, u_func=None):
        """
        Solves the IVP from t=0 to t_end.

        Args:
            t_end: Final simulation time.
            x0: Initial state vector.
            u_func: Optional function u(t) for time-varying inputs.

        Returns:
            tuple: (time_array, state_history_array)
        """
        t = 0.0
        x = x0.astype(float)
        dt = 0.001

        t_hist = [t]
        x_hist = [x]

        while t < t_end:
            if t + dt > t_end:
                dt = t_end - t

            k = np.zeros((7, x.shape[0]))
            u_val = u_func(t) if u_func else 0.0

            k[0] = self.f(t, x, u_val).flatten()

            for i in range(1, 7):
                dx_sum = np.zeros_like(x)
                for j in range(i):
                    dx_sum += self.a[i][j] * k[j]

                t_inner = t + self.c[i] * dt
                u_inner = u_func(t_inner) if u_func else 0.0
                k[i] = self.f(t_inner, x + dt * dx_sum, u_inner).flatten()

            x_5 = x + dt * (self.b @ k)
            x_4 = x + dt * (self.b_hat @ k)

            error = np.max(np.abs(x_5 - x_4))

            if error < self.tol or dt <= self.dt_min:
                t += dt
                x = x_5
                t_hist.append(t)
                x_hist.append(x)

            if error == 0:
                dt_new = dt * 2
            else:
                dt_new = 0.9 * dt * (self.tol / error) ** 0.2

            dt = max(self.dt_min, min(dt_new, self.dt_max))

        return np.array(t_hist), np.array(x_hist)
