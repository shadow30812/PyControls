import numpy as np


# --- 1. Matrix Operations ---
def manual_matrix_exp(A, order=10):
    """
    Computes matrix exponential using Scaling and Squaring with Taylor Series.
    e^A = (e^(A/2^s))^(2^s)
    """
    # 1. Scaling: Find s such that norm(A) is small (e.g., < 0.5)
    norm_A = np.max(np.sum(np.abs(A), axis=1))
    s = 0
    while norm_A > 0.5:
        norm_A /= 2.0
        s += 1

    A_scaled = A / (2**s)

    # 2. Taylor Series Approximation for small A_scaled
    # E = I + A + A^2/2! + ...
    E = np.eye(A.shape[0])
    term = np.eye(A.shape[0])

    for k in range(1, order + 1):
        term = term @ A_scaled / k
        E = E + term

    # 3. Squaring: Square the result s times
    for _ in range(s):
        E = E @ E

    return E


# --- 2. Solvers ---
class ExactSolver:
    """
    Exact Discretization (Zero-Order Hold) for Linear Systems.
    Uses custom matrix exponential implementation.
    """

    def __init__(self, A, B, C, D, dt):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x = np.zeros((A.shape[0], 1))

        # Build combined matrix M for simultaneous Phi and Gamma calc
        # [ A  B ]
        # [ 0  0 ]
        n_states = A.shape[0]
        n_inputs = B.shape[1]

        top = np.hstack((A, B))
        bottom = np.zeros((n_inputs, n_states + n_inputs))
        M = np.vstack((top, bottom))

        # Compute Matrix Exp manually
        M_exp = manual_matrix_exp(M * dt)

        # Extract discrete matrices
        self.Phi = M_exp[:n_states, :n_states]
        self.Gamma = M_exp[:n_states, n_states:]

    def step(self, u_input):
        self.x = self.Phi @ self.x + self.Gamma * u_input
        y = self.C @ self.x + self.D * u_input
        return y[0, 0]

    def reset(self):
        self.x = np.zeros_like(self.x)


class NonlinearSolver:
    """
    Adaptive Step-Size Solver using Manual Dormand-Prince (RK45/RK5(4)7M) method.
    """

    def __init__(self, dynamics_func, dt_min=1e-5, dt_max=0.5, tol=1e-6):
        self.f = dynamics_func
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tol = tol

        # Dormand-Prince Coefficients (RK5(4))
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
        )  # Order 5
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
        )  # Order 4

    def solve_adaptive(self, t_end, x0, u_func=None):
        t = 0.0
        x = x0.astype(float)
        dt = 0.001  # Initial guess

        t_hist = [t]
        x_hist = [x]

        while t < t_end:
            # Cap dt to hit t_end exactly
            if t + dt > t_end:
                dt = t_end - t

            # 1. Compute Stages k1-k7 (7M)
            k = np.zeros((7, x.shape[0]))

            # Get input u at current t
            u_val = u_func(t) if u_func else 0.0
            k[0] = self.f(t, x, u_val).flatten()

            for i in range(1, 7):
                dx_sum = np.zeros_like(x)
                for j in range(i):
                    dx_sum += self.a[i][j] * k[j]

                t_inner = t + self.c[i] * dt
                u_inner = u_func(t_inner) if u_func else 0.0
                k[i] = self.f(t_inner, x + dt * dx_sum, u_inner).flatten()

            # 2. Compute Updates (Order 5 and Order 4)
            x_5 = x + dt * (self.b @ k)
            x_4 = x + dt * (self.b_hat @ k)

            # 3. Error Estimation
            error = np.max(np.abs(x_5 - x_4))

            # 4. Step Size Control
            if error < self.tol or dt <= self.dt_min:
                # Accept Step
                t += dt
                x = x_5
                t_hist.append(t)
                x_hist.append(x)

            # Calculate new dt for next step (or retry)
            if error == 0:
                dt_new = dt * 2
            else:
                dt_new = 0.9 * dt * (self.tol / error) ** 0.2

            dt = max(self.dt_min, min(dt_new, self.dt_max))

        return np.array(t_hist), np.array(x_hist)
