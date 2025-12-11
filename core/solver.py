import numpy as np


class RK4Solver:
    def __init__(self, A, B, C, D, dt=0.001):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt
        self.x = np.zeros((A.shape[0], 1))

    def step(self, u_input):
        def f(x_curr, u_curr):
            return self.A @ x_curr + self.B * u_curr

        k1 = f(self.x, u_input)
        k2 = f(self.x + 0.5 * self.dt * k1, u_input)
        k3 = f(self.x + 0.5 * self.dt * k2, u_input)
        k4 = f(self.x + self.dt * k3, u_input)

        self.x += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = self.C @ self.x + self.D * u_input
        return y[0, 0]

    def reset(self):
        self.x = np.zeros_like(self.x)
