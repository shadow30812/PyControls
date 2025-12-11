import numpy as np


class RK4Solver:
    def __init__(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        dt=0.001,
        dynamics_func=None,
        output_func=None,
    ):
        """
        A Hybrid Solver: Supports both Linear Matrices and Custom Non-Linear Functions.

        Mode 1: Linear (Matrices provided)
            dx/dt = Ax + Bu
            y     = Cx + Du

        Mode 2: Non-Linear (Functions provided)
            dx/dt = dynamics_func(x, u)
            y     = output_func(x, u)
        """
        self.dt = dt

        # Mode detection
        if dynamics_func is not None:
            self.mode = "nonlinear"
            self.f = dynamics_func
            self.h = output_func if output_func else lambda x, u: x[0, 0]
            # Default state (user overrides this in main.py usually)
            self.x = np.zeros((1, 1))
        else:
            self.mode = "linear"
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.x = np.zeros((A.shape[0], 1))

    def step(self, u_input):
        """Advances simulation by one time step dt."""

        # Define the derivative function based on mode
        if self.mode == "linear":

            def derivative(x, u):
                return self.A @ x + self.B * u
        else:

            def derivative(x, u):
                return self.f(x, u)

        # RK4 Integration Steps
        k1 = derivative(self.x, u_input)
        k2 = derivative(self.x + 0.5 * self.dt * k1, u_input)
        k3 = derivative(self.x + 0.5 * self.dt * k2, u_input)
        k4 = derivative(self.x + self.dt * k3, u_input)

        self.x += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Calculate Output
        if self.mode == "linear":
            y = self.C @ self.x + self.D * u_input
            return y[0, 0]
        else:
            return self.h(self.x, u_input)

    def reset(self):
        self.x = np.zeros_like(self.x)
