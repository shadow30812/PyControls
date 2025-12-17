import numpy as np

from core.math_utils import jacobian
from core.solver import manual_matrix_exp


class DiscreteExtendedKalmanFilter:
    """
    Textbook discrete-time Extended Kalman Filter.
    """

    def __init__(self, f, h, Q, R, x0, dt):
        self.f = f
        self.h = h
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.atleast_2d(x0).astype(float).T
        self.P = np.eye(self.x.shape[0])
        self.dt = dt

    def predict(self, u=None):
        x_flat = self.x.flatten()

        A = jacobian(lambda x: self.f(x, u), x_flat)
        Phi = manual_matrix_exp(A * self.dt)

        self.x = Phi @ self.x
        self.P = Phi @ self.P @ Phi.T + self.Q

    def update(self, y):
        y = np.atleast_2d(y).T
        x_flat = self.x.flatten()

        H = jacobian(self.h, x_flat)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        y_err = y - self.h(x_flat).reshape(-1, 1)

        I = np.eye(self.P.shape[0])
        self.x = self.x + K @ y_err
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
