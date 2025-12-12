import numpy as np


class KalmanFilter:
    """
    Standard Linear Discrete-Time Kalman Filter.
    Estimates the state x of a linear system from noisy measurements y.

    System Model:
    x[k+1] = Phi * x[k] + Gamma * u[k] + w[k]
    y[k]   = C * x[k] + v[k]
    """

    def __init__(self, A, B, C, Q, R, x0):
        """
        Args:
            A (Phi): State transition matrix (Discrete).
            B (Gamma): Input control matrix (Discrete).
            C: Measurement matrix.
            Q: Process noise covariance.
            R: Measurement noise covariance.
            x0: Initial state guess.
        """
        self.Phi = A
        self.Gamma = B
        self.C = C

        self.Q = Q
        self.R = R

        self.x_hat = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.eye(self.x_hat.shape[0]) * 0.1

    def update(self, u, y_meas):
        """
        Performs both Prediction and Correction in a single step.

        Args:
            u: Known control input vector.
            y_meas: Noisy measurement vector.

        Returns:
            np.array: The updated state estimate.
        """
        u = np.atleast_2d(u)
        y_meas = np.atleast_2d(y_meas)
        if u.shape[0] == 1 and u.shape[1] != 1:
            u = u.T
        if y_meas.shape[0] == 1 and y_meas.shape[1] != 1:
            y_meas = y_meas.T

        x_pred = self.Phi @ self.x_hat + self.Gamma @ u
        P_pred = self.Phi @ self.P @ self.Phi.T + self.Q

        y_pred = self.C @ x_pred
        y_err = y_meas - y_pred

        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        self.x_hat = x_pred + K @ y_err

        I = np.eye(self.x_hat.shape[0])
        self.P = (I - K @ self.C) @ P_pred

        return self.x_hat.flatten()
