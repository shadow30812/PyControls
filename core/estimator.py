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

    def predict(self, u, dt=None):
        """
        Performs the a priori prediction step.
        x[k|k-1] = Phi * x[k-1|k-1] + Gamma * u[k]
        P[k|k-1] = Phi * P[k-1|k-1] * Phi' + Q

        Args:
            u: Control input vector.
            dt: Time step (unused here as Phi/Gamma are already discrete,
                but kept for interface consistency with EKF/UKF).
        """
        u = np.atleast_2d(u)
        if u.shape[0] == 1 and u.shape[1] != 1:
            u = u.T

        self.x_hat = self.Phi @ self.x_hat + self.Gamma @ u
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q

    def update(self, y_meas):
        """
        Performs the a posteriori correction step.
        x[k|k] = x[k|k-1] + K * (y - C * x[k|k-1])
        P[k|k] = (I - K * C) * P[k|k-1]

        Args:
            y_meas: Noisy measurement vector.

        Returns:
            np.array: The updated state estimate (flattened).
        """
        y_meas = np.atleast_2d(y_meas)
        if y_meas.shape[0] == 1 and y_meas.shape[1] != 1:
            y_meas = y_meas.T

        y_pred = self.C @ self.x_hat
        y_err = y_meas - y_pred

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        self.x_hat = self.x_hat + K @ y_err

        I = np.eye(self.x_hat.shape[0])
        self.P = (I - K @ self.C) @ self.P

        return self.x_hat.flatten()
