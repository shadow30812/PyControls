import numpy as np


class KalmanFilter:
    """
    Discrete-Time Kalman Filter.
    Estimates state x from noisy measurements y.
    """

    def __init__(self, A, B, C, Q, R, x0):
        # System Matrices (Discrete)
        self.Phi = A
        self.Gamma = B
        self.C = C

        # Covariance Matrices
        self.Q = Q  # Process Noise
        self.R = R  # Measurement Noise

        # Estimate and Error Covariance
        self.x_hat = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.eye(self.x_hat.shape[0]) * 0.1

    def update(self, u, y_meas):
        """
        u: Control Input (known voltage)
        y_meas: Noisy Measurements
        """
        # Ensure shapes
        u = np.atleast_2d(u)
        y_meas = np.atleast_2d(y_meas)
        if u.shape[0] == 1 and u.shape[1] != 1:
            u = u.T
        if y_meas.shape[0] == 1 and y_meas.shape[1] != 1:
            y_meas = y_meas.T

        # 1. Predict
        x_pred = self.Phi @ self.x_hat + self.Gamma @ u
        P_pred = self.Phi @ self.P @ self.Phi.T + self.Q

        # 2. Correct
        y_pred = self.C @ x_pred
        y_err = y_meas - y_pred

        # Kalman Gain
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        # Update
        self.x_hat = x_pred + K @ y_err

        # Joseph form update for stability: (I - KC)P(I - KC)' + KRK'
        # Or simple form: (I - KC)P
        I = np.eye(self.x_hat.shape[0])
        self.P = (I - K @ self.C) @ P_pred

        return self.x_hat.flatten()
