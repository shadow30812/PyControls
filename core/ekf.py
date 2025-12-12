import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for Non-Linear Parameter Estimation.

    Features:
    - Uses Complex Step Differentiation (CSD) to compute Jacobians numerically
      to machine precision without analytical derivatives.
    - Designed for simultaneous state and parameter estimation.
    """

    def __init__(self, f_dynamics, h_measurement, Q, R, x0, p_init_scale=0.1):
        """
        Initializes the EKF.

        Args:
            f_dynamics: The state transition function f(x, u). Must support complex arguments.
            h_measurement: The measurement function h(x). Must support complex arguments.
            Q: Process Noise Covariance Matrix (n x n).
            R: Measurement Noise Covariance Matrix (m x m).
            x0: Initial state vector (n x 1).
            p_init_scale: Scalar multiplier for the initial identity P matrix.
        """
        self.f = f_dynamics
        self.h = h_measurement
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.x_hat = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.eye(len(x0)) * p_init_scale
        self.n = len(x0)

    def compute_jacobian(self, func, x, u=None, epsilon=1e-20):
        """
        Computes the Jacobian matrix using Complex Step Differentiation.

        Formula: J = Im[f(x + i*h)] / h

        Args:
            func: The function to differentiate.
            x: The point at which to differentiate.
            u: Optional input vector.
            epsilon: The complex perturbation step size.

        Returns:
            J: The computed Jacobian matrix.
        """
        n_out = len(func(x, u)) if u is not None else len(func(x))
        n_in = len(x)
        J = np.zeros((n_out, n_in))

        x_complex = np.array(x, dtype=complex)

        for j in range(n_in):
            x_perturb = x_complex.copy()
            x_perturb[j] += 1j * epsilon

            if u is not None:
                f_val = func(x_perturb, u)
            else:
                f_val = func(x_perturb)

            J[:, j] = f_val.imag.flatten() / epsilon

        return J

    def predict(self, u, dt):
        """
        Performs the Time Update (Prediction) step.

        1. Predicts the next state using the non-linear dynamics f(x, u).
        2. Linearizes f(x, u) to find the transition matrix F.
        3. Propagates the error covariance P.
        """
        x_dot = self.f(self.x_hat.astype(complex), u).real
        self.x_pred = self.x_hat + x_dot * dt

        A_c = self.compute_jacobian(self.f, self.x_hat, u)
        F = np.eye(self.n) + A_c * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, y_meas):
        """
        Performs the Measurement Update (Correction) step.

        1. Linearizes h(x) to find the measurement matrix H.
        2. Computes the Innovation (Measurement Residual).
        3. Computes the Kalman Gain K.
        4. Updates the state estimate x_hat and covariance P.
        """
        H = self.compute_jacobian(lambda x: self.h(x), self.x_hat, u=None)

        y_pred = self.h(self.x_hat).real
        y_err = y_meas - y_pred

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x_hat = self.x_pred + K @ y_err
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x_hat.flatten()
