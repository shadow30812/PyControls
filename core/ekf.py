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

        self._I_complex = np.eye(self.n, dtype=complex)

    def compute_jacobian(self, func, x, u=None, epsilon=1e-20):
        """
        Computes the Jacobian matrix using Vectorized Complex Step Differentiation.

        Optimized to avoid Python loops by evaluating the function on a
        matrix of perturbed states if the function supports it, or looping efficiently.

        Formula: J = Im[f(x + i*h*e_j)] / h
        """
        n_in = x.shape[0]

        x_complex = x.astype(complex)
        X_perturb = x_complex + 1j * epsilon * self._I_complex

        try:
            if u is not None:
                raise TypeError("Vectorization fallback")
            else:
                Y_perturb = func(X_perturb)
                J = Y_perturb.imag / epsilon
                return J

        except (TypeError, ValueError, AttributeError):
            x_perturb = x_complex.copy()
            x_perturb[0] += 1j * epsilon

            if u is not None:
                y0 = func(x_perturb, u)
            else:
                y0 = func(x_perturb)

            n_out = len(y0)
            J = np.zeros((n_out, n_in))

            J[:, 0] = y0.imag.flatten() / epsilon

            for j in range(1, n_in):
                x_perturb = x_complex.copy()
                x_perturb[j] += 1j * epsilon

                if u is not None:
                    val = func(x_perturb, u)
                else:
                    val = func(x_perturb)

                J[:, j] = val.imag.flatten() / epsilon

            return J

    def predict(self, u, dt):
        """
        Performs the Time Update (Prediction) step.
        """
        # 1. Non-linear state propagation
        # We use the real part for the state update
        x_dot = self.f(self.x_hat.astype(complex), u).real
        self.x_pred = self.x_hat + x_dot * dt

        # 2. Linearization (Jacobian F)
        # F = I + A_c * dt
        A_c = self.compute_jacobian(self.f, self.x_hat, u)
        F = np.eye(self.n) + A_c * dt

        # 3. Covariance Propagation
        # P = F P F.T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y_meas):
        """
        Performs the Measurement Update (Correction) step.
        """
        # 1. Linearize measurement function (Jacobian H)
        H = self.compute_jacobian(lambda x: self.h(x), self.x_hat, u=None)

        # 2. Innovation
        y_pred = self.h(self.x_hat).real
        y_err = y_meas - y_pred

        # 3. Kalman Gain
        S = H @ self.P @ H.T + self.R

        # Use solve instead of inv for numerical stability: K = P @ H.T @ inv(S)
        # K = (solve(S.T, (P @ H.T).T)).T  ... simpler:
        # K = P H^T S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback if S is singular (rare with proper R)
            K = np.zeros((self.n, y_meas.shape[0]))

        # 4. State Update
        self.x_hat = self.x_pred + K @ y_err

        # 5. Covariance Update
        # Joseph Form for stability: P = (I - KH)P(I - KH)^T + KRK^T
        # But standard form is faster: P = (I - KH)P
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x_hat.flatten()
