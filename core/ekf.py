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

        if self._I_complex.shape[0] != n_in:
            self._I_complex = np.eye(n_in, dtype=complex)

        x_complex = x.astype(complex)
        X_perturb = x_complex + 1j * epsilon * self._I_complex

        try:
            try:
                if u is not None:
                    Y_perturb = func(X_perturb, u)
                else:
                    Y_perturb = func(X_perturb)

                J = Y_perturb.imag / epsilon
                return J

            except Exception as e:
                print(
                    "Error in core/ekf/ExtendedKalmanFilter/compute_jacobian",
                    e,
                    sep="\n",
                )

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
        x_dot = self.f(self.x_hat.astype(complex), u).real
        self.x_pred = self.x_hat + x_dot * dt

        A_c = self.compute_jacobian(self.f, self.x_hat, u)
        F = np.eye(self.n) + A_c * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, y_meas):
        """
        Performs the Measurement Update (Correction) step.
        """
        H = self.compute_jacobian(lambda x: self.h(x), self.x_hat, u=None)

        y_pred = self.h(self.x_pred).real
        y_err = y_meas - y_pred

        S = H @ self.P @ H.T + self.R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(
                "Np LinAlg error in core/ekf/ExtendedKalmanFilter/update",
            )
            K = np.zeros((self.n, y_meas.shape[0]))

        self.x_hat = self.x_pred + K @ y_err

        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x_hat.flatten()
