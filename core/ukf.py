import numpy as np


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for Non-Linear Estimation.

    Uses the Unscented Transform (Sigma Points) to propagate mean and covariance
    through non-linear functions without linearization (Jacobians).
    """

    def __init__(
        self, f_dynamics, h_measurement, Q, R, x0, P0, alpha=1e-3, beta=2.0, kappa=0.0
    ):
        """
        Args:
            f_dynamics: Function f(x, u, dt) -> x_next
            h_measurement: Function h(x) -> y_pred
            Q, R: Process and Measurement Noise Covariances
            x0: Initial State
            P0: Initial Covariance
            alpha, beta, kappa: UKF Scaling parameters (Van der Merwe)
        """
        self.f = f_dynamics
        self.h = h_measurement
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

        self.n = len(x0)
        self.m = len(self.R)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        self._compute_weights()

    def _compute_weights(self):
        """Pre-computes weights for mean and covariance reconstruction."""
        num_sigmas = 2 * self.n + 1
        self.Wm = np.full(num_sigmas, 0.5 / (self.n + self.lam))
        self.Wc = np.full(num_sigmas, 0.5 / (self.n + self.lam))

        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)

    def _generate_sigma_points(self, x, P):
        """Generates 2n+1 sigma points based on current state and covariance."""
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x

        try:
            S = np.linalg.cholesky((self.n + self.lam) * P)
        except np.linalg.LinAlgError:
            P_stab = P + np.eye(self.n) * 1e-6
            S = np.linalg.cholesky((self.n + self.lam) * P_stab)

        for i in range(self.n):
            sigmas[i + 1] = x + S[i]
            sigmas[self.n + i + 1] = x - S[i]

        return sigmas

    def predict(self, u, dt):
        """
        Time Update Step: Propagates sigma points through f(x).
        """
        self.sigmas_f = self._generate_sigma_points(self.x, self.P)

        self.sigmas_h = np.zeros_like(self.sigmas_f)
        for i, s in enumerate(self.sigmas_f):
            self.sigmas_h[i] = self.f(s, u, dt)

        x_pred = np.dot(self.Wm, self.sigmas_h)

        Y = self.sigmas_h - x_pred
        P_pred = (self.Wc[:, None, None] * (Y[:, :, None] * Y[:, None, :])).sum(axis=0)
        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred
        self.sigmas_f = self.sigmas_h

    def update(self, z):
        """
        Measurement Update Step: Maps sigma points to measurement space.
        """
        z = np.asarray(z)

        if z.shape[0] != self.m:
            raise ValueError(
                f"Measurement dimension mismatch: expected {self.m}, got {z.shape[0]}"
            )

        test_z = self.h(self.sigmas_f[0])
        if np.asarray(test_z).shape[0] != self.m:
            raise ValueError("Measurement function output dimension mismatch")

        num_sigmas = 2 * self.n + 1
        Z_sigmas = np.zeros((num_sigmas, self.m))

        for i, s in enumerate(self.sigmas_f):
            Z_sigmas[i] = self.h(s)

        z_pred = np.dot(self.Wm, Z_sigmas)

        S = np.zeros((self.m, self.m))
        Pxz = np.zeros((self.n, self.m))

        DZ = Z_sigmas - z_pred
        DX = self.sigmas_f - self.x

        S = (self.Wc[:, None, None] * (DZ[:, :, None] * DZ[:, None, :])).sum(axis=0)
        S += self.R

        Pxz = (self.Wc[:, None, None] * (DX[:, :, None] * DZ[:, None, :])).sum(axis=0)
        S += self.R

        K = np.dot(Pxz, np.linalg.inv(S))

        y = z - z_pred
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(S, K.T))

        return self.x
