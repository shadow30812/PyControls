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

        # Sigma Point Parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights generation
        self._compute_weights()

    def _compute_weights(self):
        """Pre-computes weights for mean and covariance reconstruction."""
        # Number of sigma points = 2n + 1
        num_sigmas = 2 * self.n + 1
        self.Wm = np.full(num_sigmas, 0.5 / (self.n + self.lam))
        self.Wc = np.full(num_sigmas, 0.5 / (self.n + self.lam))

        # Center point weights
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)

    def _generate_sigma_points(self, x, P):
        """Generates 2n+1 sigma points based on current state and covariance."""
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x

        # S = sqrt( (n + lambda) * P ) -> Cholesky Decomposition
        try:
            S = np.linalg.cholesky((self.n + self.lam) * P)
        except np.linalg.LinAlgError:
            # Fallback if P is not positive definite (numerical error)
            # Add small epsilon to diagonal
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
        # 1. Generate Sigma Points
        self.sigmas_f = self._generate_sigma_points(self.x, self.P)

        # 2. Propagate Sigma Points
        self.sigmas_h = np.zeros_like(self.sigmas_f)
        for i, s in enumerate(self.sigmas_f):
            self.sigmas_h[i] = self.f(s, u, dt)

        # 3. Compute Predicted Mean
        x_pred = np.dot(self.Wm, self.sigmas_h)

        # 4. Compute Predicted Covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            y = self.sigmas_h[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred
        # Update sigma points for measurement step
        self.sigmas_f = self.sigmas_h

    def update(self, z):
        """
        Measurement Update Step: Maps sigma points to measurement space.
        """
        # 1. Transform Sigma Points into Measurement Space
        num_sigmas = 2 * self.n + 1
        Z_sigmas = np.zeros((num_sigmas, self.m))

        for i, s in enumerate(self.sigmas_f):
            Z_sigmas[i] = self.h(s)

        # 2. Predicted Measurement Mean
        z_pred = np.dot(self.Wm, Z_sigmas)

        # 3. Innovation Covariance (S) and Cross Covariance (Pxz)
        S = np.zeros((self.m, self.m))
        Pxz = np.zeros((self.n, self.m))

        for i in range(num_sigmas):
            # Residuals
            dz = Z_sigmas[i] - z_pred
            dx = self.sigmas_f[i] - self.x

            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        S += self.R

        # 4. Kalman Gain
        K = np.dot(Pxz, np.linalg.inv(S))

        # 5. State Update
        y = z - z_pred
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(S, K.T))

        return self.x
