from typing import Any, Callable, Final, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for Non-Linear Estimation.

    Uses the Unscented Transform (Sigma Points) to propagate mean and covariance
    through non-linear functions without linearization (Jacobians).
    """

    def __init__(
        self,
        f_dynamics: Callable[
            [NDArray[np.float64], Union[float, NDArray[np.float64]], float],
            NDArray[np.float64],
        ],
        h_measurement: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        P0: ArrayLike,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        """
        Args:
            f_dynamics: Function f(x, u, dt) -> x_next
            h_measurement: Function h(x) -> y_pred
            Q, R: Process and Measurement Noise Covariances
            x0: Initial State
            P0: Initial Covariance
            alpha, beta, kappa: UKF Scaling parameters (Van der Merwe)
        """
        self.f: Callable[
            [NDArray[np.float64], Union[float, NDArray[np.float64]], float],
            NDArray[np.float64],
        ] = f_dynamics
        self.h: Callable[[NDArray[np.float64]], NDArray[np.float64]] = h_measurement
        self.Q: NDArray[np.float64] = np.array(Q, dtype=float)
        self.R: NDArray[np.float64] = np.array(R, dtype=float)
        self.x: NDArray[np.float64] = np.array(x0, dtype=float)
        self.P: NDArray[np.float64] = np.array(P0, dtype=float)

        self.n: Final[int] = len(self.x)
        self.m: Final[int] = len(self.R)

        self.alpha: Final[float] = alpha
        self.beta: Final[float] = beta
        self.kappa: Final[float] = kappa
        self.lam: Final[float] = self.alpha**2 * (self.n + self.kappa) - self.n

        self._compute_weights()

    def _compute_weights(self) -> None:
        """Pre-computes weights for mean and covariance reconstruction."""
        num_sigmas: int = 2 * self.n + 1
        self.Wm: NDArray[np.float64] = np.full(num_sigmas, 0.5 / (self.n + self.lam))
        self.Wc: NDArray[np.float64] = np.full(num_sigmas, 0.5 / (self.n + self.lam))

        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)

    def _generate_sigma_points(
        self, x: NDArray[np.float64], P: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Generates 2n+1 sigma points based on current state and covariance."""
        sigmas: NDArray[np.float64] = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x

        try:
            S: NDArray[np.floating] = np.linalg.cholesky((self.n + self.lam) * P)
        except np.linalg.LinAlgError:
            P_stab: NDArray[np.float64] = P + np.eye(self.n) * 1e-6
            S: NDArray[np.floating] = np.linalg.cholesky((self.n + self.lam) * P_stab)

        for i in range(self.n):
            sigmas[i + 1] = x + S[i]
            sigmas[self.n + i + 1] = x - S[i]

        return sigmas

    def predict(self, u: Union[float, NDArray[np.float64]], dt: float) -> None:
        """
        Time Update Step: Propagates sigma points through f(x).
        """
        self.sigmas_f: NDArray[np.float64] = self._generate_sigma_points(
            self.x,
            self.P,
        )

        self.sigmas_h: NDArray[np.float64] = np.zeros_like(self.sigmas_f)
        for i, s in enumerate(self.sigmas_f):
            self.sigmas_h[i] = self.f(s, u, dt)

        x_pred: NDArray[np.float64] = np.dot(self.Wm, self.sigmas_h)

        Y: NDArray[np.float64] = self.sigmas_h - x_pred
        P_pred: NDArray[np.float64] = (
            self.Wc[:, None, None] * (Y[:, :, None] * Y[:, None, :])
        ).sum(axis=0)
        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred
        self.sigmas_f = self.sigmas_h

    def update(self, z: Union[ArrayLike, NDArray[Any]]) -> NDArray[np.float64]:
        """
        Measurement Update Step: Maps sigma points to measurement space.
        """
        z = np.asarray(z)

        if z.shape[0] != self.m:
            raise ValueError(
                f"Measurement dimension mismatch: expected {self.m}, got {z.shape[0]}"
            )

        test_z: NDArray[np.float64] = self.h(self.sigmas_f[0])
        if np.asarray(test_z).shape[0] != self.m:
            raise ValueError("Measurement function output dimension mismatch")

        num_sigmas: int = 2 * self.n + 1
        Z_sigmas: NDArray[np.float64] = np.zeros((num_sigmas, self.m))

        for i, s in enumerate(self.sigmas_f):
            Z_sigmas[i] = self.h(s)

        z_pred: NDArray[np.float64] = np.dot(self.Wm, Z_sigmas)

        S: NDArray[np.float64] = np.zeros((self.m, self.m))
        Pxz: NDArray[np.float64] = np.zeros((self.n, self.m))

        DZ: NDArray[np.float64] = Z_sigmas - z_pred
        DX: NDArray[np.float64] = self.sigmas_f - self.x

        S = (self.Wc[:, None, None] * (DZ[:, :, None] * DZ[:, None, :])).sum(axis=0)
        S += self.R

        Pxz = (self.Wc[:, None, None] * (DX[:, :, None] * DZ[:, None, :])).sum(axis=0)
        S += self.R

        K: NDArray[np.float64] = np.dot(Pxz, np.linalg.inv(S))

        y: NDArray[np.float64] = z - z_pred
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(S, K.T))

        return self.x
