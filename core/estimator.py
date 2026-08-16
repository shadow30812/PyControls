from typing import Any, Dict, Final, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.base import BaseEstimator


class KalmanFilter(BaseEstimator):
    """
    Standard Linear Discrete-Time Kalman Filter.
    Estimates the state x of a linear system from noisy measurements y.

    System Model:
    x[k+1] = Phi * x[k] + Gamma * u[k] + w[k]
    y[k]   = C * x[k] + v[k]
    """

    def __init__(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        C: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        x0: ArrayLike,
    ) -> None:
        """
        Args:
            A (Phi): State transition matrix (Discrete).
            B (Gamma): Input control matrix (Discrete).
            C: Measurement matrix.
            Q: Process noise covariance.
            R: Measurement noise covariance.
            x0: Initial state guess.
        """
        self.Phi: Final[NDArray[np.float64]] = A
        self.Gamma: Final[NDArray[np.float64]] = B
        self.C: Final[NDArray[np.float64]] = C

        self.Q: Final[NDArray[np.float64]] = Q
        self.R: Final[NDArray[np.float64]] = R

        self.x_hat: NDArray[np.float64] = np.array(x0, dtype=float).reshape(-1, 1)
        self.P: NDArray[np.float64] = np.eye(self.x_hat.shape[0]) * 0.1

    def predict(
        self, u: Union[ArrayLike, NDArray[Any]], _: Optional[float] = None
    ) -> None:
        """
        Performs the a priori prediction step.
        x[k|k-1] = Phi * x[k-1|k-1] + Gamma * u[k]
        P[k|k-1] = Phi * P[k-1|k-1] * Phi' + Q

        Args:
            u: Control input vector.
            _: Time step [dt] (unused here as Phi/Gamma are already discrete,
               but kept for interface consistency with EKF/UKF).
        """
        u = np.atleast_2d(u)
        if u.shape[0] == 1 and u.shape[1] != 1:
            u = u.T

        self.x_hat = self.Phi @ self.x_hat + self.Gamma @ u
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q

    def update(self, y_meas: Union[ArrayLike, NDArray[Any]]) -> NDArray[np.float64]:
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

        y_pred: NDArray[np.float64] = self.C @ self.x_hat
        y_err: NDArray[np.float64] = y_meas - y_pred

        S: NDArray[np.float64] = self.C @ self.P @ self.C.T + self.R
        K: NDArray[np.float64] = self.P @ self.C.T @ np.linalg.inv(S)

        self.x_hat = self.x_hat + K @ y_err

        I: NDArray[np.float64] = np.eye(self.x_hat.shape[0])
        self.P = (I - K @ self.C) @ self.P

        return self.x_hat.flatten()

    def get_state(self) -> NDArray[np.float64]:
        """Returns the current state estimate as a flat array."""
        return self.x_hat.flatten()

    def get_covariance(self) -> NDArray[np.float64]:
        """Returns a copy of the current error covariance matrix."""
        return self.P.copy()

    def reset(self, x0: ArrayLike, P0: Optional[ArrayLike] = None) -> None:
        """Resets filter state. Useful for Simulink scenario restarts."""
        self.x_hat = np.array(x0, dtype=float).reshape(-1, 1)
        if P0 is not None:
            self.P = np.array(P0, dtype=float)
        else:
            self.P = np.eye(self.x_hat.shape[0]) * 0.1
