from typing import Callable, Final, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.math_utils import jacobian
from core.solver import manual_matrix_exp


class DiscreteExtendedKalmanFilter:
    """
    Textbook discrete-time Extended Kalman Filter.
    """

    def __init__(
        self,
        f: Callable[
            [NDArray[np.float64], Optional[Union[float, NDArray[np.float64]]]],
            NDArray[np.float64],
        ],
        h: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Q: ArrayLike,
        R: ArrayLike,
        x0: ArrayLike,
        dt: float,
    ) -> None:
        self.f: Callable[
            [NDArray[np.float64], Optional[Union[float, NDArray[np.float64]]]],
            NDArray[np.float64],
        ] = f
        self.h: Callable[[NDArray[np.float64]], NDArray[np.float64]] = h
        self.Q: NDArray[np.float64] = np.asarray(Q, dtype=float)
        self.R: NDArray[np.float64] = np.asarray(R, dtype=float)
        self.x: NDArray[np.float64] = np.atleast_2d(x0).astype(float).T
        self.P: NDArray[np.float64] = np.eye(self.x.shape[0])
        self.dt: Final[float] = dt

    def predict(self, u: Optional[Union[float, NDArray[np.float64]]] = None) -> None:
        x_flat: NDArray[np.float64] = self.x.flatten()

        A: NDArray[np.float64] = jacobian(lambda x: self.f(x, u), x_flat)
        Phi: NDArray[np.float64] = manual_matrix_exp(A * self.dt)

        self.x = Phi @ self.x
        self.P = Phi @ self.P @ Phi.T + self.Q

    def update(self, y: Union[ArrayLike, NDArray]) -> None:
        y = np.atleast_2d(y).T
        x_flat: NDArray[np.float64] = self.x.flatten()

        H: NDArray[np.float64] = jacobian(self.h, x_flat)

        S: NDArray[np.float64] = H @ self.P @ H.T + self.R
        K: NDArray[np.float64] = self.P @ H.T @ np.linalg.inv(S)

        y_err: NDArray[np.float64] = y - self.h(x_flat).reshape(-1, 1)

        I: NDArray[np.float64] = np.eye(self.P.shape[0])
        self.x = self.x + K @ y_err
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
