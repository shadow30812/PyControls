from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for Non-Linear Parameter Estimation.

    Features:
    - Uses Complex Step Differentiation (CSD) to compute Jacobians numerically
      to machine precision without analytical derivatives.
    - Designed for simultaneous state and parameter estimation.
    """

    def __init__(
        self,
        f_dynamics: Callable[..., NDArray[np.complex128]],
        h_measurement: Callable[[NDArray[Any]], NDArray[np.complex128]],
        Q: ArrayLike,
        R: ArrayLike,
        x0: NDArray[Any],
        p_init_scale: float = 0.1,
    ) -> None:
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
        self.f: Callable[..., NDArray[np.complex128]] = f_dynamics
        self.h: Callable[[NDArray[Any]], NDArray[np.complex128]] = h_measurement
        self.Q: NDArray[np.float64] = np.array(Q, dtype=float)
        self.R: NDArray[np.float64] = np.array(R, dtype=float)

        self.x_hat: NDArray[np.float64] = np.array(x0, dtype=float).reshape(-1, 1)
        self.P: NDArray[np.float64] = np.eye(len(x0)) * p_init_scale
        self.n: int = len(x0)

        self._I_complex: NDArray[np.complex128] = np.eye(self.n, dtype=complex)

    def compute_jacobian(
        self,
        func: Callable[..., NDArray[np.complex128]],
        x: NDArray[np.float64],
        u: Optional[Union[float, NDArray[np.float64]]] = None,
        epsilon: float = 1e-20,
    ) -> NDArray[np.float64]:
        """
        Computes the Jacobian matrix using Vectorized Complex Step Differentiation.

        Optimized to avoid Python loops by evaluating the function on a
        matrix of perturbed states if the function supports it, or looping efficiently.

        Formula: J = Im[f(x + i*h*e_j)] / h
        """
        n_in: int = x.shape[0]

        if self._I_complex.shape[0] != n_in:
            self._I_complex = np.eye(n_in, dtype=complex)

        x_complex: NDArray[np.complex128] = x.astype(complex)
        X_perturb: NDArray[np.complex128] = x_complex + 1j * epsilon * self._I_complex

        try:
            if u is not None:
                Y_perturb: NDArray[np.complex128] = func(X_perturb, u)
            else:
                Y_perturb: NDArray[np.complex128] = func(X_perturb)

            J: NDArray[np.float64] = Y_perturb.imag / epsilon
            return J

        except (TypeError, ValueError, AttributeError):
            x_perturb: NDArray[np.complex128] = x_complex.copy()
            x_perturb[0] += 1j * epsilon

            if u is not None:
                y0: NDArray[np.complex128] = func(x_perturb, u)
            else:
                y0: NDArray[np.complex128] = func(x_perturb)

            n_out: int = len(y0)
            J: NDArray[np.float64] = np.zeros((n_out, n_in))

            J[:, 0] = y0.imag.flatten() / epsilon

            for i in range(1, n_in):
                x_perturb = x_complex.copy()
                x_perturb[i] += 1j * epsilon

                if u is not None:
                    val: NDArray[np.complex128] = func(x_perturb, u)
                else:
                    val: NDArray[np.complex128] = func(x_perturb)

                J[:, i] = val.imag.flatten() / epsilon

            return J

    def predict(self, u: Union[float, NDArray[np.float64]], dt: float) -> None:
        """
        Performs the Time Update (Prediction) step.
        """
        x_dot: NDArray[np.float64] = self.f(self.x_hat.astype(complex), u).real
        self.x_pred: NDArray[np.float64] = self.x_hat + x_dot * dt

        A_c: NDArray[np.float64] = self.compute_jacobian(self.f, self.x_hat, u)
        F: NDArray[np.float64] = np.eye(self.n) + A_c * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, y_meas: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Performs the Measurement Update (Correction) step.
        """
        H: NDArray[np.float64] = self.compute_jacobian(
            lambda x: self.h(x), self.x_hat, u=None
        )

        y_pred: NDArray[np.float64] = self.h(self.x_pred).real
        y_err: NDArray[np.float64] = y_meas - y_pred

        S: NDArray[np.float64] = H @ self.P @ H.T + self.R

        try:
            K: NDArray[np.float64] = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(
                "Np LinAlg error in core/ekf/ExtendedKalmanFilter/update",
            )
            K: NDArray[np.float64] = np.zeros((self.n, y_meas.shape[0]))

        self.x_hat = self.x_pred + K @ y_err

        I: NDArray[np.float64] = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x_hat.flatten()
