from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TransferFunction:
    """
    Representation of a Single-Input Single-Output (SISO) Transfer Function.
    G(s) = Num(s) / Den(s)
    """

    def __init__(self, num: ArrayLike, den: ArrayLike) -> None:
        self.num: NDArray[np.float64] = np.array(num, dtype=float)
        self.den: NDArray[np.float64] = np.array(den, dtype=float)
        self.repr_num: ArrayLike = num
        self.repr_den: ArrayLike = den

    def __repr__(self) -> str:
        """
        String representation of the transfer function.
        """
        return f"TF(Num={self.repr_num}, Den={self.repr_den})"

    def evaluate(
        self, s: Union[complex, float, NDArray]
    ) -> NDArray[np.float64] | float:
        """Evaluates G(s) at a complex number s using Horner's method (via np.polyval)."""
        n_val: Union[complex, float, NDArray] = np.polyval(self.num, s)
        d_val: Union[complex, float, NDArray] = np.polyval(self.den, s)
        return n_val / d_val if d_val != 0 else np.inf

    def bode_response(
        self, omega_range: ArrayLike
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculates Magnitude (dB) and Phase (deg) over a frequency range."""
        omega: NDArray[np.float64] = np.asarray(omega_range)
        mags: NDArray[np.float64] = np.empty_like(omega, dtype=float)
        phases: NDArray[np.float64] = np.empty_like(omega, dtype=float)

        for k, w in enumerate(omega):
            s: complex = 1j * w
            resp: Union[complex, float, NDArray] = self.evaluate(s)
            mags[k] = 20.0 * np.log10(np.abs(resp))
            phases[k] = np.degrees(np.angle(resp))

        return mags, phases

    def to_state_space(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64] | int,
    ]:
        """
        Converts the SISO Transfer Function to State-Space Control Canonical Form.

        Returns:
            tuple: (A, B, C, D) matrices.
        """
        norm: float = self.den[0]
        a: NDArray[np.float64] = self.den / norm
        b: NDArray[np.float64] = self.num / norm

        n: int = len(a) - 1
        if len(b) < len(a):
            b = np.pad(b, (len(a) - len(b), 0), "constant")

        A: NDArray[np.float64] = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1
        A[n - 1, :] = -a[1:][::-1]

        B: NDArray[np.float64] = np.zeros((n, 1))
        B[n - 1, 0] = 1

        x: NDArray[np.float64] = b[1:][::-1] - b[0] * a[1:][::-1]
        C: NDArray[np.float64] = x.reshape(1, n)
        D: Union[int, NDArray[np.float64]] = b[0]

        return A, B, C, D
